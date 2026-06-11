//! `rustcox cells <TYPE> [options]` — left cells by parabolic induction.
//!
//! Computes the left-cell partition of a finite Coxeter group with the
//! `klcells` driver (parabolic induction; equal parameters).  Three output
//! modes, composable:
//!
//! - default / `--summary`: a one-line summary to stdout.
//! - `-o FILE`: a whole-document canonical `kind: "cells"` JSON (small groups
//!   only — the whole partition is held in RAM and canonicalized).
//! - `--stream FILE.jsonl.gz`: streaming JSON-lines, one record per cell, never
//!   holding all cells in RAM (the E8-scale path).  With `--checkpoint-dir DIR`
//!   the run is checkpoint/resume-safe; with `--save-reps DIR` the star-rep
//!   W-graphs are written to `DIR/reps/NNNNNN.json.gz`.
//!
//! ## Resume / kill-safety
//!
//! `--stream` + `--checkpoint-dir` auto-resumes: on start it loads a matching
//! checkpoint, truncates the stream file to the checkpointed record count, and
//! replays from the checkpointed rep.  SIGTERM/SIGINT are not trapped (no
//! `unsafe`, no extra deps); instead the driver checkpoints after every rep, so
//! a kill at any moment loses at most one rep of work and the next resubmit
//! recovers it.

use std::cell::RefCell;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use flate2::{write::GzEncoder, Compression};
use rustcox_core::{
    group::CoxeterGroup,
    io::{cellgraph_json, cells_json_doc},
    kl::{
        klcells, klcells_streaming_with_flush, CellRecord, CellsOpts, CheckpointCfg, FlushSink,
        RepsSink,
    },
};

use super::kl::write_json_doc;

pub struct CellsArgs {
    pub type_str: String,
    pub threads: Option<usize>,
    pub summary: bool,
    pub output: Option<String>,
    /// Stream one JSON record per cell to this file (gz if it ends `.gz`).
    pub stream: Option<String>,
    /// Directory holding the checkpoint (auto-resume).  Requires `--stream`.
    pub checkpoint_dir: Option<String>,
    /// Directory to save star-rep W-graphs into (`DIR/reps/NNNNNN.json.gz`).
    pub save_reps: Option<String>,
}

pub fn run(args: CellsArgs) -> Result<()> {
    let group = CoxeterGroup::from_type(&args.type_str)
        .with_context(|| format!("invalid type '{}'", args.type_str))?;

    let opts = CellsOpts {
        all_cells: true,
        threads: args.threads,
    };

    if args.stream.is_some() {
        run_streaming(&group, &opts, &args)
    } else {
        run_in_memory(&group, &opts, &args)
    }
}

// ---------------------------------------------------------------------------
// In-memory mode (small groups): whole-document -o + summary
// ---------------------------------------------------------------------------

fn run_in_memory(group: &CoxeterGroup, opts: &CellsOpts, args: &CellsArgs) -> Result<()> {
    if args.checkpoint_dir.is_some() || args.save_reps.is_some() {
        anyhow::bail!("--checkpoint-dir / --save-reps require --stream");
    }

    let t0 = Instant::now();
    let res = klcells(group, opts).with_context(|| "klcells computation failed")?;
    let elapsed = t0.elapsed();

    let show_summary = args.summary || args.output.is_none();
    if show_summary {
        print_summary(
            group,
            res.cells.len(),
            res.n_star_reps,
            group.order,
            elapsed,
        );
    }

    if let Some(ref out_path) = args.output {
        let doc = cells_json_doc(group, &res);
        write_json_doc(out_path, &doc)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Streaming mode (E8-scale): JSONL stream + checkpoint/resume + reps
// ---------------------------------------------------------------------------

fn run_streaming(group: &CoxeterGroup, opts: &CellsOpts, args: &CellsArgs) -> Result<()> {
    let stream_path = args.stream.as_ref().expect("stream mode requires --stream");
    let ckpt = args
        .checkpoint_dir
        .as_ref()
        .map(|d| CheckpointCfg::new(PathBuf::from(d)));

    // Determine the resume truncation point BEFORE opening the stream file: load
    // the checkpoint (if any) and keep only its `records` prefix.  We compute the
    // keep-count via a dry probe of the summary's `records_kept`, but the driver
    // reports that only after running; so instead we peek the checkpoint here.
    let records_to_keep = match &ckpt {
        Some(cfg) => peek_records_kept(cfg, group, opts.all_cells),
        None => 0,
    };

    // Open the stream writer, truncating to `records_to_keep` lines on resume.
    // Wrap in RefCell so two closures (cell_sink and flush_fn) can share it.
    let stream = RefCell::new(
        StreamWriter::open(stream_path, records_to_keep)
            .with_context(|| format!("opening stream file '{stream_path}'"))?,
    );
    if records_to_keep > 0 {
        eprintln!(
            "resumed: kept {records_to_keep} cell records already on disk in '{stream_path}'"
        );
    }

    // Optional reps directory.
    let reps_dir = args.save_reps.as_ref().map(PathBuf::from);
    if let Some(ref d) = reps_dir {
        std::fs::create_dir_all(d.join("reps"))
            .with_context(|| format!("creating reps dir '{}'", d.display()))?;
    }

    let t0 = Instant::now();

    let mut cell_sink =
        |rec: CellRecord| -> io::Result<()> { stream.borrow_mut().write_record(&rec) };

    // Pre-checkpoint flush: called just before each checkpoint write so the
    // BufWriter/GzEncoder bytes are committed to disk before the checkpoint
    // records count is persisted.  Without this flush a SIGTERM mid-rep could
    // leave fewer recoverable records in the gz than checkpoint.records says.
    let mut flush_fn = || -> io::Result<()> { stream.borrow_mut().flush_inner() };

    // reps sink, only if --save-reps given.
    let reps_dir_ref = reps_dir.clone();
    let mut reps_sink = |idx: usize, cg: &rustcox_core::cellgraph::CellGraph| -> io::Result<()> {
        match reps_dir_ref {
            Some(ref d) => write_rep(d, idx, cg),
            None => Ok(()),
        }
    };

    let summary = {
        let reps_opt: Option<&mut RepsSink<'_>> = if reps_dir.is_some() {
            Some(&mut reps_sink)
        } else {
            None
        };
        let flush_opt: Option<&mut FlushSink<'_>> = Some(&mut flush_fn);
        klcells_streaming_with_flush(group, opts, &mut cell_sink, reps_opt, flush_opt, ckpt.as_ref())
    };

    let summary = summary.with_context(|| "streaming klcells failed")?;

    stream
        .into_inner()
        .finish()
        .with_context(|| "finalizing stream file")?;
    let elapsed = t0.elapsed();

    if let Some(k) = summary.resumed_at_rep {
        eprintln!(
            "resumed at rep {k}; {} cell records on disk before this run",
            summary.records_kept
        );
    }

    // Summary to stdout (always, in streaming mode).
    print_summary(
        group,
        summary.ncells,
        summary.n_star_reps,
        summary.total_elements,
        elapsed,
    );

    Ok(())
}

/// Peek the checkpoint to learn how many stream records to keep on resume.
///
/// Returns `0` when there is no checkpoint or it does not match this run.
fn peek_records_kept(cfg: &CheckpointCfg, group: &CoxeterGroup, all_cells: bool) -> u128 {
    use rustcox_core::kl::{checkpoint, run_fingerprint};
    if !checkpoint::exists(cfg) {
        return 0;
    }
    let fp = run_fingerprint(group, all_cells);
    match checkpoint::load_matching(cfg, &fp) {
        Ok(ck) => ck.records,
        Err(e) => {
            eprintln!("ignoring checkpoint in '{}': {e}", cfg.dir.display());
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Stream writer (JSON-lines, optional gz, resume-truncating)
// ---------------------------------------------------------------------------

/// A JSON-lines stream writer.  On resume it keeps only the first
/// `keep_records` lines of an existing file, then appends new records.
///
/// gz streams cannot be truncated in place, so on resume we decompress, keep the
/// prefix, and rewrite — affordable because resume happens rarely (once per
/// SLURM restart) and the prefix is read once.
enum StreamWriter {
    Plain(BufWriter<std::fs::File>),
    Gz(GzEncoder<BufWriter<std::fs::File>>),
}

impl StreamWriter {
    fn open(path: &str, keep_records: u128) -> Result<StreamWriter> {
        let is_gz = path.ends_with(".gz");
        // Read + retain the prefix to keep (if the file exists and we are
        // resuming), then reopen for (over)write.
        let prefix: Vec<u8> = if keep_records > 0 && Path::new(path).exists() {
            read_prefix_lines(path, is_gz, keep_records)?
        } else {
            Vec::new()
        };

        let file =
            std::fs::File::create(path).with_context(|| format!("cannot create '{path}'"))?;
        if is_gz {
            let mut enc = GzEncoder::new(BufWriter::new(file), Compression::default());
            enc.write_all(&prefix).context("rewriting kept prefix")?;
            Ok(StreamWriter::Gz(enc))
        } else {
            let mut w = BufWriter::new(file);
            w.write_all(&prefix).context("rewriting kept prefix")?;
            Ok(StreamWriter::Plain(w))
        }
    }

    fn write_record(&mut self, rec: &CellRecord) -> io::Result<()> {
        let cell: Vec<Vec<u64>> = rec
            .words
            .iter()
            .map(|w| w.iter().map(|&s| s as u64).collect())
            .collect();
        let v = serde_json::json!({
            "rep": rec.rep_index,
            "orbit": rec.orbit_index,
            "cell": cell,
        });
        let mut line = serde_json::to_vec(&v)?;
        line.push(b'\n');
        match self {
            StreamWriter::Plain(w) => w.write_all(&line),
            StreamWriter::Gz(w) => w.write_all(&line),
        }
    }

    /// Flush the BufWriter (plain) or GzEncoder (gz) to the OS.
    ///
    /// For gz, `GzEncoder::flush()` calls `Write::flush` on the underlying
    /// `BufWriter`, committing all compressed data written so far to the OS
    /// page cache.  This is called before each checkpoint write so the stream
    /// record count on disk matches `checkpoint.records` at that instant.
    fn flush_inner(&mut self) -> io::Result<()> {
        match self {
            StreamWriter::Plain(w) => w.flush(),
            StreamWriter::Gz(w) => w.flush(),
        }
    }

    fn finish(self) -> io::Result<()> {
        match self {
            StreamWriter::Plain(mut w) => w.flush(),
            StreamWriter::Gz(w) => {
                w.finish()?;
                Ok(())
            }
        }
    }
}

/// Read the first `keep` newline-terminated records of `path` (gz-transparent),
/// returning the exact bytes (including trailing newlines) to retain.
///
/// For gz files, the stream may be truncated (no end-of-stream marker) if the
/// process was killed mid-write.  In that case we recover as many complete
/// newline-terminated lines as possible and cap at `keep` — this is always safe
/// because the checkpoint invariant guarantees `keep ≤ actual_records_emitted`,
/// so any complete line recovered from a partial gz is a valid cell record.
fn read_prefix_lines(path: &str, is_gz: bool, keep: u128) -> Result<Vec<u8>> {
    use std::io::BufRead;

    let keep_n = keep.min(usize::MAX as u128) as usize;

    if is_gz {
        // Read line-by-line through the GzDecoder so we can stop at `keep` lines
        // and also tolerate a truncated gz (SIGTERM mid-write leaves no gz trailer).
        // Lines that were fully written but whose containing DEFLATE block was not
        // yet flushed are simply absent — that is fine; the checkpoint records count
        // tells us the exact number of records that are safe to keep.
        let raw = std::fs::read(path).with_context(|| format!("reading '{path}'"))?;
        let dec = flate2::read::GzDecoder::new(raw.as_slice());
        let mut reader = std::io::BufReader::new(dec);
        let mut out = Vec::new();
        let mut line = Vec::new();
        let mut count = 0usize;
        loop {
            if count >= keep_n {
                break;
            }
            line.clear();
            let n = match reader.read_until(b'\n', &mut line) {
                Ok(n) => n,
                Err(_) => break, // truncated gz: stop at the last complete line
            };
            if n == 0 {
                break; // clean EOF
            }
            // Only include lines that are newline-terminated (complete records).
            // A partial last line (written but gz not flushed) is discarded.
            if line.ends_with(b"\n") {
                out.extend_from_slice(&line);
                count += 1;
            }
            // If no trailing newline the gz ran out mid-line — stop here.
            if !line.ends_with(b"\n") {
                break;
            }
        }
        if count < keep_n {
            eprintln!(
                "warning: gz stream in '{path}' is truncated; \
                 recovered {count} of {keep_n} expected records — \
                 the resume will re-emit the missing records"
            );
        }
        Ok(out)
    } else {
        let raw = std::fs::read(path).with_context(|| format!("reading '{path}'"))?;
        let mut out = Vec::new();
        for line in raw.split_inclusive(|&b| b == b'\n').take(keep_n) {
            out.extend_from_slice(line);
        }
        Ok(out)
    }
}

/// Write one star-rep W-graph to `dir/reps/NNNNNN.json.gz`.
fn write_rep(dir: &Path, idx: usize, cg: &rustcox_core::cellgraph::CellGraph) -> io::Result<()> {
    let path = dir.join("reps").join(format!("{idx:06}.json.gz"));
    let doc = cellgraph_json(cg);
    let bytes = serde_json::to_vec(&doc)?;
    let file = std::fs::File::create(&path)?;
    let mut enc = GzEncoder::new(BufWriter::new(file), Compression::default());
    enc.write_all(&bytes)?;
    enc.finish()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

/// Print the one-line cells summary to stdout.
///
/// Format (space-separated `key=value`; `time` last and not stable for
/// scripting):
///
/// ```text
/// ncells=<n> nstarreps=<n> order=<n> time=<seconds>s
/// ```
fn print_summary(
    _group: &CoxeterGroup,
    ncells: usize,
    n_star_reps: usize,
    order: u128,
    elapsed: Duration,
) {
    println!(
        "ncells={} nstarreps={} order={} time={:.3}s",
        ncells,
        n_star_reps,
        order,
        elapsed.as_secs_f64()
    );
}
