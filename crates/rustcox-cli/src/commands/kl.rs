//! `rustcox kl <TYPE> [options]` — compute KL polynomials and cells.

use std::{
    io::{BufWriter, Write},
    path::Path,
    time::{Duration, Instant},
};

use anyhow::Context;
use rustcox_core::{
    group::CoxeterGroup,
    io::to_canonical_json,
    kl::{cells::CellData, klpolynomials, klpolynomials_seq, KlOpts, KlTable},
};

/// Parsed weight specification from `--weights` argument.
pub enum WeightSpec {
    /// All weights equal to this value.
    Uniform(u32),
    /// Explicit per-generator weights.
    List(Vec<u32>),
}

impl WeightSpec {
    /// Parse `--weights` value: either a single integer K (all equal) or a
    /// comma-separated list `w0,w1,...`.
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        if let Ok(k) = s.parse::<u32>() {
            return Ok(WeightSpec::Uniform(k));
        }
        let parts: Result<Vec<u32>, _> = s.split(',').map(|p| p.trim().parse::<u32>()).collect();
        let parts = parts.with_context(|| {
            format!("invalid weights '{s}': expected a single integer or comma-separated integers")
        })?;
        Ok(WeightSpec::List(parts))
    }

    /// Resolve the spec into a weight vector of length `rank`.
    pub fn resolve(&self, rank: usize) -> Vec<u32> {
        match self {
            WeightSpec::Uniform(k) => vec![*k; rank],
            WeightSpec::List(v) => v.clone(),
        }
    }
}

pub struct KlArgs {
    pub type_str: String,
    pub weight_spec: Option<WeightSpec>,
    pub threads: Option<usize>,
    pub layer_chunk: Option<usize>,
    pub summary: bool,
    pub output: Option<String>,
    pub seq: bool,
}

pub fn run(args: KlArgs) -> anyhow::Result<()> {
    let group = CoxeterGroup::from_type(&args.type_str)
        .with_context(|| format!("invalid type '{}'", args.type_str))?;

    let weights = match &args.weight_spec {
        Some(spec) => spec.resolve(group.rank),
        None => vec![1u32; group.rank],
    };

    // Validate weight count before calling KlOpts
    if weights.len() != group.rank {
        anyhow::bail!(
            "weights length {} != rank {} for type '{}'",
            weights.len(),
            group.rank,
            args.type_str
        );
    }

    let opts = KlOpts {
        weights: weights.clone(),
        threads: args.threads,
        layer_chunk: args.layer_chunk,
    };

    opts.validate(&group)
        .with_context(|| "invalid weights for this group")?;

    let t0 = Instant::now();

    let table = if args.seq {
        klpolynomials_seq(&group, &opts).with_context(|| "KL computation failed")?
    } else {
        klpolynomials(&group, &opts).with_context(|| "KL computation failed")?
    };

    let cells = CellData::from_table(&table);
    let elapsed = t0.elapsed();

    let show_summary = args.summary || args.output.is_none();

    if show_summary {
        print_summary(&table, &cells, elapsed);
    }

    if let Some(ref out_path) = args.output {
        let doc = to_canonical_json(&table, &cells, &group);
        write_json_doc(out_path, &doc)?;
    }

    Ok(())
}

/// Print the one-line KL summary to stdout.
///
/// # Summary-line format
///
/// Space-separated `key=value` pairs on a single line.  Stable keys (suitable
/// for scripting and golden comparisons) are always emitted in this order:
///
/// ```text
/// npols=<N> mues=<N> lcells=<N> rcells=<N> tcells=<N> duflo=<N> arrows=<N> checks_ok=<bool>
/// ```
///
/// The `time=<seconds>s` field is **always emitted last** and is **not stable
/// for scripting** — its value depends on hardware and load.
///
/// Key definitions:
/// - `npols`     — number of distinct KL polynomials (including the trivial `1`)
/// - `mues`      — number of non-zero μ-coefficients (leading terms of KL polys)
/// - `lcells`    — number of left cells
/// - `rcells`    — number of right cells
/// - `tcells`    — number of two-sided cells
/// - `duflo`     — number of Duflo involutions
/// - `arrows`    — number of W-graph arrows
/// - `checks_ok` — whether all internal consistency checks passed
/// - `time`      — wall time in seconds (last, unstable for scripting)
pub fn print_summary(table: &KlTable, cells: &CellData, elapsed: Duration) {
    let npols = table.pols.len();
    let mues = table.mu_count();
    let lcells = cells.lcells.len();
    let rcells = cells.rcells.len();
    let tcells = cells.tcells.len();
    let duflo = cells.duflo.len();
    let arrows = cells.arrows.len();
    let checks_ok = cells.checks_ok;
    println!(
        "npols={npols} mues={mues} lcells={lcells} rcells={rcells} tcells={tcells} \
         duflo={duflo} arrows={arrows} checks_ok={checks_ok} \
         time={:.3}s",
        elapsed.as_secs_f64()
    );
}

/// Write a `serde_json::Value` to a file; gzip if the path ends with `.gz`.
pub fn write_json_doc(path: &str, doc: &serde_json::Value) -> anyhow::Result<()> {
    let bytes = serde_json::to_vec(doc).context("JSON serialization failed")?;
    let p = Path::new(path);

    if path.ends_with(".gz") {
        use flate2::{write::GzEncoder, Compression};
        let file = std::fs::File::create(p).with_context(|| format!("cannot create '{path}'"))?;
        let mut enc = GzEncoder::new(BufWriter::new(file), Compression::default());
        enc.write_all(&bytes)
            .with_context(|| format!("gz write failed for '{path}'"))?;
        enc.finish()
            .with_context(|| format!("gz finish failed for '{path}'"))?;
    } else {
        std::fs::write(p, &bytes).with_context(|| format!("cannot write '{path}'"))?;
    }

    Ok(())
}

/// Read a JSON document from a file; gzip-transparent.
pub fn read_json_doc(path: &str) -> anyhow::Result<serde_json::Value> {
    let p = Path::new(path);
    let bytes = std::fs::read(p).with_context(|| format!("cannot read '{path}'"))?;

    let value = if path.ends_with(".gz") {
        use flate2::read::GzDecoder;
        use std::io::Read;
        let mut dec = GzDecoder::new(bytes.as_slice());
        let mut out = Vec::new();
        dec.read_to_end(&mut out)
            .with_context(|| format!("gz decode failed for '{path}'"))?;
        serde_json::from_slice(&out).with_context(|| format!("JSON parse failed for '{path}'"))?
    } else {
        serde_json::from_slice(&bytes).with_context(|| format!("JSON parse failed for '{path}'"))?
    };

    Ok(value)
}
