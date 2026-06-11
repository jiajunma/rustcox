//! Layer-granular checkpoint/resume for [`relklpols`](super::relklpols) (Task Q4).
//!
//! A single `relklpols` call for a monster induction step (E8-scale) can run for
//! hours or days — longer than one SLURM box.  The driver-level checkpoint (Task
//! Q1, [`checkpoint`](super::checkpoint)) snapshots *between* reps, but cannot
//! help when one rep alone exceeds the box: E8's per-rep checkpoints would
//! livelock, re-running the same monster rep from scratch on every resume.
//!
//! This module logs the relkl wavefront **layer by layer** so a monster call can
//! be paused and resumed at a layer boundary.  It is purely additive: it never
//! touches the driver checkpoint binary format or fingerprint semantics — only
//! new files (`<rep_tag>.blklog` + `<rep_tag>.blkhdr`) are written.
//!
//! # The wavefront shape it persists
//!
//! [`relklpols`](super::relklpols) computes layers `y = 0..|X1|`.  Layer `y`'s
//! blocks `(y, x)` (with `x < y`) depend only on *frozen* lower layers
//! (first-index `< y`) plus the cell diagonal `(0, 0)`; once a layer completes,
//! its blocks are immutable, and the two pools (`rklpols`, `mues`) only grow
//! (two-phase intern, P6).  So after each completed layer we can append:
//!
//! 1. the finalized **off-diagonal** blocks `(y, x)` of that layer (the diagonal
//!    `(y, y)` block and the initial `Pending`/`Absent` grids are recomputed
//!    deterministically by the setup phase on resume, so they are NOT logged);
//! 2. the **pool deltas** — entries appended to `rklpols` and `mues` *during this
//!    layer's wavefront* (the setup-seeded `mues` prefix is recomputed on resume,
//!    so only the wavefront growth is logged).
//!
//! On resume the setup phase runs fresh (cheap, deterministic), the log is
//! replayed to overwrite the finalized blocks and re-grow the pools, and the
//! wavefront continues at `last_layer + 1`.
//!
//! # On-disk format
//!
//! Two files per in-flight rep, both **versioned, little-endian**, hand-rolled
//! (same rationale as [`checkpoint`](super::checkpoint)):
//!
//! - `<rep_tag>.blklog` — an append-only sequence of framed layer records:
//!   ```text
//!   per record:
//!     4   record magic = b"RKL\n"
//!     4   layer index y (u32 LE)
//!     8   record payload byte length P (u64 LE)  — frames the body
//!     P   payload:
//!         4         block count B (u32 LE)
//!         B× block:
//!           4       x          (u32 LE)
//!           4       nc         (u32 LE)  — grid is nc×nc
//!           nc*nc×  slot:  1 tag byte + (if Done) 4 rk + 4 mu (all LE)
//!         4         rklpols delta count Dr (u32 LE)
//!         Dr× laurent (see `write_laurent`)
//!         4         mues delta count Dm (u32 LE)
//!         Dm× laurent
//!   ```
//!   Records are appended then `fsync`ed; a crash mid-append leaves a trailing
//!   partial record that resume ignores via the header's byte length.
//!
//! - `<rep_tag>.blkhdr` — a tiny side header, rewritten atomically (tmp+rename)
//!   after each record's bytes are durable:
//!   ```text
//!     8   magic = b"RKLHDR\0\0"
//!     4   format version (u32 LE) = BLK_VERSION
//!     4   fingerprint byte length F (u32 LE)
//!     F   fingerprint (UTF-8: group + W1 + cell1 content hash)
//!     4   last completed layer (u32 LE)
//!     8   record count (u64 LE)
//!     8   log byte length (u64 LE)  — truncation-recovery bound
//!     8   rklpols length after last layer (u64 LE)  — replay sanity
//!     8   mues length after last layer (u64 LE)      — replay sanity
//!   ```
//!
//! On resume: read the header, verify magic + version + fingerprint; replay the
//! log up to `log_byte_len` (ignoring any trailing partial record); verify the
//! reconstructed pool sizes match the header.  Any mismatch/corruption → caller
//! starts fresh (deleting the stale files).

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;

use crate::laurent::Laurent;

use super::relkl_recur::{Cx, SlotState};

/// Header file magic.
const HDR_MAGIC: &[u8; 8] = b"RKLHDR\0\0";
/// Per-record frame magic in the log.
const REC_MAGIC: &[u8; 4] = b"RKL\n";
/// On-disk format version.  Bump on any layout change.
pub const BLK_VERSION: u32 = 1;

/// Slot tag bytes (1 byte per slot in the grid encoding).
const TAG_ABSENT: u8 = 0;
const TAG_PENDING: u8 = 1;
const TAG_DONE: u8 = 2;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for layer-granular checkpointing of one [`relklpols`] call.
///
/// Lives alongside the rep being processed: `dir/<rep_tag>.blklog` +
/// `dir/<rep_tag>.blkhdr`.  Only the *in-flight* rep has a log (the inner call
/// deletes its files on clean completion), so disk usage is bounded.
#[derive(Clone, Debug)]
pub struct RelKlCkptCfg {
    /// Directory holding the per-rep log + header (created if absent).
    pub dir: PathBuf,
    /// Stable tag identifying the rep, e.g. `"rep00042"`.  Names both files.
    pub rep_tag: String,
    /// Log a checkpoint after every this-many completed layers.  Default `1`.
    pub every_layers: usize,
    /// **Test-only** hook: stop the wavefront immediately after completing this
    /// layer index (simulating a SLURM kill at a layer boundary), returning a
    /// [`Stopped`](RelKlRunOutcome::Stopped) outcome.  `None` in production.
    ///
    /// Kept as a plain field (rather than `#[cfg(test)]`) so integration tests in
    /// `tests/` can drive it; it defaults to `None` and is documented test-only.
    pub test_stop_after_layer: Option<usize>,
}

impl RelKlCkptCfg {
    /// A config logging after every layer into `dir` for `rep_tag`.
    pub fn new(dir: impl Into<PathBuf>, rep_tag: impl Into<String>) -> Self {
        RelKlCkptCfg {
            dir: dir.into(),
            rep_tag: rep_tag.into(),
            every_layers: 1,
            test_stop_after_layer: None,
        }
    }

    /// Path of the block-log file.
    pub fn log_path(&self) -> PathBuf {
        self.dir.join(format!("{}.blklog", self.rep_tag))
    }

    /// Path of the side-header file.
    pub fn hdr_path(&self) -> PathBuf {
        self.dir.join(format!("{}.blkhdr", self.rep_tag))
    }

    /// Path of the header temp file (atomic write).
    fn hdr_tmp_path(&self) -> PathBuf {
        self.dir.join(format!("{}.blkhdr.tmp", self.rep_tag))
    }

    /// Delete both files (log + header).  Used on clean completion and when a
    /// stale/mismatched log must be discarded before a fresh run.
    pub fn delete_files(&self) {
        let _ = std::fs::remove_file(self.log_path());
        let _ = std::fs::remove_file(self.hdr_path());
        let _ = std::fs::remove_file(self.hdr_tmp_path());
    }

    /// `true` iff a header file is present (a prior interrupted run left a log).
    pub fn header_exists(&self) -> bool {
        self.hdr_path().is_file()
    }
}

// ---------------------------------------------------------------------------
// Driver wiring helpers (klcells ↔ relkl block logs)
// ---------------------------------------------------------------------------

/// Subdirectory of a driver checkpoint dir that holds the inner relkl logs.
pub const RELKL_SUBDIR: &str = "relkl";

/// The canonical rep tag for rep index `i` (zero-padded, stable, sortable).
pub fn rep_tag(i: usize) -> String {
    format!("rep{i:06}")
}

/// Build the per-rep [`RelKlCkptCfg`] under a driver checkpoint directory.
///
/// Logs live in `driver_dir/relkl/`; the rep tag is [`rep_tag(i)`](rep_tag).
/// `every_layers` is forwarded (the driver default is `1`).
pub fn rep_ckpt_cfg(driver_dir: &std::path::Path, i: usize, every_layers: usize) -> RelKlCkptCfg {
    RelKlCkptCfg {
        dir: driver_dir.join(RELKL_SUBDIR),
        rep_tag: rep_tag(i),
        every_layers: every_layers.max(1),
        test_stop_after_layer: None,
    }
}

/// Delete every relkl log/header under `driver_dir/relkl/` whose rep index is
/// `< keep_from` (stale logs for already-completed reps on a driver resume).
///
/// Best-effort: ignores I/O errors (a missing dir is fine).  Files for reps
/// `>= keep_from` (including the in-flight rep about to be resumed) are kept.
pub fn delete_stale_rep_logs(driver_dir: &std::path::Path, keep_from: usize) {
    let dir = driver_dir.join(RELKL_SUBDIR);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        // Match `repNNNNNN.blklog` / `repNNNNNN.blkhdr` / `.tmp`.
        if let Some(rest) = name.strip_prefix("rep") {
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(idx) = digits.parse::<usize>() {
                if idx < keep_from {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// In-memory side header
// ---------------------------------------------------------------------------

/// The parsed side header (`<rep_tag>.blkhdr`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlkHeader {
    /// `(group, W1, cell1)` content fingerprint binding the log to its rep.
    pub fingerprint: String,
    /// Index of the last layer whose record is fully durable in the log.
    pub last_layer: u32,
    /// Number of records in the log.
    pub records: u64,
    /// Byte length of the valid log prefix (records beyond this are ignored).
    pub log_bytes: u64,
    /// `rklpols.len()` after replaying through `last_layer` (replay sanity).
    pub rklpols_len: u64,
    /// `mues.len()` after replaying through `last_layer` (replay sanity).
    pub mues_len: u64,
}

impl BlkHeader {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(HDR_MAGIC);
        out.extend_from_slice(&BLK_VERSION.to_le_bytes());
        let fp = self.fingerprint.as_bytes();
        out.extend_from_slice(&(fp.len() as u32).to_le_bytes());
        out.extend_from_slice(fp);
        out.extend_from_slice(&self.last_layer.to_le_bytes());
        out.extend_from_slice(&self.records.to_le_bytes());
        out.extend_from_slice(&self.log_bytes.to_le_bytes());
        out.extend_from_slice(&self.rklpols_len.to_le_bytes());
        out.extend_from_slice(&self.mues_len.to_le_bytes());
        out
    }

    fn from_bytes(buf: &[u8]) -> io::Result<BlkHeader> {
        let mut r = Cursor { buf, pos: 0 };
        if r.next_bytes(8)? != HDR_MAGIC {
            return Err(corrupt("bad header magic — not a relkl block header"));
        }
        let version = r.read_u32()?;
        if version != BLK_VERSION {
            return Err(corrupt(format!(
                "unsupported block-log version {version} (expected {BLK_VERSION})"
            )));
        }
        let fp_len = r.read_u32()? as usize;
        let fp_bytes = r.next_bytes(fp_len)?;
        let fingerprint = String::from_utf8(fp_bytes.to_vec())
            .map_err(|_| corrupt("fingerprint is not valid UTF-8"))?;
        let last_layer = r.read_u32()?;
        let records = r.read_u64()?;
        let log_bytes = r.read_u64()?;
        let rklpols_len = r.read_u64()?;
        let mues_len = r.read_u64()?;
        Ok(BlkHeader {
            fingerprint,
            last_layer,
            records,
            log_bytes,
            rklpols_len,
            mues_len,
        })
    }

    /// Atomically write this header into `cfg.dir` (tmp file + `rename`).
    fn write_atomic(&self, cfg: &RelKlCkptCfg) -> io::Result<()> {
        std::fs::create_dir_all(&cfg.dir)?;
        let tmp = cfg.hdr_tmp_path();
        let bytes = self.to_bytes();
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, cfg.hdr_path())?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// One completed layer's payload (in memory, before framing)
// ---------------------------------------------------------------------------

/// The finalized off-diagonal blocks + pool deltas of one completed layer.
///
/// `blocks` is `(x, grid)` for each present `(y, x)` with `x < y`; the diagonal
/// `(y, y)` block is recomputed deterministically on resume and not stored.
pub(super) struct LayerRecord {
    pub y: Cx,
    pub blocks: Vec<(Cx, Vec<Vec<SlotState>>)>,
    pub rklpols_delta: Vec<Laurent>,
    pub mues_delta: Vec<Laurent>,
}

impl LayerRecord {
    /// Encode the framed record bytes (magic + y + payload-length + payload).
    fn to_frame(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(self.blocks.len() as u32).to_le_bytes());
        for (x, grid) in &self.blocks {
            payload.extend_from_slice(&(*x as u32).to_le_bytes());
            let nc = grid.len();
            payload.extend_from_slice(&(nc as u32).to_le_bytes());
            for row in grid {
                for slot in row {
                    write_slot(&mut payload, *slot);
                }
            }
        }
        write_laurents(&mut payload, &self.rklpols_delta);
        write_laurents(&mut payload, &self.mues_delta);

        let mut frame = Vec::with_capacity(payload.len() + 16);
        frame.extend_from_slice(REC_MAGIC);
        frame.extend_from_slice(&(self.y as u32).to_le_bytes());
        frame.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        frame.extend_from_slice(&payload);
        frame
    }
}

/// One replayed layer (decoded from the log): the off-diagonal blocks to
/// overwrite + the pool entries to append.
pub(super) struct ReplayedLayer {
    pub y: Cx,
    pub blocks: Vec<(Cx, Vec<Vec<SlotState>>)>,
    pub rklpols_delta: Vec<Laurent>,
    pub mues_delta: Vec<Laurent>,
}

// ---------------------------------------------------------------------------
// Writer: append a layer record + update the header atomically
// ---------------------------------------------------------------------------

/// Append-only block-log writer that keeps the side header in sync.
///
/// The header is rewritten atomically *after* each record's bytes are durable,
/// so the header's `log_bytes` always bounds a fully-written record prefix.
pub(super) struct BlkLogWriter<'a> {
    cfg: &'a RelKlCkptCfg,
    fingerprint: String,
    file: std::fs::File,
    log_bytes: u64,
    records: u64,
}

impl<'a> BlkLogWriter<'a> {
    /// Create (truncating) a fresh log + header for `cfg`.
    pub fn create(cfg: &'a RelKlCkptCfg, fingerprint: &str) -> io::Result<Self> {
        std::fs::create_dir_all(&cfg.dir)?;
        let file = std::fs::File::create(cfg.log_path())?;
        Ok(BlkLogWriter {
            cfg,
            fingerprint: fingerprint.to_string(),
            file,
            log_bytes: 0,
            records: 0,
        })
    }

    /// Reopen an existing log for append, after a successful replay.
    ///
    /// Truncates the physical file to the header's `log_bytes` (discarding any
    /// trailing partial record from a crash mid-append), seeks to the end, and
    /// resumes the running record/byte counters from `header`.  New layers are
    /// appended after the validated prefix.
    pub fn open_existing(
        cfg: &'a RelKlCkptCfg,
        fingerprint: &str,
        header: &BlkHeader,
    ) -> io::Result<Self> {
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(cfg.log_path())?;
        // Drop any trailing partial record beyond the durable prefix.
        file.set_len(header.log_bytes)?;
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::Start(header.log_bytes))?;
        Ok(BlkLogWriter {
            cfg,
            fingerprint: fingerprint.to_string(),
            file,
            log_bytes: header.log_bytes,
            records: header.records,
        })
    }

    /// Append one completed layer, fsync the log, then atomically rewrite the
    /// header.  `rklpols_len`/`mues_len` are the pool sizes after this layer.
    pub fn append_layer(
        &mut self,
        rec: &LayerRecord,
        rklpols_len: usize,
        mues_len: usize,
    ) -> io::Result<()> {
        let frame = rec.to_frame();
        self.file.write_all(&frame)?;
        self.file.sync_all()?;
        self.log_bytes += frame.len() as u64;
        self.records += 1;

        let hdr = BlkHeader {
            fingerprint: self.fingerprint.clone(),
            last_layer: rec.y as u32,
            records: self.records,
            log_bytes: self.log_bytes,
            rklpols_len: rklpols_len as u64,
            mues_len: mues_len as u64,
        };
        hdr.write_atomic(self.cfg)
    }
}

// ---------------------------------------------------------------------------
// Resume: load header + replay the log
// ---------------------------------------------------------------------------

/// The replay result: every completed layer (in order) and the validated header.
pub(super) struct ReplayState {
    pub header: BlkHeader,
    pub layers: Vec<ReplayedLayer>,
}

/// Try to load + replay an existing log for `cfg` matching `fingerprint`.
///
/// Returns:
/// - `Ok(None)`        — no header file (fresh run);
/// - `Ok(Some(state))` — a valid, matching log replayed up to its header bound;
/// - `Err(_)`          — header/log missing-partner, bad magic/version,
///   fingerprint mismatch, corruption, or a pool-size sanity failure.  The
///   caller logs a warning and starts fresh (deleting the stale files).
pub(super) fn load_and_replay(
    cfg: &RelKlCkptCfg,
    fingerprint: &str,
) -> io::Result<Option<ReplayState>> {
    let hdr_path = cfg.hdr_path();
    if !hdr_path.is_file() {
        return Ok(None);
    }
    let hdr_bytes = std::fs::read(&hdr_path)?;
    let header = BlkHeader::from_bytes(&hdr_bytes)?;
    if header.fingerprint != fingerprint {
        return Err(corrupt(format!(
            "block-log fingerprint mismatch: file has '{}', run is '{}'",
            header.fingerprint, fingerprint
        )));
    }

    // Read only the header-bounded prefix of the log; anything beyond
    // `log_bytes` is a trailing partial record from a crash mid-append.
    let log_all = std::fs::read(cfg.log_path())?;
    let bound = header.log_bytes.min(log_all.len() as u128 as u64) as usize;
    let buf = &log_all[..bound];

    let mut r = Cursor { buf, pos: 0 };
    let mut layers = Vec::with_capacity(header.records as usize);
    for _ in 0..header.records {
        layers.push(read_layer(&mut r)?);
    }

    // Sanity: the replayed layer count and last index must match the header.
    if let Some(last) = layers.last() {
        if last.y as u32 != header.last_layer {
            return Err(corrupt(format!(
                "block-log last layer {} != header {}",
                last.y, header.last_layer
            )));
        }
    } else if header.records != 0 {
        return Err(corrupt("block-log header claims records but log is empty"));
    }

    Ok(Some(ReplayState { header, layers }))
}

/// Apply replayed layers onto a freshly-recomputed setup `mat`/pools.
///
/// Overwrites each block `(y, x)` with its finalized grid and appends each
/// layer's pool deltas in order.  Returns the next layer index to compute and
/// validates the final pool sizes against the header.
pub(super) fn apply_replay(
    state: &ReplayState,
    mat: &mut HashMap<(Cx, Cx), Vec<Vec<SlotState>>>,
    rklpols: &mut Vec<Laurent>,
    mues: &mut Vec<Laurent>,
) -> io::Result<usize> {
    for layer in &state.layers {
        for (x, grid) in &layer.blocks {
            mat.insert((layer.y, *x), grid.clone());
        }
        rklpols.extend(layer.rklpols_delta.iter().cloned());
        mues.extend(layer.mues_delta.iter().cloned());
    }
    if rklpols.len() as u64 != state.header.rklpols_len {
        return Err(corrupt(format!(
            "replayed rklpols len {} != header {}",
            rklpols.len(),
            state.header.rklpols_len
        )));
    }
    if mues.len() as u64 != state.header.mues_len {
        return Err(corrupt(format!(
            "replayed mues len {} != header {}",
            mues.len(),
            state.header.mues_len
        )));
    }
    Ok(state.header.last_layer as usize + 1)
}

// ---------------------------------------------------------------------------
// Record (de)serialization
// ---------------------------------------------------------------------------

fn read_layer(r: &mut Cursor<'_>) -> io::Result<ReplayedLayer> {
    if r.next_bytes(4)? != REC_MAGIC {
        return Err(corrupt("bad record magic in block log"));
    }
    let y = r.read_u32()? as Cx;
    let payload_len = r.read_u64()? as usize;
    // Frame the payload so a malformed inner length cannot read past it.
    let start = r.pos;
    let _ = r.next_bytes(payload_len)?; // ensures payload is present
    let mut pr = Cursor {
        buf: &r.buf[start..start + payload_len],
        pos: 0,
    };

    let nblocks = pr.read_u32()? as usize;
    let mut blocks = Vec::with_capacity(nblocks);
    for _ in 0..nblocks {
        let x = pr.read_u32()? as Cx;
        let nc = pr.read_u32()? as usize;
        let mut grid = vec![vec![SlotState::Absent; nc]; nc];
        for row in grid.iter_mut() {
            for slot in row.iter_mut() {
                *slot = read_slot(&mut pr)?;
            }
        }
        blocks.push((x, grid));
    }
    let rklpols_delta = read_laurents(&mut pr)?;
    let mues_delta = read_laurents(&mut pr)?;
    Ok(ReplayedLayer {
        y,
        blocks,
        rklpols_delta,
        mues_delta,
    })
}

fn write_slot(out: &mut Vec<u8>, slot: SlotState) {
    match slot {
        SlotState::Absent => out.push(TAG_ABSENT),
        SlotState::Pending => out.push(TAG_PENDING),
        SlotState::Done { rk, mu } => {
            out.push(TAG_DONE);
            out.extend_from_slice(&rk.to_le_bytes());
            out.extend_from_slice(&mu.to_le_bytes());
        }
    }
}

fn read_slot(r: &mut Cursor<'_>) -> io::Result<SlotState> {
    let tag = r.next_bytes(1)?[0];
    match tag {
        TAG_ABSENT => Ok(SlotState::Absent),
        TAG_PENDING => Ok(SlotState::Pending),
        TAG_DONE => {
            let rk = r.read_u32()?;
            let mu = r.read_u32()?;
            Ok(SlotState::Done { rk, mu })
        }
        other => Err(corrupt(format!("unknown slot tag {other}"))),
    }
}

/// Serialize a Laurent as `val (i32) + len (u32) + len× coeff (i64)`.
fn write_laurent(out: &mut Vec<u8>, p: &Laurent) {
    out.extend_from_slice(&p.val().to_le_bytes());
    let coeffs = p.coeffs();
    out.extend_from_slice(&(coeffs.len() as u32).to_le_bytes());
    for &c in coeffs {
        out.extend_from_slice(&c.to_le_bytes());
    }
}

fn read_laurent(r: &mut Cursor<'_>) -> io::Result<Laurent> {
    let val = r.read_i32()?;
    let n = r.read_u32()? as usize;
    let mut coeffs = Vec::with_capacity(n);
    for _ in 0..n {
        coeffs.push(r.read_i64()?);
    }
    Ok(Laurent::from_coeffs(val, coeffs))
}

fn write_laurents(out: &mut Vec<u8>, pols: &[Laurent]) {
    out.extend_from_slice(&(pols.len() as u32).to_le_bytes());
    for p in pols {
        write_laurent(out, p);
    }
}

fn read_laurents(r: &mut Cursor<'_>) -> io::Result<Vec<Laurent>> {
    let n = r.read_u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(read_laurent(r)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Minimal little-endian cursor (no extra dependency)
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn next_bytes(&mut self, n: usize) -> io::Result<&[u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| corrupt("block log truncated"))?;
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }
    fn read_u32(&mut self) -> io::Result<u32> {
        let b = self.next_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_i32(&mut self) -> io::Result<i32> {
        let b = self.next_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_u64(&mut self) -> io::Result<u64> {
        let b = self.next_bytes(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }
    fn read_i64(&mut self) -> io::Result<i64> {
        let b = self.next_bytes(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(i64::from_le_bytes(a))
    }
}

fn corrupt(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Truncate the physical log file to `bytes` (used by the truncated-log test to
/// simulate a crash mid-append).  Kept here so tests do not duplicate the path
/// logic; the resume path is already header-byte-bounded, so this only matters
/// when the header survives but the log's tail does not.
pub fn truncate_log_for_test(cfg: &RelKlCkptCfg, bytes: u64) -> io::Result<()> {
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(cfg.log_path())?;
    f.set_len(bytes)?;
    Ok(())
}

/// Read the raw log file length in bytes (test helper).
pub fn log_len_for_test(cfg: &RelKlCkptCfg) -> io::Result<u64> {
    Ok(std::fs::metadata(cfg.log_path())?.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let h = BlkHeader {
            fingerprint: "B4|nc=12|h=deadbeef".into(),
            last_layer: 7,
            records: 8,
            log_bytes: 4096,
            rklpols_len: 33,
            mues_len: 41,
        };
        let bytes = h.to_bytes();
        let back = BlkHeader::from_bytes(&bytes).unwrap();
        assert_eq!(back, h);
    }

    #[test]
    fn header_rejects_bad_magic_and_version() {
        let mut bytes = vec![0u8; 40];
        bytes[..8].copy_from_slice(b"NOPEHDR\0");
        assert!(BlkHeader::from_bytes(&bytes).is_err());

        let h = BlkHeader {
            fingerprint: "x".into(),
            last_layer: 0,
            records: 0,
            log_bytes: 0,
            rklpols_len: 2,
            mues_len: 2,
        };
        let mut bytes = h.to_bytes();
        bytes[8..12].copy_from_slice(&999u32.to_le_bytes()); // bump version
        assert!(BlkHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn header_rejects_truncation() {
        let h = BlkHeader {
            fingerprint: "abc".into(),
            last_layer: 1,
            records: 2,
            log_bytes: 10,
            rklpols_len: 3,
            mues_len: 4,
        };
        let bytes = h.to_bytes();
        for cut in [6usize, 12, bytes.len() - 1] {
            assert!(
                BlkHeader::from_bytes(&bytes[..cut]).is_err(),
                "truncation at {cut} must error"
            );
        }
    }

    #[test]
    fn slot_roundtrip_all_variants() {
        let grid = [
            SlotState::Absent,
            SlotState::Pending,
            SlotState::Done { rk: 0, mu: 0 },
            SlotState::Done { rk: 5, mu: 9 },
        ];
        let mut buf = Vec::new();
        for s in &grid {
            write_slot(&mut buf, *s);
        }
        let mut r = Cursor { buf: &buf, pos: 0 };
        for s in &grid {
            assert_eq!(read_slot(&mut r).unwrap(), *s);
        }
    }

    #[test]
    fn laurent_roundtrip() {
        let pols = vec![
            Laurent::zero(),
            Laurent::one(),
            Laurent::monomial(3, -2),
            Laurent::from_coeffs(-1, vec![1, 0, 0, 7, -4]),
        ];
        let mut buf = Vec::new();
        write_laurents(&mut buf, &pols);
        let mut r = Cursor { buf: &buf, pos: 0 };
        let back = read_laurents(&mut r).unwrap();
        assert_eq!(back, pols);
    }

    #[test]
    fn layer_record_frame_roundtrip() {
        let rec = LayerRecord {
            y: 3,
            blocks: vec![
                (
                    1,
                    vec![
                        vec![SlotState::Absent, SlotState::Done { rk: 1, mu: 0 }],
                        vec![SlotState::Pending, SlotState::Done { rk: 2, mu: 5 }],
                    ],
                ),
                (
                    0,
                    vec![
                        vec![SlotState::Done { rk: 0, mu: 0 }, SlotState::Absent],
                        vec![SlotState::Absent, SlotState::Absent],
                    ],
                ),
            ],
            rklpols_delta: vec![Laurent::monomial(1, 2), Laurent::from_coeffs(0, vec![1, 1])],
            mues_delta: vec![Laurent::one()],
        };
        let frame = rec.to_frame();
        let mut r = Cursor {
            buf: &frame,
            pos: 0,
        };
        let back = read_layer(&mut r).unwrap();
        assert_eq!(back.y, rec.y);
        assert_eq!(back.blocks, rec.blocks);
        assert_eq!(back.rklpols_delta, rec.rklpols_delta);
        assert_eq!(back.mues_delta, rec.mues_delta);
        // The cursor consumed exactly the frame (no trailing slack).
        assert_eq!(r.pos, frame.len());
    }

    #[test]
    fn rep_tag_and_stale_deletion() {
        assert_eq!(rep_tag(42), "rep000042");
        let dir = std::env::temp_dir().join(format!("rcx_stale_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let relkl = dir.join(RELKL_SUBDIR);
        std::fs::create_dir_all(&relkl).unwrap();
        for i in [0usize, 1, 2, 5] {
            std::fs::write(relkl.join(format!("{}.blkhdr", rep_tag(i))), b"x").unwrap();
            std::fs::write(relkl.join(format!("{}.blklog", rep_tag(i))), b"x").unwrap();
        }
        delete_stale_rep_logs(&dir, 2);
        // reps 0,1 gone; reps 2,5 kept.
        assert!(!relkl.join(format!("{}.blkhdr", rep_tag(0))).exists());
        assert!(!relkl.join(format!("{}.blkhdr", rep_tag(1))).exists());
        assert!(relkl.join(format!("{}.blkhdr", rep_tag(2))).exists());
        assert!(relkl.join(format!("{}.blkhdr", rep_tag(5))).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
