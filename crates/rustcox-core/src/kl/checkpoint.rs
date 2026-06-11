//! Checkpoint persistence for the streaming `klcells` driver (Task Q1).
//!
//! The multi-day E8 `klcells` run must survive SLURM timeouts and node
//! preemption.  The driver's persistent loop state is compact (see the module
//! docs of [`klcells`](crate::kl::klcells)): only the involution skip-set
//! `celms`, the running `tot`, the index `i` into the W1 star-reps, the recorded
//! star-rep registry fingerprints, and the count of cell records emitted to the
//! stream so far.  All of that is serialized here.
//!
//! # On-disk format (`klcells.ckpt`)
//!
//! A hand-rolled, **versioned, little-endian** binary blob.  Hand-rolled (rather
//! than serde) because the payload is a few flat arrays of `u32`/`u128` and a
//! string fingerprint — a stable, self-describing layout is easier to audit and
//! version than deriving `Serialize` on the internal element types.  Layout:
//!
//! ```text
//! offset  bytes  field
//! 0       8      magic  = b"RCXCKPT\0"
//! 8       4      format version (u32 LE) = CKPT_VERSION
//! 12      4      fingerprint byte length F (u32 LE)
//! 16      F      fingerprint string (UTF-8: group type + opts hash)
//! ...     16     next_rep   (u128 LE)  — index `i` into W1 star-reps to resume at
//! ...     16     tot        (u128 LE)  — elements of W placed so far
//! ...     16     records    (u128 LE)  — cell records written to the stream so far
//! ...     4      rank       (u32 LE)   — coxelm word length (all celms share it)
//! ...     8      n_celms    (u64 LE)
//! ...     n*rank*4   celms   (sorted; each a rank-long array of u32 LE)
//! ...     8      n_reg      (u64 LE)   — star-rep registry fingerprint count
//! ...     n_reg*rank*4  registry (each a rank-long array of u32 LE; xrep[0])
//! ```
//!
//! Atomic update: write to `dir/klcells.ckpt.tmp`, `fsync`, then `rename` over
//! `dir/klcells.ckpt`.  A crash mid-write leaves the previous good checkpoint
//! intact (rename is atomic on POSIX).
//!
//! The fingerprint binds a checkpoint to its `(group, opts)`: resume is refused
//! (and the run starts fresh) if the fingerprint does not match the current run.

use std::collections::BTreeSet;
use std::io::{self, Write};
use std::path::PathBuf;

use crate::element::CoxElm;

/// Magic bytes at the head of every checkpoint file.
const MAGIC: &[u8; 8] = b"RCXCKPT\0";
/// On-disk format version.  Bump on any layout change.
pub const CKPT_VERSION: u32 = 1;
/// Checkpoint file name within the checkpoint directory.
pub const CKPT_FILE: &str = "klcells.ckpt";
/// Temp file name used for the atomic write.
const CKPT_TMP: &str = "klcells.ckpt.tmp";

/// Configuration for checkpointing a streaming `klcells` run.
#[derive(Clone, Debug)]
pub struct CheckpointCfg {
    /// Directory holding `klcells.ckpt` (created if absent).
    pub dir: PathBuf,
    /// Write a checkpoint after this many processed reps.  Default `1` (after
    /// every rep) — the safest setting and the one the E8 long-run uses.
    pub every_reps: usize,
}

impl CheckpointCfg {
    /// A config that checkpoints after every rep into `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        CheckpointCfg {
            dir: dir.into(),
            every_reps: 1,
        }
    }

    /// Path of the checkpoint file.
    pub fn ckpt_path(&self) -> PathBuf {
        self.dir.join(CKPT_FILE)
    }
}

/// The serialized persistent loop state of a `klcells` run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    /// `(group, opts)` fingerprint binding this checkpoint to its run.
    pub fingerprint: String,
    /// Next W1 star-rep index to process (`i`); reps `< next_rep` are skipped.
    pub next_rep: u128,
    /// Elements of `W` placed so far (`tot`).
    pub tot: u128,
    /// Cell records written to the stream so far.  On resume the CLI truncates
    /// its stream file to this many records, then re-emits from `next_rep`.
    pub records: u128,
    /// The coxelm word length (= group rank); all celms/registry entries are
    /// `rank`-long.
    pub rank: usize,
    /// Involution coxelms (the skip-set), sorted for a canonical encoding.
    pub celms: Vec<CoxElm>,
    /// Star-rep registry fingerprints (`xrep[0]` of each recorded rep).
    pub registry: Vec<CoxElm>,
}

impl Checkpoint {
    /// Serialize to the versioned binary layout (see module docs).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&CKPT_VERSION.to_le_bytes());

        let fp = self.fingerprint.as_bytes();
        out.extend_from_slice(&(fp.len() as u32).to_le_bytes());
        out.extend_from_slice(fp);

        out.extend_from_slice(&self.next_rep.to_le_bytes());
        out.extend_from_slice(&self.tot.to_le_bytes());
        out.extend_from_slice(&self.records.to_le_bytes());
        out.extend_from_slice(&(self.rank as u32).to_le_bytes());

        write_coxelms(&mut out, &self.celms, self.rank);
        write_coxelms(&mut out, &self.registry, self.rank);
        out
    }

    /// Parse from the versioned binary layout.  Returns an error on a bad magic,
    /// an unknown version, or a truncated/corrupt body.
    pub fn from_bytes(buf: &[u8]) -> io::Result<Checkpoint> {
        let mut r = Cursor { buf, pos: 0 };

        let magic = r.next_bytes(8)?;
        if magic != MAGIC {
            return Err(corrupt("bad magic — not a klcells checkpoint"));
        }
        let version = r.read_u32()?;
        if version != CKPT_VERSION {
            return Err(corrupt(format!(
                "unsupported checkpoint version {version} (expected {CKPT_VERSION})"
            )));
        }

        let fp_len = r.read_u32()? as usize;
        let fp_bytes = r.next_bytes(fp_len)?;
        let fingerprint = String::from_utf8(fp_bytes.to_vec())
            .map_err(|_| corrupt("fingerprint is not valid UTF-8"))?;

        let next_rep = r.read_u128()?;
        let tot = r.read_u128()?;
        let records = r.read_u128()?;
        let rank = r.read_u32()? as usize;

        let celms = read_coxelms(&mut r, rank)?;
        let registry = read_coxelms(&mut r, rank)?;

        Ok(Checkpoint {
            fingerprint,
            next_rep,
            tot,
            records,
            rank,
            celms,
            registry,
        })
    }

    /// Atomically write this checkpoint into `cfg.dir` (tmp file + `rename`).
    ///
    /// Creates `cfg.dir` if it does not yet exist.  The previous checkpoint, if
    /// any, is replaced only after the new bytes are fully flushed and synced —
    /// a crash mid-write never corrupts the resume point.
    pub fn write_atomic(&self, cfg: &CheckpointCfg) -> io::Result<()> {
        std::fs::create_dir_all(&cfg.dir)?;
        let tmp = cfg.dir.join(CKPT_TMP);
        let bytes = self.to_bytes();
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, cfg.ckpt_path())?;
        Ok(())
    }
}

/// Load and validate the checkpoint in `cfg.dir`, if any.
///
/// Returns:
/// - `Ok(None)` if no checkpoint file exists (a fresh run);
/// - `Ok(Some(ckpt))` if a checkpoint exists, is well-formed, and its
///   fingerprint matches `expected_fingerprint`;
/// - `Err(_)` if a checkpoint exists but is corrupt OR its fingerprint does not
///   match (caller decides whether to start fresh and warn).
pub fn load_matching(cfg: &CheckpointCfg, expected_fingerprint: &str) -> io::Result<Checkpoint> {
    let path = cfg.ckpt_path();
    let bytes = std::fs::read(&path)?;
    let ckpt = Checkpoint::from_bytes(&bytes)?;
    if ckpt.fingerprint != expected_fingerprint {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "checkpoint fingerprint mismatch: file has '{}', run is '{}'",
                ckpt.fingerprint, expected_fingerprint
            ),
        ));
    }
    Ok(ckpt)
}

/// `true` iff a checkpoint file exists in `cfg.dir`.
pub fn exists(cfg: &CheckpointCfg) -> bool {
    cfg.ckpt_path().is_file()
}

// ---------------------------------------------------------------------------
// CoxElm (de)serialization
// ---------------------------------------------------------------------------

/// Write a length-prefixed, sorted list of `rank`-long coxelms.
///
/// The input order is irrelevant — we encode a sorted set, so the byte image is
/// canonical for a given mathematical state (helps debugging diffs across runs).
fn write_coxelms(out: &mut Vec<u8>, elms: &[CoxElm], rank: usize) {
    // Canonicalize: collect a sorted set of the underlying u32 arrays.
    let sorted: BTreeSet<&[u32]> = elms.iter().map(|c| &c.0[..]).collect();
    out.extend_from_slice(&(sorted.len() as u64).to_le_bytes());
    for arr in sorted {
        debug_assert_eq!(arr.len(), rank, "coxelm length must equal rank");
        for &x in arr {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
}

/// Read a length-prefixed list of `rank`-long coxelms.
fn read_coxelms(r: &mut Cursor<'_>, rank: usize) -> io::Result<Vec<CoxElm>> {
    let n = r.read_u64()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut arr = Vec::with_capacity(rank);
        for _ in 0..rank {
            arr.push(r.read_u32()?);
        }
        out.push(CoxElm(arr.into_boxed_slice()));
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
            .ok_or_else(|| corrupt("checkpoint truncated"))?;
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        let b = self.next_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        let b = self.next_bytes(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }

    fn read_u128(&mut self) -> io::Result<u128> {
        let b = self.next_bytes(16)?;
        let mut a = [0u8; 16];
        a.copy_from_slice(b);
        Ok(u128::from_le_bytes(a))
    }
}

fn corrupt(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ce(xs: &[u32]) -> CoxElm {
        CoxElm(xs.to_vec().into_boxed_slice())
    }

    #[test]
    fn roundtrip_full_state() {
        let ck = Checkpoint {
            fingerprint: "B4|h=12345".to_string(),
            next_rep: 7,
            tot: 384,
            records: 42,
            rank: 4,
            celms: vec![ce(&[3, 1, 0, 2]), ce(&[0, 0, 0, 0]), ce(&[9, 9, 9, 9])],
            registry: vec![ce(&[1, 2, 3, 4])],
        };
        let bytes = ck.to_bytes();
        let back = Checkpoint::from_bytes(&bytes).unwrap();
        // celms come back sorted; compare as sets.
        assert_eq!(back.fingerprint, ck.fingerprint);
        assert_eq!(back.next_rep, ck.next_rep);
        assert_eq!(back.tot, ck.tot);
        assert_eq!(back.records, ck.records);
        assert_eq!(back.rank, ck.rank);
        let got: BTreeSet<_> = back.celms.iter().map(|c| c.0.to_vec()).collect();
        let want: BTreeSet<_> = ck.celms.iter().map(|c| c.0.to_vec()).collect();
        assert_eq!(got, want);
        assert_eq!(back.registry, ck.registry);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = vec![0u8; 32];
        bytes[..8].copy_from_slice(b"NOTACKPT");
        assert!(Checkpoint::from_bytes(&bytes).is_err());
    }

    #[test]
    fn rejects_truncated() {
        let ck = Checkpoint {
            fingerprint: "x".into(),
            next_rep: 1,
            tot: 1,
            records: 0,
            rank: 2,
            celms: vec![ce(&[0, 0])],
            registry: vec![],
        };
        let bytes = ck.to_bytes();
        for cut in [10usize, 20, bytes.len() - 1] {
            assert!(
                Checkpoint::from_bytes(&bytes[..cut]).is_err(),
                "truncation at {cut} must error"
            );
        }
    }

    #[test]
    fn atomic_write_and_load_matching() {
        let dir = std::env::temp_dir().join(format!("rcx_ckpt_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = CheckpointCfg::new(&dir);
        assert!(!exists(&cfg));

        let ck = Checkpoint {
            fingerprint: "F4|h=9".into(),
            next_rep: 3,
            tot: 100,
            records: 5,
            rank: 4,
            celms: vec![ce(&[1, 0, 0, 0])],
            registry: vec![ce(&[0, 0, 0, 0])],
        };
        ck.write_atomic(&cfg).unwrap();
        assert!(exists(&cfg));

        // Matching fingerprint loads.
        let loaded = load_matching(&cfg, "F4|h=9").unwrap();
        assert_eq!(loaded.next_rep, 3);
        assert_eq!(loaded.records, 5);

        // Mismatched fingerprint is rejected.
        assert!(load_matching(&cfg, "F4|h=10").is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
