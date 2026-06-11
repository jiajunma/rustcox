//! Sparse per-rep persistence of a [`RelKlOutput`](super::RelKlOutput) (Task Q5).
//!
//! A `klcells` run on a monster group (E8 takes *days*) computes one
//! [`relklpols`](super::relklpols) output per processed W1 star-rep, feeds it
//! straight into [`CellGraph::from_relkl`](crate::cellgraph::CellGraph::from_relkl),
//! and then **discards** it.  That discarded output *is* the relative-KL
//! polynomial data — the expensive thing the induction produced.  This module
//! persists it to a small, self-contained archive so a future analysis (a
//! different cell decomposition, a W-graph cross-check, an mu-pattern study) can
//! reload the relkl data without re-running the whole induction.
//!
//! # Sparsity — why this is small
//!
//! The induced `klmat` is dominated by *no-edge* slots.  Measured E7 aggregate:
//! ~2.7M absent / ~27.0M zero / ~1.7M nonzero slots — only ~5.5% of marked slots
//! carry a non-zero relative-KL mu.  In the [`RelKlOutput`](super::RelKlOutput)
//! `input.klmat` (a flat strict-lower-triangular matrix of single-`Global`-index
//! [`SlotData`](crate::cellgraph::SlotData)s) a slot is *load-bearing for the
//! W-graph* iff its Global `mu` index is non-zero: [`from_relkl`] only records an
//! `mmat` edge when the interned pool value is real (`mu != 0`); a `mu == 0` slot
//! (whether it was PyCox `'0c0'` or a marked-but-zero `relmue`) yields no edge,
//! exactly like an absent (`'f'`) slot.  So we store **only the `mu != 0`
//! slots** as flat `(flat_y, flat_x, rk, mu)` triplets and reconstruct every
//! dropped slot as `None`.
//!
//! [`from_relkl`]: crate::cellgraph::CellGraph::from_relkl
//!
//! # LOSSY: zero-vs-incomparable is dropped (W-graph payload is exact)
//!
//! The sparse file drops **both** absent (`'f'`, incomparable / no Bruhat
//! relation) and zero (`'0c0'` / marked-zero, comparable but zero mu) slots —
//! they are byte-indistinguishable once dropped, so reconstruction cannot tell
//! them apart and rebuilds *all* dropped slots as `None` (`'f'`).  This loses the
//! zero-vs-incomparable (Bruhat-comparability) distinction, which matters **only**
//! for reconstructing the Bruhat order from the matrix.  It does **not** affect
//! the W-graph / cells payload: [`from_relkl`] treats `'0c0'` (no edge) and `'f'`
//! (no slot) identically (verified in the P4 review), and `to_relkl_input` →
//! `from_relkl` reproduces a byte-identical [`CellGraph`] (vertices, `isets`,
//! `mmat`, `mpols`).  In short: the cells / W-graph data is preserved exactly;
//! the dropped distinction is irrelevant to it.
//!
//! # On-disk format (versioned, magic header, gz via flate2)
//!
//! A single gz-compressed file, little-endian, framed:
//!
//! ```text
//!   8   magic           = b"RKLSAVE\0"
//!   4   format version  (u32 LE) = SAVE_VERSION
//!   4   group label byte length G (u32 LE)
//!   G   group label     (UTF-8, e.g. "E8")
//!   4   rep tag byte length R (u32 LE)
//!   R   rep tag         (UTF-8, e.g. "rep000042")
//!   4   n  = number of induced elements (u32 LE)
//!   4   rank            (u32 LE)
//!   4   elms count = n  (u32 LE)            — redundant frame for the words
//!   n×  word:   4 len (u32 LE) + len× generator byte (u8)
//!   --- rklpols pool ---  (Laurents: 4 count + each `val i32 + len u32 + coeffs i64`)
//!   --- mues pool ---     (Laurents, the Global mu pool)
//!   4   sparse slot count S (u32 LE)
//!   S×  slot:   4 flat_y (u32) + 4 flat_x (u32) + 4 rk (u32) + 4 mu (u32)   — mu != 0 only
//! ```
//!
//! The Laurent wire format and the bounds-checked little-endian cursor are
//! reused verbatim from [`relkl_ckpt`](super::relkl_ckpt) (the Q4 layer log),
//! per the task brief.

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use flate2::{read::GzDecoder, write::GzEncoder, Compression};

use crate::cellgraph::{KlSlot, MuPools, RelKlInput, SlotData};
use crate::element::Word;
use crate::laurent::Laurent;

use super::relkl_ckpt::{
    corrupt, read_laurents, write_laurents, Cursor,
};
use super::RelKlOutput;

/// Archive magic (8 bytes, NUL-padded).
const SAVE_MAGIC: &[u8; 8] = b"RKLSAVE\0";
/// On-disk format version.  Bump on any layout change.
pub const SAVE_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// SavedRelKl — the loaded archive
// ---------------------------------------------------------------------------

/// A loaded sparse relkl archive (see [`load_relkl`]).
///
/// Reconstructs a [`RelKlOutput`](super::RelKlOutput)-equivalent.  Two notes on
/// what is and is not recovered:
///
/// - **`perms` are not stored** — they are recomputed on demand from `elms`
///   words via the group ([`perms`](SavedRelKl::perms)); persisting perms would
///   bloat the file with derivable data.
/// - **LOSSY**: dropped (`mu == 0`) slots reconstruct as `None`, so the
///   zero-vs-incomparable distinction is gone (see the [module docs](self)).  The
///   W-graph / cells payload ([`to_relkl_input`](SavedRelKl::to_relkl_input) →
///   `from_relkl`) is exact.
#[derive(Clone, Debug, PartialEq)]
pub struct SavedRelKl {
    /// Group label as passed to [`save_relkl`] (e.g. `"E8"`).  Provenance only.
    pub group_label: String,
    /// Rep tag as passed to [`save_relkl`] (e.g. `"rep000042"`).  Provenance only.
    pub rep_tag: String,
    /// Number of induced elements (`= elms.len()`).
    pub n: usize,
    /// Rank of the ambient group `W`.
    pub rank: usize,
    /// Induced-set words (the `RelKlInput::elms`), increasing length.
    pub elms: Vec<Word>,
    /// The relative-KL polynomial pool, seeded `[zero, one]`.
    pub rklpols: Vec<Laurent>,
    /// The global mu pool (`mues`), seeded `[zero, one]`.
    pub mues: Vec<Laurent>,
    /// Sparse slots that carry a non-zero mu, as `(flat_y, flat_x, rk, mu)`.
    /// Every other lower-triangular position reconstructs to `None`.
    pub slots: Vec<SparseSlot>,
}

/// One persisted sparse slot: a lower-triangular position with `mu != 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseSlot {
    /// Row index in the flat strict-lower-triangular `klmat` (`> flat_x`).
    pub flat_y: u32,
    /// Column index (`< flat_y`).
    pub flat_x: u32,
    /// The `rklpols` index (relative-KL polynomial) of this slot, if known.
    ///
    /// The [`RelKlOutput`](super::RelKlOutput) `klmat` retains only the `mu`
    /// index (a single Global index), not the working-recursion `rk`; so on the
    /// save path this is always `0` (no `rk` is available from the output).  The
    /// field is kept in the wire format for forward-compatibility with a future
    /// output that also exposes `rk`.
    pub rk: u32,
    /// The `mues` (Global) index of this slot's mu value.  Always non-zero for a
    /// persisted slot (zero-mu slots are dropped).
    pub mu: u32,
}

impl SavedRelKl {
    /// Recompute the induced-element perms from `elms` words via `g`.
    ///
    /// Perms are derivable from the stored words, so they are not persisted; a
    /// consumer that needs them (e.g. to feed `decompose_tiered`) calls this.
    pub fn perms(&self, g: &crate::group::CoxeterGroup) -> Vec<crate::element::Perm> {
        self.elms.iter().map(|w| g.word_to_perm(w)).collect()
    }

    /// Rebuild the full [`RelKlInput`] (`Global` form) from the sparse slots.
    ///
    /// The `klmat` is a flat strict-lower-triangular matrix (`klmat[fy]` has
    /// length `fy`).  Every persisted sparse slot becomes
    /// `Some(SlotData { mu: vec![mu] })`; every other position is `None`.
    ///
    /// LOSSY: dropped slots (both `'0c0'` zero and `'f'` absent in the original)
    /// all reconstruct as `None`.  This is exact for the W-graph / cells payload —
    /// `from_relkl` produces a byte-identical [`CellGraph`] — but loses the
    /// zero-vs-incomparable distinction (see the [module docs](self)).
    pub fn to_relkl_input(&self) -> RelKlInput {
        let n = self.n;
        let mut klmat: Vec<Vec<KlSlot>> = (0..n).map(|fy| vec![None; fy]).collect();
        for s in &self.slots {
            let fy = s.flat_y as usize;
            let fx = s.flat_x as usize;
            // Defensive: ignore any out-of-range / non-lower-triangular triplet
            // rather than panic on a hand-corrupted file (load already validated
            // ranges, but to_relkl_input may run on a constructed value).
            if fy < n && fx < fy {
                klmat[fy][fx] = Some(SlotData { mu: vec![s.mu] });
            }
        }
        RelKlInput {
            elms: self.elms.clone(),
            klmat,
            mpols: MuPools::Global(self.mues.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// save_relkl
// ---------------------------------------------------------------------------

/// Persist `out` as a sparse, self-contained archive at `path` (gz-compressed).
///
/// Only slots whose Global `mu` index is non-zero are stored; the rest are
/// reconstructed as `None` on load.  See the [module docs](self) for the exact
/// wire format and the (W-graph-irrelevant) lossiness of dropping zero slots.
///
/// `group_label` and `rep_tag` are provenance strings echoed back by
/// [`load_relkl`].
pub fn save_relkl(
    out: &RelKlOutput,
    group_label: &str,
    rep_tag: &str,
    path: &Path,
) -> io::Result<()> {
    let mues = match &out.input.mpols {
        MuPools::Global(pool) => pool,
        MuPools::PerGen(_) => {
            return Err(corrupt(
                "save_relkl expects a Global mu pool (relklpols output); got PerGen",
            ));
        }
    };

    let n = out.input.elms.len();
    let mut buf = Vec::new();

    // Header.
    buf.extend_from_slice(SAVE_MAGIC);
    buf.extend_from_slice(&SAVE_VERSION.to_le_bytes());
    write_str(&mut buf, group_label);
    write_str(&mut buf, rep_tag);
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    // Rank: derived from the group is not available here; we store the longest
    // element's word length is NOT the rank.  The induced words live in W's
    // generator labels, so the maximum generator label + 1 lower-bounds the rank.
    // We store it as a best-effort hint (provenance), not a load invariant.
    let rank_hint = out
        .input
        .elms
        .iter()
        .flat_map(|w| w.iter())
        .map(|&s| s as u32 + 1)
        .max()
        .unwrap_or(0);
    buf.extend_from_slice(&rank_hint.to_le_bytes());

    // Elements (words), count-prefixed for a self-describing frame.
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    for w in &out.input.elms {
        write_word(&mut buf, w);
    }

    // Pools.
    write_laurents(&mut buf, &out.rklpols);
    write_laurents(&mut buf, mues);

    // Sparse slots: only mu != 0.  The invariant `mu != 0 ⇒ pool[mu] is a real,
    // non-zero polynomial` (i.e. a W-graph-relevant slot) is checked in debug
    // builds: the relkl recursion only interns a non-zero mu when the slot's
    // relative-KL value is non-zero, so a non-zero mu index never points at the
    // zero pool entry.  (If this ever fired we would also need to store zero-mu
    // slots; it does not, per the E7-scale measurement and the B4/H3 probe.)
    let mut slots: Vec<SparseSlot> = Vec::new();
    for (fy, row) in out.input.klmat.iter().enumerate() {
        for (fx, slot) in row.iter().enumerate() {
            let Some(sd) = slot else { continue };
            let mu = sd.mu.first().copied().unwrap_or(0);
            if mu == 0 {
                continue; // dropped (zero / no-edge) slot
            }
            debug_assert!(
                !mues[mu as usize].is_zero(),
                "save_relkl sparse invariant: mu index {mu} at ({fy},{fx}) points at \
                 a zero pool entry — zero-mu slots must be dropped, not stored"
            );
            slots.push(SparseSlot {
                flat_y: fy as u32,
                flat_x: fx as u32,
                rk: 0, // see SparseSlot::rk — not available from the output klmat
                mu,
            });
        }
    }
    buf.extend_from_slice(&(slots.len() as u32).to_le_bytes());
    for s in &slots {
        buf.extend_from_slice(&s.flat_y.to_le_bytes());
        buf.extend_from_slice(&s.flat_x.to_le_bytes());
        buf.extend_from_slice(&s.rk.to_le_bytes());
        buf.extend_from_slice(&s.mu.to_le_bytes());
    }

    // gz-compress + write atomically (tmp + rename) so a crash mid-write never
    // leaves a half-written archive at `path`.
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let tmp = tmp_path(path);
    {
        let file = File::create(&tmp)?;
        let mut enc = GzEncoder::new(file, Compression::default());
        enc.write_all(&buf)?;
        let file = enc.finish()?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// load_relkl
// ---------------------------------------------------------------------------

/// Load a sparse relkl archive written by [`save_relkl`].
///
/// Decompresses the gz file, validates the magic + version, and decodes the
/// header, words, pools, and sparse slots.  Range-checks every slot triplet
/// (`flat_x < flat_y < n`) so a corrupt file errors instead of producing an
/// out-of-bounds [`to_relkl_input`](SavedRelKl::to_relkl_input).
pub fn load_relkl(path: &Path) -> io::Result<SavedRelKl> {
    let file = File::open(path)?;
    let mut dec = GzDecoder::new(file);
    let mut buf = Vec::new();
    dec.read_to_end(&mut buf)?;

    let mut r = Cursor { buf: &buf, pos: 0 };
    if r.next_bytes(8)? != SAVE_MAGIC {
        return Err(corrupt("bad magic — not a relkl save archive"));
    }
    let version = r.read_u32()?;
    if version != SAVE_VERSION {
        return Err(corrupt(format!(
            "unsupported relkl-save version {version} (expected {SAVE_VERSION})"
        )));
    }
    let group_label = read_str(&mut r)?;
    let rep_tag = read_str(&mut r)?;
    let n = r.read_u32()? as usize;
    let rank = r.read_u32()? as usize;

    let elms_count = r.read_u32()? as usize;
    if elms_count != n {
        return Err(corrupt(format!(
            "elms count {elms_count} != n {n}"
        )));
    }
    let mut elms: Vec<Word> = Vec::with_capacity(n);
    for _ in 0..n {
        elms.push(read_word(&mut r)?);
    }

    let rklpols = read_laurents(&mut r)?;
    let mues = read_laurents(&mut r)?;

    let nslots = r.read_u32()? as usize;
    let mut slots: Vec<SparseSlot> = Vec::with_capacity(nslots);
    for _ in 0..nslots {
        let flat_y = r.read_u32()?;
        let flat_x = r.read_u32()?;
        let rk = r.read_u32()?;
        let mu = r.read_u32()?;
        // Validate: strict lower triangular, in range, non-zero mu.
        if (flat_y as usize) >= n || flat_x >= flat_y {
            return Err(corrupt(format!(
                "slot ({flat_y},{flat_x}) out of strict-lower-triangular range for n={n}"
            )));
        }
        if mu == 0 || (mu as usize) >= mues.len() {
            return Err(corrupt(format!(
                "slot ({flat_y},{flat_x}) has invalid mu index {mu} (pool len {})",
                mues.len()
            )));
        }
        if rk as usize >= rklpols.len() {
            return Err(corrupt(format!(
                "slot ({flat_y},{flat_x}) has rk index {rk} out of rklpols range {}",
                rklpols.len()
            )));
        }
        slots.push(SparseSlot {
            flat_y,
            flat_x,
            rk,
            mu,
        });
    }

    Ok(SavedRelKl {
        group_label,
        rep_tag,
        n,
        rank,
        elms,
        rklpols,
        mues,
        slots,
    })
}

// ---------------------------------------------------------------------------
// Small framing helpers (strings + words)
// ---------------------------------------------------------------------------

fn write_str(out: &mut Vec<u8>, s: &str) {
    let b = s.as_bytes();
    out.extend_from_slice(&(b.len() as u32).to_le_bytes());
    out.extend_from_slice(b);
}

fn read_str(r: &mut Cursor<'_>) -> io::Result<String> {
    let len = r.read_u32()? as usize;
    let bytes = r.next_bytes(len)?.to_vec();
    String::from_utf8(bytes).map_err(|_| corrupt("string is not valid UTF-8"))
}

/// A word is `len (u32) + len× generator byte (u8)` (generators are `u8`).
fn write_word(out: &mut Vec<u8>, w: &Word) {
    out.extend_from_slice(&(w.len() as u32).to_le_bytes());
    out.extend_from_slice(w);
}

fn read_word(r: &mut Cursor<'_>) -> io::Result<Word> {
    let len = r.read_u32()? as usize;
    Ok(r.next_bytes(len)?.to_vec())
}

/// The temp path a [`save_relkl`] writes to before the atomic rename.
fn tmp_path(path: &Path) -> std::path::PathBuf {
    let mut os = path.as_os_str().to_owned();
    os.push(".tmp");
    std::path::PathBuf::from(os)
}

// ---------------------------------------------------------------------------
// Tests (unit): in-memory framing round-trips that need no group.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn str_and_word_roundtrip() {
        let mut buf = Vec::new();
        write_str(&mut buf, "E8");
        write_str(&mut buf, "rep000042");
        write_word(&mut buf, &vec![0u8, 3, 7, 2]);
        write_word(&mut buf, &Vec::<u8>::new());
        let mut r = Cursor { buf: &buf, pos: 0 };
        assert_eq!(read_str(&mut r).unwrap(), "E8");
        assert_eq!(read_str(&mut r).unwrap(), "rep000042");
        assert_eq!(read_word(&mut r).unwrap(), vec![0u8, 3, 7, 2]);
        assert_eq!(read_word(&mut r).unwrap(), Vec::<u8>::new());
        assert_eq!(r.pos, buf.len());
    }

    /// A hand-built tiny `RelKlOutput` round-trips through save→load and rebuilds
    /// the expected sparse `klmat` (the dropped `mu==0` slot is `None`).
    #[test]
    fn save_load_tiny() {
        // n = 3, one nonzero slot at (2,0), one zero slot at (1,0) that must drop.
        let mues = vec![
            Laurent::zero(),
            Laurent::one(),
            Laurent::monomial(1, 0), // mu index 2 — a real value.
        ];
        let klmat: Vec<Vec<KlSlot>> = vec![
            vec![],
            vec![Some(SlotData { mu: vec![0] })],            // (1,0): mu==0 → drop
            vec![None, Some(SlotData { mu: vec![2] })],      // (2,1): mu==2 → keep
        ];
        let input = RelKlInput {
            elms: vec![vec![], vec![0], vec![1, 0]],
            klmat,
            mpols: MuPools::Global(mues.clone()),
        };
        let out = RelKlOutput {
            input,
            perms: Vec::new(),
            rklpols: vec![Laurent::zero(), Laurent::one()],
            mues,
            stats: super::super::RelKlStats::default(),
        };

        let dir = std::env::temp_dir().join(format!("rcx_relkl_save_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rep000000.relkl.gz");

        save_relkl(&out, "TINY", "rep000000", &path).unwrap();
        let loaded = load_relkl(&path).unwrap();

        assert_eq!(loaded.group_label, "TINY");
        assert_eq!(loaded.rep_tag, "rep000000");
        assert_eq!(loaded.n, 3);
        assert_eq!(loaded.elms, out.input.elms);
        assert_eq!(loaded.rklpols, out.rklpols);
        assert_eq!(loaded.mues, out.mues);
        assert_eq!(loaded.slots.len(), 1);
        assert_eq!(loaded.slots[0].flat_y, 2);
        assert_eq!(loaded.slots[0].flat_x, 1);
        assert_eq!(loaded.slots[0].mu, 2);

        // to_relkl_input drops the zero slot (1,0) and keeps (2,1).
        let rebuilt = loaded.to_relkl_input();
        assert!(rebuilt.klmat[1][0].is_none(), "zero slot must reconstruct as None");
        assert_eq!(
            rebuilt.klmat[2][1],
            Some(SlotData { mu: vec![2] }),
            "nonzero slot must reconstruct exactly"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let dir = std::env::temp_dir().join(format!("rcx_relkl_save_mag_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.relkl.gz");
        // Write a gz file with bogus contents.
        let file = File::create(&path).unwrap();
        let mut enc = GzEncoder::new(file, Compression::default());
        enc.write_all(b"NOTRKLSAVExxxxxxxxxxx").unwrap();
        enc.finish().unwrap();
        assert!(load_relkl(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
