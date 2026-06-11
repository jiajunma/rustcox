//! Relative Kazhdan–Lusztig polynomials for parabolic induction (Task P4).
//!
//! Exact port of PyCox `relklpols` (`pycox-ref/pycox_ref.py` 10496–10773) and
//! `relmue` (10483–10494), **equal parameters only**.  On any discrepancy the
//! Python source wins.  The normative extraction is
//! `docs/superpowers/plans/2026-06-11-pycox-relklpols-notes.md`.
//!
//! Given a Coxeter group `W`, a parabolic subgroup `W1 = W_J ⊂ W`, and a left
//! cell (or union of left cells) `C` of `W1` described by a [`RelKlInput`]
//! (`cell1`), [`relklpols`] computes the relative KL polynomials of the induced
//! set `X1·C`, where `X1` is the set of minimal-length left coset
//! representatives of `W1` in `W`.  By Geck's induction theorem, `X1·C` is a
//! union of left cells of `W`.  The output [`RelKlOutput::input`] is a
//! [`RelKlInput`] in [`MuPools::Global`] form, ready to feed into
//! [`CellGraph::from_relkl`](crate::cellgraph::CellGraph::from_relkl).
//!
//! # The five index spaces
//!
//! The recursion juggles five distinct index spaces.  We keep them straight with
//! named type aliases and a disciplined naming convention:
//!
//! | space            | meaning                                   | alias / var |
//! |------------------|-------------------------------------------|-------------|
//! | W-generator      | a simple generator of `W`                 | `s` (`Gen`) |
//! | coset index      | position in `X1` (coset reps)             | `x`, `y` (`Cx`) |
//! | cell index       | position in `cell1.elms` (elements of `C`)| `u`, `v` (`Cu`) |
//! | flat `ap` index  | position in the induced set `X·C`         | `u32` |
//! | W1 element       | a perm of `W1` in `W1`'s own root system  | `p1[u]` |
//!
//! The W1-local generator space appears only transiently in the `lft1` lookups;
//! we resolve it to W-generator indices at the boundary (see below).
//!
//! # The `lft` / `lft1` keying convention (the subtle part)
//!
//! `Lft` (defined in the `relkl_recur` submodule) encodes
//! left-multiplication of a coset rep by a W-generator:
//! - `Lft::In(x)` — `s·X1[x]` stays in `X1` at coset index `x`;
//! - `Lft::Out(t)` — `s·X1[x]` leaves `X1`; it equals `X1[x]·t'` for a unique
//!   W1-generator `t'`, and `t` is the **W-generator index** `J[t']` (i.e. the
//!   *global* generator `gen_map[t']`).
//!
//! PyCox encodes this case as the integer `-t-1` and keys the `lft1` dictionary
//! by `J[t]` (the W-index).  We adopt that same W-index convention: `lft1` is a
//! `Vec` indexed by W-generator (length `W.rank`), with only the `J`-entries
//! populated, so `lft1[t]` for a `Lft::Out(t)` payload works directly with no
//! local/global conversion.  This matches the PyCox lookups `lft1[-1-sx]` and
//! `lft1[t]` (where `t = -1-sx`) verbatim.

use std::collections::HashMap;

use crate::{
    bruhat,
    cellgraph::{KlSlot, MuPools, RelKlInput, SlotData},
    element::{Gen, Perm, Word},
    group::CoxeterGroup,
    laurent::Laurent,
    parabolic::{red_left_coset_reps, Parabolic},
};

use rayon::prelude::*;

use super::relkl_ckpt::{apply_replay, load_and_replay, BlkLogWriter, LayerRecord, RelKlCkptCfg};
use super::relkl_recur::{
    classify_block, compute_caseb_block, diag_block_mu, intern, relmue, CaseBBlock, Cu, Cx,
    LayerCtx, Lft, SlotState, XBlockKind,
};

/// The working KL matrix: present blocks `(y, x)` → an `nc×nc` slot grid.
type Mat = HashMap<(Cx, Cx), Vec<Vec<SlotState>>>;

// ---------------------------------------------------------------------------
// Public options + output
// ---------------------------------------------------------------------------

/// Options for [`relklpols`].
///
/// `threads` controls the `y`-wavefront parallelism (Task P6).  The output is
/// **byte-identical** for any thread count — only *when* a block is computed
/// changes, never the order in which polynomials are interned.  Convention
/// (matching the Phase-1 KL driver [`klpolynomials`](crate::kl::klpolynomials)):
/// - `None`        → the global Rayon pool;
/// - `Some(0 | 1)` → fully sequential (no pool overhead);
/// - `Some(t > 1)` → a private pool of `t` threads for this call.
#[derive(Clone, Debug, Default)]
pub struct RelKlOpts {
    /// Thread count for the `y`-wavefront.  See the type docs for the
    /// convention.  `None` ⇒ global pool; `Some(0|1)` ⇒ sequential.
    pub threads: Option<usize>,
}

/// Output of [`relklpols`].
///
/// `input` is the [`RelKlInput`] contract consumed by
/// [`CellGraph::from_relkl`](crate::cellgraph::CellGraph::from_relkl): its
/// `elms` are the induced-set canonical words sorted by **length** (stable),
/// `klmat` is the flat strict-lower-triangular matrix with single-`Global`-index
/// [`SlotData`]s, and `mpols` is [`MuPools::Global`] (the `mues` pool, seeded
/// `[zero, one]`).
///
/// `perms` matches `input.elms` order.  `rklpols` is the relative-KL polynomial
/// pool, seeded `[zero, one]`.
///
/// # Why only `input` + `perms`
///
/// Task P5 (`klcells`) consumes the induced graph by building a
/// [`CellGraph`](crate::cellgraph::CellGraph) via `from_relkl(output.input)` and
/// decomposing it; it constructs its own `pairs`/involution data independently
/// from the group, so it needs only `input` (the graph) and `perms` (the
/// element identities).  Per YAGNI we expose exactly that, plus the two pools
/// for inspection/testing.  `elmsX` (coset-rep words) and the `(y, v) → flat`
/// bijection are internal to the recursion and not re-exposed.
#[derive(Clone, Debug)]
pub struct RelKlOutput {
    /// The induced-cell W-graph in `RelKlInput`/`Global` form.
    pub input: RelKlInput,
    /// Perms of `input.elms`, in the same order.
    pub perms: Vec<Perm>,
    /// The relative-KL polynomial pool, seeded `[zero, one]`.
    pub rklpols: Vec<Laurent>,
    /// The global mu pool (`mues`), seeded `[zero, one]`.
    pub mues: Vec<Laurent>,
    /// Cheap slot-occupancy + memory stats for this call (always collected).
    pub stats: RelKlStats,
}

/// Cheap, always-on statistics about one [`relklpols`] call.
///
/// Counts the working-matrix slots by state at the end of the recursion and the
/// peak block-memory estimate.  These feed the future sparse-encoding decision
/// (most slots in big inductions are `absent`); the CLI surfaces them at
/// `--summary` as `relkl_slots: absent=… zero=… nonzero=…`.
///
/// Definitions (over every present working block `(y, x)`, all `nc×nc` slots):
/// - `absent`  — `'f'` slots (no entry / [`SlotState::Absent`]);
/// - `zero`    — completed slots whose relative-KL value is zero (`'0c0'`,
///   i.e. [`SlotState::Done`] with `rk == 0`);
/// - `nonzero` — completed slots with a non-zero relative-KL value
///   ([`SlotState::Done`] with `rk != 0`).
///
/// `peak_block_bytes` estimates the largest in-flight `mat` footprint
/// (`present_blocks × nc² × size_of::<SlotState>()`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RelKlStats {
    /// Count of absent (`'f'`) slots.
    pub absent: u64,
    /// Count of completed zero-valued slots.
    pub zero: u64,
    /// Count of completed non-zero-valued slots.
    pub nonzero: u64,
    /// Peak working-matrix memory estimate, in bytes.
    pub peak_block_bytes: u64,
}

impl RelKlStats {
    /// Merge another call's stats into this one (used by `klcells` aggregation):
    /// counts add; `peak_block_bytes` is the max (peaks are not concurrent
    /// across reps — only one relklpols runs at a time).
    pub fn merge(&mut self, other: &RelKlStats) {
        self.absent += other.absent;
        self.zero += other.zero;
        self.nonzero += other.nonzero;
        self.peak_block_bytes = self.peak_block_bytes.max(other.peak_block_bytes);
    }
}

/// Outcome of a resumable [`relklpols_resumable`] run.
#[derive(Clone, Debug)]
pub enum RelKlRunOutcome {
    /// The whole call completed; the log files (if any) have been deleted.
    Done(Box<RelKlOutput>),
    /// The run stopped after completing the named layer (only via the test-only
    /// `RelKlCkptCfg::test_stop_after_layer` hook).  The block log + header are
    /// durable on disk so a follow-up resume continues from `last_layer + 1`.
    Stopped { last_layer: usize },
}

/// A `(group, W1, cell1)` content fingerprint binding a block log to its rep.
///
/// Two runs share a relkl block log iff this string matches.  It must detect a
/// *different rep or cell* (the task's hard requirement): it folds in the group
/// type, the parabolic `J`, and a content hash of `cell1` (its element words,
/// the filled-slot structure, and the mu-pool Laurents).  Thread count is
/// excluded — the output is byte-identical across thread counts.
pub fn relkl_fingerprint(w: &CoxeterGroup, w1: &Parabolic, cell1: &RelKlInput) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();

    // Group type (series + rank per component) + order.
    for c in &w.components {
        format!("{}{}", c.series, c.indices.len()).hash(&mut h);
    }
    w.order.hash(&mut h);
    // Parabolic generators J (W-generator indices).
    w1.gen_map().hash(&mut h);

    // cell1 content: element words.
    for word in &cell1.elms {
        word.hash(&mut h);
    }
    // cell1 filled-slot structure: (row, col, mu-indices) for each present slot.
    for (i, row) in cell1.klmat.iter().enumerate() {
        for (jj, slot) in row.iter().enumerate() {
            if let Some(sd) = slot {
                i.hash(&mut h);
                jj.hash(&mut h);
                sd.mu.hash(&mut h);
            }
        }
    }
    // cell1 mu pools (Laurent val + coeffs), per pool.
    let hash_pool = |pool: &[Laurent], h: &mut std::collections::hash_map::DefaultHasher| {
        for p in pool {
            p.val().hash(h);
            p.coeffs().hash(h);
        }
    };
    match &cell1.mpols {
        MuPools::PerGen(pools) => {
            "pergen".hash(&mut h);
            for pool in pools {
                hash_pool(pool, &mut h);
            }
        }
        MuPools::Global(pool) => {
            "global".hash(&mut h);
            hash_pool(pool, &mut h);
        }
    }

    let mut s = String::from("rustcox-relkl-v1|");
    for (k, c) in w.components.iter().enumerate() {
        if k > 0 {
            s.push('x');
        }
        s.push_str(&format!("{}{}", c.series, c.indices.len()));
    }
    s.push_str(&format!("|nc={}|h={:016x}", cell1.elms.len(), h.finish()));
    s
}

// ---------------------------------------------------------------------------
// relklpols
// ---------------------------------------------------------------------------

/// Relative KL polynomials of the induced set `X1·C`.
///
/// `cell1` describes the left cell (or union of cells) `C` of `W1` as a
/// [`RelKlInput`] in [`MuPools::PerGen`] form (the output of
/// [`CellGraph::to_relkl`](crate::cellgraph::CellGraph::to_relkl)).  Its `elms`
/// are reduced words in `W1`'s **own** generator labels.
///
/// Equal parameters only: all generator weights are implicitly `1` (PyCox's
/// `weightL = 1` branch).  No weights parameter is taken; the `Lw`/`Lw1` length
/// sums below are plain word lengths.
///
/// `opts.threads` selects the wavefront thread pool (see [`RelKlOpts`]); the
/// output is byte-identical for every thread count.
pub fn relklpols(
    w: &CoxeterGroup,
    w1: &Parabolic,
    cell1: &RelKlInput,
    opts: &RelKlOpts,
) -> RelKlOutput {
    // No checkpointing: run to completion.  `Done` is the only possible outcome
    // when `ckpt` is `None` (the stop hook lives on the cfg), so unwrap it.
    match run_threaded(w, w1, cell1, opts, None) {
        RelKlRunOutcome::Done(out) => *out,
        RelKlRunOutcome::Stopped { .. } => {
            unreachable!("relklpols without a checkpoint cfg cannot stop early")
        }
    }
}

/// Resumable [`relklpols`]: layer-granular checkpoint/resume (Task Q4).
///
/// Behaves exactly like [`relklpols`] but, when `ckpt` is `Some`, logs each
/// completed wavefront layer to `dir/<rep_tag>.blklog` (+ `.blkhdr`) so a call
/// that exceeds its time box can be paused and resumed.  On entry, if a valid
/// matching log exists, the wavefront replays it and continues at the next
/// uncomputed layer; on clean completion the log files are deleted (bounded
/// disk: only the in-flight rep keeps a log).
///
/// Returns [`RelKlRunOutcome::Done`] with the full output on completion, or
/// [`RelKlRunOutcome::Stopped`] when the test-only
/// [`RelKlCkptCfg::test_stop_after_layer`] hook fires.
///
/// A fingerprint mismatch or a corrupt/truncated log is non-fatal: the run logs
/// a warning to stderr, deletes the stale files, and starts fresh.  The output
/// is **byte-identical** to an uninterrupted [`relklpols`] call regardless of
/// how many times the run was interrupted and resumed.
pub fn relklpols_resumable(
    w: &CoxeterGroup,
    w1: &Parabolic,
    cell1: &RelKlInput,
    opts: &RelKlOpts,
    ckpt: Option<&RelKlCkptCfg>,
) -> RelKlRunOutcome {
    run_threaded(w, w1, cell1, opts, ckpt)
}

/// Apply the thread-pool convention, then run the (optionally resumable) body.
fn run_threaded(
    w: &CoxeterGroup,
    w1: &Parabolic,
    cell1: &RelKlInput,
    opts: &RelKlOpts,
    ckpt: Option<&RelKlCkptCfg>,
) -> RelKlRunOutcome {
    // Threading convention (matches the Phase-1 KL driver): Some(0|1) runs the
    // recursion directly (no pool); Some(t>1) installs a private t-thread pool;
    // None uses the ambient/global pool.  Determinism is independent of this
    // choice — only `bruhatX` and the per-layer Case-B compute use the pool.
    match opts.threads {
        Some(0) | Some(1) => relklpols_inner(w, w1, cell1, ckpt),
        Some(t) => {
            // A private pool isolates this call; if the builder fails (rare) we
            // fall back to the ambient pool rather than erroring out — the math
            // is identical either way.
            match rayon::ThreadPoolBuilder::new().num_threads(t).build() {
                Ok(pool) => pool.install(|| relklpols_inner(w, w1, cell1, ckpt)),
                Err(_) => relklpols_inner(w, w1, cell1, ckpt),
            }
        }
        None => relklpols_inner(w, w1, cell1, ckpt),
    }
}

/// The recursion body (runs inside the caller-selected Rayon pool).
// The recursion is a faithful matrix port of PyCox `relklpols`: the positional
// `(y, x, v, u)` index loops mirror the source line-for-line and index multiple
// parallel structures (`mat`, `bruhatX`, `bij`, the coset/cell tables), so
// iterator rewrites would obscure the correspondence rather than clarify it.
#[allow(clippy::needless_range_loop)]
fn relklpols_inner(
    w: &CoxeterGroup,
    w1: &Parabolic,
    cell1: &RelKlInput,
    ckpt: Option<&RelKlCkptCfg>,
) -> RelKlRunOutcome {
    debug_assert!(
        matches!(cell1.mpols, MuPools::PerGen(_)),
        "relklpols expects cell1 in PerGen form (from CellGraph::to_relkl)"
    );

    let rank = w.rank;
    let j = w1.gen_map(); // J[t'] = W-generator index of W1-local generator t'.

    // --- Setup: coset reps X1 ------------------------------------------------
    let x1w: Vec<Word> = red_left_coset_reps(w, &w1.sub_j);
    let x1: Vec<Perm> = x1w.iter().map(|word| w.word_to_perm(word)).collect();
    let nx = x1.len();
    // Lw[x] = ℓ_W(X1w[x]) (= word length, equal params).
    let lw: Vec<u32> = x1w.iter().map(|word| word.len() as u32).collect();

    // Index X1 by coxelm for the `s·X1[x] ∈ X1?` membership test.
    let x1_pos: HashMap<_, Cx> = x1
        .iter()
        .enumerate()
        .map(|(i, p)| (p.coxelm_sr(&w.simple_root), i))
        .collect();

    // lft[s][x]: see `Lft`.  s over W.rank, x over X1.
    let lft: Vec<Vec<Lft>> = (0..rank)
        .map(|s| {
            (0..nx)
                .map(|x| {
                    // sw = s · X1[x] (LEFT multiply): PyCox `[w[i] for i in
                    // permgens[s]]` = then(permgens[s], X1[x]).
                    let sw = w.permgens[s].then(&x1[x]);
                    let sw_ce = sw.coxelm_sr(&w.simple_root);
                    if let Some(&xi) = x1_pos.get(&sw_ce) {
                        Lft::In(xi)
                    } else {
                        // Leaves X: sw = X1[x]·t for a unique W-generator t (in J).
                        // PyCox `[permgens[t][i] for i in w]` = then(X1[x],
                        // permgens[t]) = X1[x]·t (RIGHT multiply) — NOT t·X1[x].
                        let t = (0..rank)
                            .find(|&t| {
                                x1[x].then(&w.permgens[t]).coxelm_sr(&w.simple_root) == sw_ce
                            })
                            .expect("s·X1[x] leaves X yet no W-generator realises it");
                        debug_assert!(
                            j.contains(&(t as Gen)),
                            "lft Out(t): t={t} not in J={j:?} (s={s}, x={x})"
                        );
                        Lft::Out(t as Gen)
                    }
                })
                .collect()
        })
        .collect();

    // --- Setup: cell C in W1 -------------------------------------------------
    let nc = cell1.elms.len();
    // Lw1[u] = Σ poids[J[s]] over cell1.elms[u]; equal params ⇒ word length.
    let lw1: Vec<u32> = cell1.elms.iter().map(|word| word.len() as u32).collect();
    // p1[u] = W1.wordtoperm(local word) — perms in W1's OWN root system.
    let p1: Vec<Perm> = cell1
        .elms
        .iter()
        .map(|word| w1.group.word_to_perm(word))
        .collect();
    let p1_pos: HashMap<_, Cu> = p1
        .iter()
        .enumerate()
        .map(|(i, p)| (p.coxelm_sr(&w1.group.simple_root), i))
        .collect();

    // lft1[J[t']][u]: left-mult of p1[u] by W1-local generator t'.
    //   in p1 → its index; else descends-out (-1 → encode `nc` sentinel? no:
    //   PyCox uses -1 and len(p1)).  We store the raw signed-ish result.
    // PyCox stores -1 (descends out) and len(p1) (ascends out).  We mirror with
    // an enum-free i64 so the `u < lft1[...][u]` and `lft1[t][w] > w` comparisons
    // port verbatim.
    const DESC_OUT: i64 = -1; // PyCox -1: t'·u < u, leaves the cell downward.
    let asc_out: i64 = nc as i64; // PyCox len(p1): t'·u > u, leaves the cell upward.
    let n1 = w1.group.n_pos as usize; // W1.N
                                      // lft1 keyed by W-generator (length rank); only J-entries are filled.
    let mut lft1: Vec<Vec<i64>> = vec![Vec::new(); rank];
    for (t_local, &t_w) in j.iter().enumerate() {
        let col: Vec<i64> = (0..nc)
            .map(|u| {
                // w1elt = t'·p1[u] (left multiply) in W1's root system.
                let w1elt = w1.group.permgens[t_local].then(&p1[u]);
                let ce = w1elt.coxelm_sr(&w1.group.simple_root);
                if let Some(&ui) = p1_pos.get(&ce) {
                    ui as i64
                } else if (p1[u].0[w1.group.simple_root[t_local]] as usize) >= n1 {
                    // p1[u][t'] >= W1.N  ⇒  t'·p1[u] < p1[u]  ⇒  descends out.
                    DESC_OUT
                } else {
                    asc_out
                }
            })
            .collect();
        lft1[t_w as usize] = col;
    }

    // --- bruhatX[y][x] = Bruhat(X1[x] <= X1[y]) for x <= y -------------------
    // Stored as a lower-triangular table bx[y][x] (row y has length y+1).  The
    // rows are mutually independent `leq` calls, so we build them in parallel
    // (Task P6).  The result is deterministic: each row is a pure function of
    // the coset reps, and `par_iter` preserves index order on collect.
    let bruhat_x: Vec<Vec<bool>> = (0..nx)
        .into_par_iter()
        .map(|y| {
            (0..=y)
                .map(|x| bruhat::leq(w, &x1[x], &x1[y]))
                .collect::<Vec<bool>>()
        })
        .collect();
    // Helper closure to query bruhatX with x <= y (only valid then).
    // bruhat_x[y] has length y+1 (indices x = 0..=y).  The recursion only ever
    // queries pairs with x <= y (a mathematical invariant — `s·x` ascending
    // stays ≤ y for s ∈ ldy), but we guard defensively so an unexpected x > y
    // returns `false` rather than panicking.
    let bx = |y: Cx, x: Cx| -> bool { x <= y && bruhat_x[y][x] };

    // --- Matrix init ---------------------------------------------------------
    // mat[(y,x)] = nc×nc grid of SlotState; only (y,x) with bruhatX present.
    // Diagonal blocks (y,y) always present.  Factored into a closure so the
    // resume path can rebuild the deterministic post-init state cheaply when a
    // stale/corrupt log must be discarded.  Returns `(mat, mues)`.
    let build_init = || -> (Mat, Vec<Laurent>) {
        let mut mues: Vec<Laurent> = vec![Laurent::zero(), Laurent::one()];
        let mut mat: Mat = HashMap::new();
        for y in 0..nx {
            for x in 0..y {
                if bx(y, x) {
                    let mut grid = vec![vec![SlotState::Absent; nc]; nc];
                    for v in 0..nc {
                        for u in 0..nc {
                            // PyCox: (x==y and u==v) or Lw[x]+Lw1[u] < Lw[y]+Lw1[v].
                            // Here x<y so the first disjunct is false.
                            if lw[x] + lw1[u] < lw[y] + lw1[v] {
                                grid[v][u] = SlotState::Pending;
                            }
                        }
                    }
                    mat.insert((y, x), grid);
                }
            }
            // Diagonal block (y,y): copied from cell1.
            let mut diag = vec![vec![SlotState::Absent; nc]; nc];
            for i in 0..nc {
                for jj in 0..i {
                    if let Some(slot) = cell1.klmat[i][jj].as_ref() {
                        // PyCox: mat[y,y][i][j]='c0'; then read the FIRST generator
                        // r with a non-''/'0' slot index; intern its mu into mues.
                        let mu_idx = diag_block_mu(slot, &cell1.mpols, &mut mues);
                        diag[i][jj] = SlotState::Done { rk: 0, mu: mu_idx };
                    }
                }
                // Diagonal of diagonal: 'c1c0' → rk=1 (one), mu=0 (zero).
                diag[i][i] = SlotState::Done { rk: 1, mu: 0 };
            }
            mat.insert((y, y), diag);
        }
        (mat, mues)
    };
    let (mut mat, mut mues) = build_init();

    // --- Main recursion (wavefront over y; parallel over x per layer) --------
    //
    // Determinism (mirrors the Phase-1 KL driver's two-phase pattern): for a
    // fixed `y`, every Case-B block `(y, x)` is a *pure* function of the frozen
    // lower layers (`mat` blocks with first index `< y`) plus the cell diagonal
    // `(0, 0)`.  We therefore compute all Case-B blocks of a layer **in
    // parallel** producing INLINE Laurent values (no pool writes), then intern
    // them **sequentially** in the exact `(x desc, v, u)` order the sequential
    // reference uses — so `rklpols`/`mues` grow identically for any thread
    // count.  Case-A blocks copy the `rk` of the same-layer block `(y, sx)`
    // (`sx > x`); that read is a same-`y` dependency, so Case A is interned
    // inline during the sequential walk (it is the cheap branch — no `rklpols`
    // growth, only a `mu`).
    let mut rklpols: Vec<Laurent> = vec![Laurent::zero(), Laurent::one()];

    // --- Resume: replay an existing block log, if one matches this rep --------
    //
    // The setup above (x1, lft, lft1, bruhat_x, the initial `mat` Pending grids,
    // the diagonal `(y,y)` Done blocks, and the init-seeded `mues` prefix) is
    // fully deterministic, so it is recomputed fresh on every run.  Resume then
    // overwrites the finalized off-diagonal blocks of completed layers and
    // re-grows the pools from the logged per-layer deltas; the wavefront
    // continues at `start_y = last_layer + 1`.
    let fingerprint = ckpt.map(|_cfg| relkl_fingerprint(w, w1, cell1));
    let mut start_y: usize = 0;
    let mut log_writer: Option<BlkLogWriter<'_>> = None;
    let mut resumed = false;
    if let (Some(cfg), Some(fp)) = (ckpt, fingerprint.as_ref()) {
        // Decide whether to resume from an existing log, and reset to a clean
        // post-init state on any failure before opening a fresh log.
        match load_and_replay(cfg, fp) {
            Ok(Some(state)) => match apply_replay(&state, &mut mat, &mut rklpols, &mut mues) {
                Ok(next_y) => {
                    start_y = next_y.min(nx);
                    resumed = true;
                    match BlkLogWriter::open_existing(cfg, fp, &state.header) {
                        Ok(wr) => log_writer = Some(wr),
                        Err(e) => {
                            eprintln!(
                                "relkl block log: cannot reopen '{}' for append ({e}); \
                                 starting fresh",
                                cfg.log_path().display()
                            );
                            cfg.delete_files();
                            (mat, mues) = build_init();
                            rklpols = vec![Laurent::zero(), Laurent::one()];
                            start_y = 0;
                            resumed = false;
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "relkl block log: replay of '{}' failed ({e}); starting fresh",
                        cfg.log_path().display()
                    );
                    cfg.delete_files();
                    (mat, mues) = build_init();
                    rklpols = vec![Laurent::zero(), Laurent::one()];
                }
            },
            Ok(None) => {
                // No prior log: start a fresh one.
                cfg.delete_files();
            }
            Err(e) => {
                // Corrupt / mismatched header: warn, discard, start fresh.
                eprintln!(
                    "relkl block log: ignoring '{}' ({e}); starting fresh",
                    cfg.hdr_path().display()
                );
                cfg.delete_files();
            }
        }
        if !resumed && log_writer.is_none() {
            // Fresh log (no valid prior state to continue).
            match BlkLogWriter::create(cfg, fp) {
                Ok(wr) => log_writer = Some(wr),
                Err(e) => {
                    eprintln!(
                        "relkl block log: cannot create '{}' ({e}); running without checkpoints",
                        cfg.log_path().display()
                    );
                }
            }
        }
    }
    let every_layers = ckpt.map(|c| c.every_layers.max(1)).unwrap_or(1);
    let stop_after = ckpt.and_then(|c| c.test_stop_after_layer);

    let mut stopped_at: Option<usize> = None;

    for y in start_y..nx {
        let ldy = w.left_descents(&x1[y]); // W-generators.

        // Descending x-list of present blocks, with each block classified.
        let xs: Vec<(Cx, XBlockKind)> = (0..y)
            .rev()
            .filter(|&x| bx(y, x))
            .map(|x| (x, classify_block(&ldy, &lft, x)))
            .collect();

        // Pool lengths at the START of this layer; the tail appended during
        // phase 2 is this layer's pool delta for the block log.
        let rk_base = rklpols.len();
        let mu_base = mues.len();

        // ---- Phase 1 (parallel): compute every Case-B block's inline h. -----
        // Keyed by x (descending order preserved by `par_iter` over `xs`).
        let layer_ctx = LayerCtx {
            y,
            ldy: &ldy,
            lw: &lw,
            lw1: &lw1,
            nc,
            bx: &bruhat_x,
            lft: &lft,
            lft1: &lft1,
            mat: &mat,
            mues: &mues,
            rklpols: &rklpols,
            cell1,
        };
        let caseb_blocks: Vec<Option<CaseBBlock>> = xs
            .par_iter()
            .map(|&(x, kind)| match kind {
                XBlockKind::CaseB => {
                    let marks = &mat[&(y, x)];
                    Some(compute_caseb_block(&layer_ctx, x, marks))
                }
                XBlockKind::CaseA { .. } => None,
            })
            .collect();

        // ---- Phase 2 (sequential, x desc then v, u): intern into the pools. -
        for (idx, &(x, kind)) in xs.iter().enumerate() {
            match kind {
                XBlockKind::CaseA { sx } => {
                    for v in 0..nc {
                        for u in 0..nc {
                            if !mat[&(y, x)][v][u].is_marked() {
                                continue;
                            }
                            // Source slot mat[y, sx][v][u] (same y-layer, sx>x,
                            // already finalized).
                            let src = if bx(y, sx) {
                                mat.get(&(y, sx)).map(|g| g[v][u])
                            } else {
                                None
                            };
                            let new_state = match src {
                                Some(s_state) if s_state.is_marked() => {
                                    let rk = s_state.rk().unwrap_or(0);
                                    let mu = if rk != 0 {
                                        let m = relmue(
                                            lw[y] + lw1[v],
                                            lw[x] + lw1[u],
                                            &rklpols[rk as usize],
                                        );
                                        intern(&mut mues, m)
                                    } else {
                                        0
                                    };
                                    SlotState::Done { rk, mu }
                                }
                                _ => SlotState::Done { rk: 0, mu: 0 }, // '0c0'
                            };
                            mat.get_mut(&(y, x)).unwrap()[v][u] = new_state;
                        }
                    }
                }
                XBlockKind::CaseB => {
                    let block = caseb_blocks[idx]
                        .as_ref()
                        .expect("Case-B block precomputed in phase 1");
                    for v in 0..nc {
                        for u in 0..nc {
                            let Some(h) = block.h[v][u].as_ref() else {
                                continue; // absent slot
                            };
                            let new_state = if h.is_zero() {
                                SlotState::Done { rk: 0, mu: 0 } // '0c0'
                            } else {
                                let rk = intern(&mut rklpols, h.clone());
                                let m = relmue(lw[y] + lw1[v], lw[x] + lw1[u], h);
                                let mu = intern(&mut mues, m);
                                SlotState::Done { rk, mu }
                            };
                            mat.get_mut(&(y, x)).unwrap()[v][u] = new_state;
                        }
                    }
                }
            }
        }

        // ---- Layer complete: log it (finalized off-diagonal blocks + pool
        //      deltas), respecting the `every_layers` cadence.  The final layer
        //      `y == nx-1` is always logged so a resume after a clean wavefront
        //      replays the whole computation if the call is re-entered.
        if let Some(wr) = log_writer.as_mut() {
            let is_last = y + 1 == nx;
            if is_last || (y + 1) % every_layers == 0 {
                let blocks: Vec<(Cx, Vec<Vec<SlotState>>)> =
                    xs.iter().map(|&(x, _)| (x, mat[&(y, x)].clone())).collect();
                let rec = LayerRecord {
                    y,
                    blocks,
                    rklpols_delta: rklpols[rk_base..].to_vec(),
                    mues_delta: mues[mu_base..].to_vec(),
                };
                if let Err(e) = wr.append_layer(&rec, rklpols.len(), mues.len()) {
                    // A log write failure must not corrupt the math: warn, drop
                    // the writer (run continues, just unrecoverable from here).
                    eprintln!(
                        "relkl block log: append for layer {y} failed ({e}); \
                               continuing without further checkpoints"
                    );
                    log_writer = None;
                }
            }
        }

        // ---- Test-only stop hook (simulate a SLURM kill at a layer boundary).
        if stop_after == Some(y) {
            stopped_at = Some(y);
            break;
        }
    }

    // Early stop (test hook): the log + header are durable; return without
    // building the (incomplete) output.  A follow-up resume continues the call.
    if let Some(last_layer) = stopped_at {
        return RelKlRunOutcome::Stopped { last_layer };
    }

    // --- Relabel: ap = X·C words sorted by length (stable) -------------------
    // ap-word for (y, v): reduce(X1w[y] ++ [J[s'] for s' in cell1.elms[v]]).
    let mut ap_pairs: Vec<((Cx, Cu), Word)> = Vec::with_capacity(nx * nc);
    for y in 0..nx {
        for v in 0..nc {
            let mut full = x1w[y].clone();
            full.extend(w1.word_to_w(&cell1.elms[v]));
            let reduced = w.perm_to_word(&w.word_to_perm(&full));
            ap_pairs.push(((y, v), reduced));
        }
    }
    // PyCox: ap.sort(key=len) — stable sort by length only.
    let mut order: Vec<usize> = (0..ap_pairs.len()).collect();
    order.sort_by_key(|&i| ap_pairs[i].1.len());
    let ap: Vec<Word> = order.iter().map(|&i| ap_pairs[i].1.clone()).collect();
    let ap_perms: Vec<Perm> = ap.iter().map(|word| w.word_to_perm(word)).collect();

    // bij[(y,v)] = ap1.index(permmult(X1[y], elms1[v])) where elms1[v] is the
    // W-perm of the cell element.  We find it by perm equality.
    let elms1: Vec<Perm> = cell1
        .elms
        .iter()
        .map(|word| w.word_to_perm(&w1.word_to_w(word)))
        .collect();
    // Index ap perms for fast lookup.
    let ap_pos: HashMap<_, u32> = ap_perms
        .iter()
        .enumerate()
        .map(|(i, p)| (p.coxelm_sr(&w.simple_root), i as u32))
        .collect();
    let mut bij: HashMap<(Cx, Cu), u32> = HashMap::new();
    for y in 0..nx {
        for v in 0..nc {
            // permmult(X1[y], elms1[v]) = then(X1[y], elms1[v])? PyCox permmult
            // is then().  But X1[y]·(cell elt) as a W element: word X1w[y]++cellword.
            let prod = x1[y].then(&elms1[v]);
            let ce = prod.coxelm_sr(&w.simple_root);
            let flat = *ap_pos
                .get(&ce)
                .expect("induced product perm must appear in ap");
            bij.insert((y, v), flat);
        }
    }

    // nmat: flat strict-lower-triangular (klmat[fy] of length fy) with single
    // Global-index SlotData.  PyCox copies every slot != 'f' (including '0c0'
    // zero slots) where bij[x,u] <= bij[y,v].
    let n_flat = ap.len();
    let mut klmat: Vec<Vec<KlSlot>> = (0..n_flat).map(|fy| vec![None; fy]).collect();
    for y in 0..nx {
        for x in 0..=y {
            if !bx(y, x) {
                continue;
            }
            let grid = match mat.get(&(y, x)) {
                Some(g) => g,
                None => continue,
            };
            for v in 0..nc {
                for u in 0..nc {
                    let fy = bij[&(y, v)] as usize;
                    let fx = bij[&(x, u)] as usize;
                    if fx <= fy && grid[v][u].is_marked() {
                        // Completed slot → store its mu index (single Global).
                        // Pending should not survive; treat defensively as zero.
                        let mu = grid[v][u].mu().unwrap_or(0);
                        // PyCox stores 'c<rk>c<mu>'; from_relkl reads mu[0].
                        if fx < fy {
                            klmat[fy][fx] = Some(SlotData { mu: vec![mu] });
                        }
                        // fx == fy is the diagonal (dropped — klmat[fy] has no fy).
                    }
                }
            }
        }
    }

    let input = RelKlInput {
        elms: ap,
        klmat,
        mpols: MuPools::Global(mues.clone()),
    };

    // --- Stats: slot occupancy over every present working block --------------
    // (absent vs completed-zero vs completed-nonzero) + peak block-memory.  Each
    // present block is `nc×nc` slots; `Pending` should not survive a completed
    // wavefront and is counted as `absent` defensively.
    let mut stats = RelKlStats::default();
    let mut present_blocks: u64 = 0;
    for grid in mat.values() {
        present_blocks += 1;
        for row in grid {
            for slot in row {
                match slot {
                    SlotState::Done { rk, .. } if *rk != 0 => stats.nonzero += 1,
                    SlotState::Done { .. } => stats.zero += 1,
                    _ => stats.absent += 1,
                }
            }
        }
    }
    let slot_bytes = std::mem::size_of::<SlotState>() as u64;
    stats.peak_block_bytes = present_blocks * (nc as u64) * (nc as u64) * slot_bytes;

    // --- Clean completion: drop the block log (bounded disk) -----------------
    // The writer is dropped first (closing the file handle), then both files are
    // removed.  Only the in-flight rep keeps a log, so a completed call leaves no
    // checkpoint residue behind.
    if let Some(cfg) = ckpt {
        drop(log_writer);
        cfg.delete_files();
    }

    RelKlRunOutcome::Done(Box::new(RelKlOutput {
        input,
        perms: ap_perms,
        rklpols,
        mues,
        stats,
    }))
}
