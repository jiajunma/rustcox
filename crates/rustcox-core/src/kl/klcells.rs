//! Left cells by parabolic induction — the `klcells` driver (Task P5).
//!
//! Ports PyCox `klcells` (`pycox-ref/pycox_ref.py` 12054–12303), equal-parameter
//! branch (`weightL = 1`).  On any discrepancy the Python source wins.
//!
//! # Algorithm (per the normative notes §klcells)
//!
//! `klcells(W)` computes the partition of `W` into left cells together with a
//! W-graph for one representative of each star-equivalence class.  It works
//! recursively, **never enumerating `W`**:
//!
//! 1. Pick `J = W.rank \ {one generator}` (the **E7 J-rule**: if the first
//!    component is type `E` with 7 nodes drop generator `0`; else drop the last
//!    generator).  Build the parabolic `W1 = W_J` and the distinguished left
//!    coset reps `X1` of `W1` in `W`.
//! 2. Recurse: `kk = klcells(W1, all_cells=false)` gives the star-class
//!    representative W-graphs of `W1` (`kk.star_reps`), whose vertices are
//!    `W1`-**local** words.
//! 3. For each `W1`-rep, induce its cell into `W` via [`relklpols`], build the
//!    induced W-graph ([`CellGraph::from_relkl`]), and [`decompose`] it into
//!    left cells of `W`.  A **size-tier** pre-partition (right-descent sets >300,
//!    generalised-tau >1500) keeps the decomposition tractable; both keys are
//!    constant on left cells, so a bucket never splits a cell.
//! 4. Each new component spawns its full **star orbit** (and the `w0`-image's
//!    orbit) — these are all the remaining left cells with the same W-graph.
//!
//! An involution `celms` skip-set short-circuits W1-reps that can only produce
//! already-seen cells, and a running `tot` early-exits when every element of `W`
//! has been placed.
//!
//! [`relklpols`]: crate::kl::relklpols
//! [`CellGraph::from_relkl`]: crate::cellgraph::CellGraph::from_relkl
//! [`decompose`]: crate::cellgraph::CellGraph::decompose

use std::collections::HashSet;
use std::io;

use crate::{
    cartan::Series,
    cellgraph::CellGraph,
    element::{CoxElm, Gen, Perm, Word},
    group::CoxeterGroup,
    kl::checkpoint::{self, Checkpoint, CheckpointCfg},
    kl::{relklpols, KlError, RelKlOpts},
    parabolic::{red_left_coset_reps, Parabolic},
    star::{generalised_tau, star_orbit_right},
};

// ---------------------------------------------------------------------------
// Size tiers (PyCox 12225–12246)
// ---------------------------------------------------------------------------

/// Induced sets up to this size decompose directly.
const TIER_DIRECT: usize = 300;
/// Above this size the pre-partition key is `generalised_tau` (else the
/// right-descent set).  `maxd = 3 * rank`.
const TIER_TAU: usize = 1500;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Options for [`klcells`].
#[derive(Clone, Debug)]
pub struct CellsOpts {
    /// Whether to return all elements of each cell (`true`, the default at top
    /// level) or only those whose inverse also lies in the cell (`false`, the
    /// recursion's setting).  Mirrors PyCox `allcells`.
    pub all_cells: bool,
    /// Thread count passed straight through to [`relklpols`] for the relative-KL
    /// wavefront (Task P6).  Convention (see [`RelKlOpts`]): `None` ⇒ global
    /// Rayon pool, `Some(0 | 1)` ⇒ sequential, `Some(t > 1)` ⇒ a private pool.
    /// The cell partition and star-class reps are **identical** for any value —
    /// `relklpols` is deterministic across thread counts.
    pub threads: Option<usize>,
}

impl Default for CellsOpts {
    fn default() -> Self {
        CellsOpts {
            all_cells: true,
            threads: None,
        }
    }
}

/// Result of [`klcells`].
#[derive(Clone, Debug)]
pub struct KlCellsResult {
    /// The left-cell partition.  At the top level (`all_cells=true`) each cell
    /// is canonicalized to match the golden `cells_*` format: each cell's words
    /// are canonical reduced words sorted by `(length, lex)`, and the cell list
    /// is sorted lexicographically.  Under `all_cells=false` the cells carry
    /// only inverse-closed elements (un-canonicalized order is irrelevant there
    /// since it feeds the recursion, but we canonicalize uniformly).
    pub cells: Vec<Vec<Word>>,
    /// Number of star-class representatives (`len(cr1)`).
    pub n_star_reps: usize,
    /// The star-class representative W-graphs (`cr1`), sorted by `|X|`.
    pub star_reps: Vec<CellGraph>,
}

// ---------------------------------------------------------------------------
// Streaming API (Task Q1)
// ---------------------------------------------------------------------------

/// One emitted left cell, with provenance, for the streaming driver.
///
/// Records are emitted in **processing order** — W1-rep ascending (`rep_index`),
/// then within a rep the decomposition-component order, then star-orbit order
/// (`orbit_index` counts cells within a single processed rep, across all its
/// components and both the component and its `w0`-image orbits).  This order is
/// deterministic (the P5 `BTreeMap` bucket fix makes the component order
/// reproducible), so re-running or resuming reproduces the same stream.
///
/// `words` is the cell's element list in raw orbit order (NOT canonicalized);
/// the in-memory [`klcells`] path canonicalizes at the very end, so emission
/// order and word reduction here are irrelevant to it.  A consumer that wants the
/// golden order canonicalizes offline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellRecord {
    /// The cell's elements as reduced words (raw orbit order).
    pub words: Vec<Word>,
    /// Index `i` of the producing W1 star-rep in `kk.star_reps`.
    pub rep_index: usize,
    /// Sequential index of this cell within the processing of `rep_index`
    /// (0-based; spans all components and the `w0`-image orbits of that rep).
    pub orbit_index: usize,
}

/// Summary returned by [`klcells_streaming`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KlCellsSummary {
    /// Total number of cell records emitted to the sink (including any re-emitted
    /// after a resume).
    pub ncells: usize,
    /// Number of star-class representatives recorded across the whole run.
    pub n_star_reps: usize,
    /// Total elements of `W` placed (`tot`); equals `|W|` on success when
    /// `all_cells = true`.
    pub total_elements: u128,
    /// `Some(k)` when the run resumed from a checkpoint at rep `k` (so the caller
    /// can truncate/append the stream); `None` for a fresh run.
    pub resumed_at_rep: Option<usize>,
    /// On resume, the number of records already on disk that the caller should
    /// keep (the rest are truncated and re-emitted).  `0` for a fresh run.
    pub records_kept: u128,
}

/// A sink that consumes one star-rep W-graph as it is discovered.
///
/// E8 wants the W-graphs saved — they are the mathematical payload — but they can
/// be large, so the streaming driver hands each one to this callback instead of
/// accumulating them in RAM.  The CLI serializes them to `dir/reps/NNNNNN.json.gz`.
pub type RepsSink<'a> = dyn FnMut(usize, &CellGraph) -> io::Result<()> + 'a;

/// A sink that consumes one [`CellRecord`] as it is emitted.
pub type CellsSink<'a> = dyn FnMut(CellRecord) -> io::Result<()> + 'a;

/// Compute the left-cell partition of `g` by parabolic induction.
///
/// See the [module docs](self) for the algorithm.  Equal parameters only
/// (implicit `L(s) = 1`); the unequal-parameter branch (`klcellsun`) is out of
/// scope for this task.
///
/// # Errors
///
/// Returns [`KlError::Internal`] if the rep-loop exhausts the `W1`
/// representatives before covering all of `W` (a logic bug that PyCox would
/// instead hang on), or if the final `Σ|cell| == |W|` invariant fails.
pub fn klcells(g: &CoxeterGroup, opts: &CellsOpts) -> Result<KlCellsResult, KlError> {
    klcells_with_tiers(g, opts, TIER_DIRECT, TIER_TAU)
}

/// [`klcells`] with explicit size-tier thresholds.
///
/// `tier_direct`: induced sets of this size or smaller decompose directly.
/// `tier_tau`: above this size the pre-partition uses `generalised_tau`; between
/// `tier_direct` and `tier_tau` it uses the right-descent set.
///
/// This is a test hook: the production [`klcells`] calls it with
/// `(TIER_DIRECT, TIER_TAU)`.  Passing tiny thresholds forces the pre-partition
/// branches on small groups; the output MUST be identical to the default-tier
/// run (the pre-partition is sound because both keys are constant on left
/// cells).
pub fn klcells_with_tiers(
    g: &CoxeterGroup,
    opts: &CellsOpts,
    tier_direct: usize,
    tier_tau: usize,
) -> Result<KlCellsResult, KlError> {
    let raw = klcells_raw(g, opts.all_cells, opts.threads, tier_direct, tier_tau)?;

    // Top-level / uniform canonicalization to the golden format.
    let cells = canonicalize_cells(g, &raw.cells);

    Ok(KlCellsResult {
        cells,
        n_star_reps: raw.star_reps.len(),
        star_reps: raw.star_reps,
    })
}

// ---------------------------------------------------------------------------
// Streaming driver (Task Q1)
// ---------------------------------------------------------------------------

/// A `(group, opts)` fingerprint binding a checkpoint to its run.
///
/// Two runs share a checkpoint iff this string matches: it encodes the group
/// type (series + rank per component), the group order, and the `all_cells`
/// flag.  Thread count is deliberately excluded — it never affects the
/// partition, so a checkpoint written by a 64-thread run resumes correctly under
/// any thread count.
pub fn run_fingerprint(g: &CoxeterGroup, all_cells: bool) -> String {
    let mut s = String::from("rustcox-klcells-v1|");
    for (k, c) in g.components.iter().enumerate() {
        if k > 0 {
            s.push('x');
        }
        s.push_str(&format!("{}{}", c.series, c.indices.len()));
    }
    s.push_str(&format!("|order={}|allcells={}", g.order, all_cells as u8));
    s
}

/// Streaming, checkpointable `klcells`: the same partition as [`klcells`] but
/// every cell is handed to `sink` as it is found (never accumulated in RAM), and
/// the persistent loop state is checkpointed so a SLURM kill is recoverable.
///
/// - `sink` receives one [`CellRecord`] per left cell, in deterministic
///   processing order (see [`CellRecord`]).  Cell words are NOT canonicalized;
///   a consumer that wants the golden order canonicalizes offline.
/// - `reps_sink`, when `Some`, receives every star-rep W-graph as it is
///   discovered (E8 wants these saved — they are the mathematical payload).
/// - `ckpt`, when `Some`, enables checkpoint/resume.  On entry the driver looks
///   for a valid matching checkpoint in `cfg.dir`; if found it recomputes the
///   cheap `kk` recursion, restores `celms`/`tot`/registry, and fast-forwards to
///   `next_rep`, re-emitting only cells from that rep onward.  The returned
///   [`KlCellsSummary`] reports the resume point and the record count to keep so
///   the caller can truncate its stream file.
///
/// SIGTERM/SIGINT are intentionally **not** trapped (no `unsafe`, no extra
/// deps): with `every_reps = 1` the process can be killed at any moment and lose
/// at most one rep of work, so an external kill is always safe.
///
/// # Errors
///
/// Propagates [`KlError`] from the induction (same invariants as [`klcells`]),
/// and wraps any I/O error from `sink`/`reps_sink`/checkpoint writes as
/// [`KlError::Internal`].
pub fn klcells_streaming<'a, 'f: 'a>(
    g: &CoxeterGroup,
    opts: &CellsOpts,
    sink: &'a mut CellsSink<'f>,
    reps_sink: Option<&'a mut RepsSink<'f>>,
    ckpt: Option<&CheckpointCfg>,
) -> Result<KlCellsSummary, KlError> {
    klcells_streaming_with_tiers(g, opts, sink, reps_sink, ckpt, TIER_DIRECT, TIER_TAU)
}

/// [`klcells_streaming`] with explicit size-tier thresholds (test hook; see
/// [`klcells_with_tiers`]).
#[allow(clippy::too_many_arguments)]
pub fn klcells_streaming_with_tiers<'a, 'f: 'a>(
    g: &CoxeterGroup,
    opts: &CellsOpts,
    sink: &'a mut CellsSink<'f>,
    reps_sink: Option<&'a mut RepsSink<'f>>,
    ckpt: Option<&CheckpointCfg>,
    tier_direct: usize,
    tier_tau: usize,
) -> Result<KlCellsSummary, KlError> {
    let all_cells = opts.all_cells;
    let threads = opts.threads;
    let fingerprint = run_fingerprint(g, all_cells);

    // Rank-0 streaming base case: a single trivial cell, no checkpointing needed.
    if g.rank == 0 {
        sink(CellRecord {
            words: vec![Vec::new()],
            rep_index: 0,
            orbit_index: 0,
        })
        .map_err(io_err)?;
        return Ok(KlCellsSummary {
            ncells: 1,
            n_star_reps: 1,
            total_elements: 1,
            resumed_at_rep: None,
            records_kept: 0,
        });
    }

    // Resume: load a valid matching checkpoint, if any.
    let mut restored: Option<Checkpoint> = None;
    if let Some(cfg) = ckpt {
        if checkpoint::exists(cfg) {
            match checkpoint::load_matching(cfg, &fingerprint) {
                Ok(ck) => restored = Some(ck),
                Err(e) => {
                    // Corrupt or mismatched: start fresh (caller / CLI warns).
                    // We surface the reason via a non-fatal Internal note path:
                    // the streaming summary's resume fields stay None.
                    let _ = e;
                }
            }
        }
    }

    // The cheap kk recursion is recomputed deterministically on every run.
    let j = choose_j(g);
    let w1 = Parabolic::new(g, &j).map_err(|e| KlError::Internal(format!("parabolic: {e}")))?;
    let x1p: Vec<Word> = red_left_coset_reps(g, &j);
    let kk = klcells_raw(&w1.group, false, threads, tier_direct, tier_tau)?;

    // Resume info for the summary (only when a checkpoint was actually loaded).
    let (resumed_at_rep, records_kept, reps_seen) = match &restored {
        Some(ck) => (Some(checkpoint_next_rep(ck)), ck.records, ck.registry.len()),
        None => (None, 0, 0),
    };
    let resume = restored.map(Resume::from_checkpoint);

    // Seed the reps-sink serial index past the reps recorded before this
    // invocation, so resumed runs do not overwrite earlier `reps/NNN` files.
    let mut emit = StreamEmitter {
        sink,
        reps_sink,
        next_rep_idx: reps_seen,
    };

    let ckpt_arg = ckpt.map(|cfg| (cfg, fingerprint.as_str()));

    let state = run_induction(
        g,
        &w1,
        &x1p,
        &kk.star_reps,
        all_cells,
        threads,
        tier_direct,
        tier_tau,
        &mut emit,
        resume,
        ckpt_arg,
    )?;

    if all_cells && state.tot != g.order {
        return Err(KlError::Internal(format!(
            "klcells_streaming: tot = {} != |W| = {} (rank {})",
            state.tot, g.order, g.rank
        )));
    }

    Ok(KlCellsSummary {
        // `records`/`registry` accumulate across resumes (restored + new), so
        // they reflect the whole run, not just this invocation.
        ncells: state.records as usize,
        n_star_reps: state.registry.len(),
        total_elements: state.tot,
        resumed_at_rep,
        records_kept,
    })
}

/// Map a sink I/O error into [`KlError`].
fn io_err(e: io::Error) -> KlError {
    KlError::Internal(format!("streaming sink error: {e}"))
}

/// Streaming emitter: forwards cells to `sink`, star-reps to `reps_sink`.
///
/// Two lifetimes: `'a` borrows the sinks for this call; `'f` (with `'f: 'a`) is
/// the trait objects' own lifetime.  Keeping them distinct (rather than the
/// invariant `&'a mut X<'a>`) is what lets call sites pass closures that borrow
/// locals.
struct StreamEmitter<'a, 'f: 'a> {
    sink: &'a mut CellsSink<'f>,
    reps_sink: Option<&'a mut RepsSink<'f>>,
    /// Count of star-reps emitted so far (their serial index for `reps_sink`).
    next_rep_idx: usize,
}

impl Emitter for StreamEmitter<'_, '_> {
    fn cell(&mut self, words: Vec<Word>, rep: usize, orbit: usize) -> Result<(), KlError> {
        (self.sink)(CellRecord {
            words,
            rep_index: rep,
            orbit_index: orbit,
        })
        .map_err(io_err)
    }
    fn star_rep(&mut self, rep: &CellGraph) -> Result<(), KlError> {
        let idx = self.next_rep_idx;
        if let Some(rs) = self.reps_sink.as_mut() {
            (rs)(idx, rep).map_err(io_err)?;
        }
        self.next_rep_idx += 1;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Raw recursive driver (un-canonicalized cells, used by the recursion)
// ---------------------------------------------------------------------------

/// Internal result of the recursion: cells as word-lists in orbit order
/// (un-canonicalized), plus the star-rep graphs.
struct RawCells {
    cells: Vec<Vec<Word>>,
    star_reps: Vec<CellGraph>,
}

/// The PyCox `klcells(W, 1, v, allcells)` recursion, equal parameters.
///
/// `threads` is forwarded verbatim to [`relklpols`]; it affects only wall time,
/// not the partition.  The recursion on `W1` runs sequentially per call but its
/// own `relklpols` invocations inherit the same `threads`.
fn klcells_raw(
    g: &CoxeterGroup,
    all_cells: bool,
    threads: Option<usize>,
    tier_direct: usize,
    tier_tau: usize,
) -> Result<RawCells, KlError> {
    // --- Rank-0 base case (PyCox 12193–12196) -------------------------------
    if g.rank == 0 {
        // One trivial cell [[]]; one 1-vertex graph (empty word, empty Iset,
        // seeded-empty pools, empty mmat).  PyCox:
        //   nc  = [[[]]]
        //   cr1 = [wgraph(W, poids, [[]], v, [[]], {}, [], [()])]
        let trivial = CellGraph {
            x: vec![Vec::new()],
            xrep: vec![CoxElm(Vec::new().into_boxed_slice())],
            isets: vec![Vec::new()],
            mpols: Vec::new(),
            mmat: std::collections::HashMap::new(),
            weights: Vec::new(),
        };
        return Ok(RawCells {
            cells: vec![vec![Vec::new()]],
            star_reps: vec![trivial],
        });
    }

    // --- Choose J (the E7 J-rule, PyCox 12199–12203) ------------------------
    let j = choose_j(g);
    let w1 = Parabolic::new(g, &j).map_err(|e| KlError::Internal(format!("parabolic: {e}")))?;
    let x1p: Vec<Word> = red_left_coset_reps(g, &j);

    // --- Recurse on W1 (allcells = false) -----------------------------------
    let kk = klcells_raw(&w1.group, false, threads, tier_direct, tier_tau)?;

    let order = g.order;

    // --- Main induction loop, in-memory collecting emitter ------------------
    let mut emit = CollectEmitter::default();
    run_induction(
        g,
        &w1,
        &x1p,
        &kk.star_reps,
        all_cells,
        threads,
        tier_direct,
        tier_tau,
        &mut emit,
        None, // fresh run (no resume)
        None, // no checkpointing in the recursive / in-memory path
    )?;

    let nc = emit.nc;
    let cr1 = emit.cr1;

    // --- Final correctness check (PyCox replaces chartable; notes §) --------
    let sum: usize = nc.iter().map(|c| c.len()).sum();
    if all_cells && sum as u128 != order {
        return Err(KlError::Internal(format!(
            "klcells: Σ|cell| = {sum} != |W| = {order} (rank {})",
            g.rank
        )));
    }

    // Optional full-distinctness check (gated on size; see notes).  For
    // all_cells=true the union must be exactly W.  Full check is affordable up
    // to ~2M elements; above that we rely on the Σ == |W| + per-orbit
    // involution invariants (E7-scale memory note).
    //
    // This is a debug-only sanity check: it is wrapped in `cfg(debug_assertions)`
    // so release builds never pay for the (up to ~2M-entry) `HashSet`.  Release
    // correctness rests on the always-on `Σ == |W|` gate above plus the golden /
    // full-table cross-checks in the test suite.
    #[cfg(debug_assertions)]
    if all_cells && order <= 2_000_000 {
        let mut seen: HashSet<CoxElm> = HashSet::with_capacity(sum);
        for cell in &nc {
            for w in cell {
                let ce = g.word_to_coxelm(w);
                assert!(
                    seen.insert(ce),
                    "klcells: duplicate element {w:?} across cells (rank {})",
                    g.rank
                );
            }
        }
    }

    // cr1 sorted by |X| (PyCox 12300).  Stable sort keeps discovery order
    // among equal sizes (matches PyCox's list.sort stability).
    let mut star_reps = cr1;
    star_reps.sort_by_key(|c| c.x.len());

    Ok(RawCells {
        cells: nc,
        star_reps,
    })
}

// ---------------------------------------------------------------------------
// Shared induction core (ONE driver, used by both in-memory and streaming)
// ---------------------------------------------------------------------------

/// Sink for cells discovered by [`run_induction`].
///
/// Implementors decide what to do with each emitted cell and each recorded
/// star-rep.  The in-memory path collects into vectors; the streaming path
/// forwards to user callbacks and writes checkpoints.  All bookkeeping that
/// affects the *math* (`celms`, `tot`) lives in [`run_induction`]; the emitter
/// only consumes results.
trait Emitter {
    /// Record one cell's words (raw orbit order) with provenance.
    fn cell(
        &mut self,
        words: Vec<Word>,
        rep_index: usize,
        orbit_index: usize,
    ) -> Result<(), KlError>;

    /// Record one star-rep W-graph as it is discovered.
    fn star_rep(&mut self, rep: &CellGraph) -> Result<(), KlError>;
}

/// In-memory emitter: collects cells into `nc` and reps into `cr1`, exactly as
/// the original driver did.  Provenance is ignored (the in-memory path
/// canonicalizes `nc` at the end, so emission order is irrelevant).
#[derive(Default)]
struct CollectEmitter {
    nc: Vec<Vec<Word>>,
    cr1: Vec<CellGraph>,
}

impl Emitter for CollectEmitter {
    fn cell(&mut self, words: Vec<Word>, _rep: usize, _orbit: usize) -> Result<(), KlError> {
        self.nc.push(words);
        Ok(())
    }
    fn star_rep(&mut self, rep: &CellGraph) -> Result<(), KlError> {
        self.cr1.push(rep.clone());
        Ok(())
    }
}

/// State threaded through the main induction loop and persisted by checkpoints.
struct LoopState {
    /// Involution coxelms (the skip-set).
    celms: HashSet<CoxElm>,
    /// Star-rep registry fingerprints (`xrep[0]` of each recorded rep), in
    /// discovery order — checkpointed so resume restores the registry size.
    registry: Vec<CoxElm>,
    /// Elements of `W` placed so far.
    tot: u128,
    /// Cell records emitted so far (across all reps).
    records: u128,
}

impl LoopState {
    /// A pristine state for a from-scratch run.
    fn fresh() -> Self {
        LoopState {
            celms: HashSet::new(),
            registry: Vec::new(),
            tot: 0,
            records: 0,
        }
    }
}

/// A resume point reconstructed from a checkpoint: the restored loop state plus
/// the rep index to fast-forward to.
struct Resume {
    state: LoopState,
    next_rep: usize,
}

impl Resume {
    /// Build a [`Resume`] from a loaded [`Checkpoint`].
    fn from_checkpoint(ck: Checkpoint) -> Self {
        let next_rep = checkpoint_next_rep(&ck);
        Resume {
            state: LoopState {
                celms: ck.celms.into_iter().collect(),
                registry: ck.registry,
                tot: ck.tot,
                records: ck.records,
            },
            next_rep,
        }
    }
}

/// The PyCox `klcells` main induction loop (lines 12210–12288), shared by the
/// in-memory and streaming drivers.
///
/// Drives the W1 star-reps `reps[i]`, skips already-covered ones, induces +
/// decomposes the rest, and feeds every discovered cell / star-rep to `emit`.
/// When `ckpt` is `Some`, a checkpoint is written after each processed rep (per
/// `cfg.every_reps`) and after the very last rep, so a kill at *any* point loses
/// at most `every_reps` reps of work.
///
/// Returns the final [`LoopState`] (caller needs `tot`/`records` for the
/// summary).
#[allow(clippy::too_many_arguments)]
fn run_induction(
    g: &CoxeterGroup,
    w1: &Parabolic,
    x1p: &[Word],
    reps: &[CellGraph],
    all_cells: bool,
    threads: Option<usize>,
    tier_direct: usize,
    tier_tau: usize,
    emit: &mut dyn Emitter,
    resume: Option<Resume>,
    ckpt: Option<(&CheckpointCfg, &str)>,
) -> Result<LoopState, KlError> {
    let order = g.order;
    let sr = &g.simple_root;
    let id_ce = g.id_perm().coxelm_sr(sr);

    let (cfg, fingerprint) = match ckpt {
        Some((cfg, fp)) => (Some(cfg), Some(fp)),
        None => (None, None),
    };

    // Initialize loop state, restoring from a resume checkpoint if provided.
    let (mut state, mut i) = match resume {
        Some(r) => (r.state, r.next_rep),
        None => (LoopState::fresh(), 0),
    };

    while state.tot < order {
        if i >= reps.len() {
            // PyCox would loop forever here; we fail loudly with diagnostics.
            return Err(KlError::Internal(format!(
                "klcells: exhausted {} W1-reps with tot={} < |W|={order} \
                 (group rank {}); the induction failed to cover W",
                reps.len(),
                state.tot,
                g.rank
            )));
        }

        // Checkpoint at the TOP of the iteration: `state` here is exactly the
        // boundary state after rep `i-1` (all of reps `< i` done, with this many
        // records on disk).  Resume re-runs rep `i` from scratch (relklpols is
        // not interruptible), so a kill anywhere inside rep `i` is recovered by
        // truncating the stream to `state.records` and replaying from rep `i`.
        maybe_checkpoint(cfg, fingerprint, &state, i, false)?;

        let rep = &reps[i];

        // pairs = [W.wordtoperm(x1 ++ [J[s] for s in w]) for x1 in X1p,
        //                                                 for w in rep.X]
        // (cartesian order is irrelevant: this is a skip-test scan.)
        let mut pairs: Vec<Perm> = Vec::with_capacity(x1p.len() * rep.x.len());
        for x1 in x1p {
            for w in &rep.x {
                let mapped = w1.word_to_w(w); // [J[s] for s in w]
                let mut word: Word = x1.clone();
                word.extend_from_slice(&mapped);
                pairs.push(g.word_to_perm(&word));
            }
        }

        // skip-test (PyCox 12217–12219): all pairs are non-involutions OR
        // already-seen involutions.
        let skip = pairs.iter().all(|pa| {
            let pa2 = pa.then(pa);
            let is_invol = pa2.coxelm_sr(sr) == id_ce;
            !is_invol || state.celms.contains(&pa.coxelm_sr(sr))
        });
        if skip {
            i += 1;
            continue;
        }

        // rk = relklpols(W, W1, rep.to_relkl(W1.group), 1, v)
        let cell1 = rep.to_relkl(&w1.group);
        let rk = relklpols(g, w1, &cell1, &RelKlOpts { threads });

        // Build the induced W-graph and decompose (with size tiers).
        let weights = vec![1u32; g.rank];
        let cg = CellGraph::from_relkl(g, &weights, &rk.input);
        let ind = decompose_tiered(g, &cg, &rk.perms, tier_direct, tier_tau);

        // `orbit_index` counts emitted cells within THIS rep (all components +
        // both the component and its w0-image orbits).
        let mut orbit_index = 0usize;

        // For each component: emit its star orbit (+ the w0-image's orbit).
        for ii in &ind {
            // First: the component itself.
            if state.tot < order && !ii.xrep.iter().any(|x| state.celms.contains(x)) {
                emit.star_rep(ii)?;
                register_rep(&mut state, ii);
                expand_orbit(
                    g,
                    ii,
                    all_cells,
                    emit,
                    &mut state,
                    &id_ce,
                    i,
                    &mut orbit_index,
                )?;
            }
            // Then: the w0-image (PyCox 12268–12287).
            if state.tot < order {
                let ii0 = ii.cell_w0(g);
                if !ii0.xrep.iter().any(|x| state.celms.contains(x)) {
                    emit.star_rep(&ii0)?;
                    register_rep(&mut state, &ii0);
                    expand_orbit(
                        g,
                        &ii0,
                        all_cells,
                        emit,
                        &mut state,
                        &id_ce,
                        i,
                        &mut orbit_index,
                    )?;
                }
            }
        }
        i += 1;
    }

    // Final checkpoint: records the completed state (next_rep == reps.len()), so
    // a resubmit after a clean finish does nothing.  Forced regardless of the
    // `every_reps` cadence so the terminal state is always persisted.
    maybe_checkpoint(cfg, fingerprint, &state, i.max(reps.len()), true)?;

    Ok(state)
}

/// The rep index a checkpoint resumes at: `next_rep`, narrowed to `usize`.
fn checkpoint_next_rep(ck: &Checkpoint) -> usize {
    ck.next_rep.min(usize::MAX as u128) as usize
}

/// Record a star-rep's fingerprint (`xrep[0]`) into the registry, if any.
fn register_rep(state: &mut LoopState, rep: &CellGraph) {
    if let Some(first) = rep.xrep.first() {
        state.registry.push(first.clone());
    }
}

/// Write a checkpoint if `cfg` is set and either `force` or the rep count is a
/// multiple of `cfg.every_reps`.
fn maybe_checkpoint(
    cfg: Option<&CheckpointCfg>,
    fingerprint: Option<&str>,
    state: &LoopState,
    next_rep: usize,
    force: bool,
) -> Result<(), KlError> {
    let (Some(cfg), Some(fp)) = (cfg, fingerprint) else {
        return Ok(());
    };
    let every = cfg.every_reps.max(1);
    if !force && next_rep % every != 0 {
        return Ok(());
    }
    let ck = Checkpoint {
        fingerprint: fp.to_string(),
        next_rep: next_rep as u128,
        tot: state.tot,
        records: state.records,
        rank: 0, // filled below from celms; rank carried for decode only
        celms: state.celms.iter().cloned().collect(),
        registry: state.registry.clone(),
    };
    // The coxelm word length is the rank; derive it from any element (celms or
    // registry).  Empty state (rank-0 / no elements yet) uses 0.
    let rank = ck
        .celms
        .first()
        .or_else(|| ck.registry.first())
        .map(|c| c.0.len())
        .unwrap_or(0);
    let ck = Checkpoint { rank, ..ck };
    ck.write_atomic(cfg)
        .map_err(|e| KlError::Internal(format!("checkpoint write failed: {e}")))
}

// ---------------------------------------------------------------------------
// J selection (the E7 J-rule)
// ---------------------------------------------------------------------------

/// Choose `J = rank \ {one generator}` per PyCox 12199–12203.
///
/// If the (first) component is series `E` with rank 7, remove generator `0`
/// (yielding a `D6` parabolic in this numbering).  Otherwise remove the LAST
/// generator.
fn choose_j(g: &CoxeterGroup) -> Vec<Gen> {
    let drop: usize = if g.components[0].series == Series::E && g.components[0].indices.len() == 7 {
        0
    } else {
        g.rank - 1
    };
    (0..g.rank as Gen).filter(|&s| s as usize != drop).collect()
}

// ---------------------------------------------------------------------------
// Tiered decomposition (PyCox 12225–12246)
// ---------------------------------------------------------------------------

/// Decompose the induced W-graph, optionally pre-partitioning by a left-cell
/// invariant for large vertex sets.
///
/// - `|elements| ≤ tier_direct`: decompose directly.
/// - `tier_direct < |elements| ≤ tier_tau`: pre-partition by right-descent set.
/// - `|elements| > tier_tau`: pre-partition by `generalised_tau(p, 3·rank)`.
///
/// `perms` are the vertex perms parallel to `cg.x` (= `rk.perms`).  Because both
/// keys are constant on a left cell, a bucket can only contain whole cells, so
/// concatenating the per-bucket decompositions is exactly the full
/// decomposition.
fn decompose_tiered(
    g: &CoxeterGroup,
    cg: &CellGraph,
    perms: &[Perm],
    tier_direct: usize,
    tier_tau: usize,
) -> Vec<CellGraph> {
    let n = cg.x.len();
    if n <= tier_direct {
        return cg.decompose(g);
    }

    // Compute the bucket key for every vertex.
    let keys: Vec<Vec<Gen>> = if n > tier_tau {
        let maxd = 3 * g.rank;
        perms
            .iter()
            .map(|p| flatten_tau(&generalised_tau(g, p, maxd)))
            .collect()
    } else {
        perms.iter().map(|p| g.right_descents(p)).collect()
    };

    // Group vertex positions by key, then restrict + decompose + concat.
    // BTreeMap (not HashMap): bucket iteration order is deterministic, so the
    // concatenated `ind` order — and hence which star-orbit member is recorded
    // into `cr1`/`star_reps` — is reproducible across runs.
    let mut buckets: std::collections::BTreeMap<Vec<Gen>, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (pos, k) in keys.iter().enumerate() {
        buckets.entry(k.clone()).or_default().push(pos);
    }

    let mut out: Vec<CellGraph> = Vec::new();
    for positions in buckets.values() {
        let sub = cg.restrict(positions);
        out.extend(sub.decompose(g));
    }
    out
}

/// Flatten a `generalised_tau` result (a list of right-descent sets) into a
/// single hashable/equatable key.  The orbit's descent-set list is itself the
/// left-cell invariant; a flat encoding with a separator preserves it.
fn flatten_tau(tau: &[Vec<Gen>]) -> Vec<Gen> {
    let mut out: Vec<Gen> = Vec::new();
    for ds in tau {
        out.push(Gen::MAX); // separator between descent sets
        out.extend_from_slice(ds);
    }
    out
}

// ---------------------------------------------------------------------------
// Star-orbit expansion (PyCox 12252–12266)
// ---------------------------------------------------------------------------

/// Expand one cell's star orbit, emitting each orbit member's cell words to
/// `emit`, registering involution coxelms into `state.celms`, and advancing
/// `state.tot`/`state.records`.
///
/// `all_cells` controls the inverse-closure filter:
/// - `true`: every orbit element's word is emitted.
/// - `false`: only elements whose inverse is also in the orbit.
///
/// `rep_index`/`orbit_index` carry provenance to the emitter; `orbit_index` is
/// advanced once per emitted cell so it is unique within the current rep.
#[allow(clippy::too_many_arguments)]
fn expand_orbit(
    g: &CoxeterGroup,
    cell: &CellGraph,
    all_cells: bool,
    emit: &mut dyn Emitter,
    state: &mut LoopState,
    id_ce: &CoxElm,
    rep_index: usize,
    orbit_index: &mut usize,
) -> Result<(), KlError> {
    let sr = &g.simple_root;
    let cell_perms: Vec<Perm> = cell.x.iter().map(|w| g.word_to_perm(w)).collect();
    let orbit = star_orbit_right(g, &cell_perms);

    for o in &orbit {
        // Cell words.
        let words: Vec<Word> = if all_cells {
            o.iter().map(|p| g.perm_to_word(p)).collect()
        } else {
            // Only elements whose inverse is also in this orbit member `o`.
            let o_ces: HashSet<CoxElm> = o.iter().map(|p| p.coxelm_sr(sr)).collect();
            o.iter()
                .filter(|p| o_ces.contains(&p.inverse().coxelm_sr(sr)))
                .map(|p| g.perm_to_word(p))
                .collect()
        };
        emit.cell(words, rep_index, *orbit_index)?;
        *orbit_index += 1;
        state.records += 1;

        // Register involution coxelms.
        for e in o {
            if e.then(e).coxelm_sr(sr) == *id_ce {
                state.celms.insert(e.coxelm_sr(sr));
            }
        }
        state.tot += o.len() as u128;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Final cell canonicalization (golden format)
// ---------------------------------------------------------------------------

/// Canonicalize cells to the golden `cells_*` format: each word re-reduced to
/// its canonical reduced word, each cell sorted by `(length, lex)`, cell list
/// sorted lexicographically.
fn canonicalize_cells(g: &CoxeterGroup, cells: &[Vec<Word>]) -> Vec<Vec<Word>> {
    let mut out: Vec<Vec<Word>> = cells
        .iter()
        .map(|c| {
            let mut can: Vec<Word> = c
                .iter()
                .map(|w| g.perm_to_word(&g.word_to_perm(w)))
                .collect();
            can.sort_by(|a, b| (a.len(), a).cmp(&(b.len(), b)));
            can
        })
        .collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank0_base_case() {
        // A0 is not constructible via from_type; build a rank-0 group by hand is
        // not exposed.  Instead exercise choose_j / canonicalize on a tiny group
        // and the rank-0 path indirectly through A1's recursion (A1 → rank 0).
        let g = CoxeterGroup::from_type("A1").unwrap();
        let res = klcells(&g, &CellsOpts::default()).unwrap();
        // A1 has 2 left cells: {[]} and {[0]}.
        assert_eq!(res.cells.len(), 2);
        let tot: usize = res.cells.iter().map(|c| c.len()).sum();
        assert_eq!(tot, 2);
    }

    #[test]
    fn choose_j_drops_last_generator_generic() {
        let g = CoxeterGroup::from_type("B4").unwrap();
        let j = choose_j(&g);
        assert_eq!(j, vec![0, 1, 2]); // dropped generator 3 (the last).
    }

    #[test]
    fn choose_j_e7_drops_generator_zero() {
        // Build E7; the rule must drop generator 0 (→ D6 parabolic).
        let g = CoxeterGroup::from_type("E7").unwrap();
        let j = choose_j(&g);
        assert_eq!(j, (1..7).collect::<Vec<Gen>>());
        // And the resulting parabolic is D6 (verifies the "not E6" note).
        let w1 = Parabolic::new(&g, &j).unwrap();
        assert_eq!(w1.group.rank, 6);
        // D6 order = 2^5 * 6! = 23040.
        assert_eq!(w1.group.order, 23040);
    }

    #[test]
    fn a2_partition() {
        let g = CoxeterGroup::from_type("A2").unwrap();
        let res = klcells(&g, &CellsOpts::default()).unwrap();
        let tot: usize = res.cells.iter().map(|c| c.len()).sum();
        assert_eq!(tot as u128, g.order);
        // A2 has 4 left cells, 3 star-class reps (verified against PyCox).
        assert_eq!(res.cells.len(), 4);
        assert_eq!(res.n_star_reps, 3);
        // Exact partition (golden-canonical order).
        let want: Vec<Vec<Word>> = vec![
            vec![vec![]],
            vec![vec![0], vec![1, 0]],
            vec![vec![0, 1, 0]],
            vec![vec![1], vec![0, 1]],
        ];
        assert_eq!(res.cells, want);
    }
}
