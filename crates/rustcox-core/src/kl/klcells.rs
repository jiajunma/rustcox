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

use crate::{
    cartan::Series,
    cellgraph::CellGraph,
    element::{CoxElm, Gen, Perm, Word},
    group::CoxeterGroup,
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
    // `threads` and other parallelism knobs are reserved for Task P6.
}

impl Default for CellsOpts {
    fn default() -> Self {
        CellsOpts { all_cells: true }
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
    let raw = klcells_raw(g, opts.all_cells, tier_direct, tier_tau)?;

    // Top-level / uniform canonicalization to the golden format.
    let cells = canonicalize_cells(g, &raw.cells);

    Ok(KlCellsResult {
        cells,
        n_star_reps: raw.star_reps.len(),
        star_reps: raw.star_reps,
    })
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
fn klcells_raw(
    g: &CoxeterGroup,
    all_cells: bool,
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
    let kk = klcells_raw(&w1.group, false, tier_direct, tier_tau)?;

    // --- Main induction loop (PyCox 12210–12288) ----------------------------
    let order = g.order;
    let sr = &g.simple_root;
    let id_ce = g.id_perm().coxelm_sr(sr);

    // celms: coxelms of INVOLUTIONS only (the skip-test set).
    let mut celms: HashSet<CoxElm> = HashSet::new();
    let mut nc: Vec<Vec<Word>> = Vec::new();
    let mut cr1: Vec<CellGraph> = Vec::new();
    let mut tot: u128 = 0;

    let mut i = 0usize;
    while tot < order {
        if i >= kk.star_reps.len() {
            // PyCox would loop forever here; we fail loudly with diagnostics.
            return Err(KlError::Internal(format!(
                "klcells: exhausted {} W1-reps with tot={tot} < |W|={order} \
                 (group rank {}); the induction failed to cover W",
                kk.star_reps.len(),
                g.rank
            )));
        }
        let rep = &kk.star_reps[i];

        // pairs = [W.wordtoperm(x1 ++ [J[s] for s in w]) for x1 in X1p,
        //                                                 for w in rep.X]
        // (cartesian order is irrelevant: this is a skip-test scan.)
        let mut pairs: Vec<Perm> = Vec::with_capacity(x1p.len() * rep.x.len());
        for x1 in &x1p {
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
            !is_invol || celms.contains(&pa.coxelm_sr(sr))
        });
        if skip {
            i += 1;
            continue;
        }

        // rk = relklpols(W, W1, rep.to_relkl(W1.group), 1, v)
        let cell1 = rep.to_relkl(&w1.group);
        let rk = relklpols(g, &w1, &cell1, &RelKlOpts::default());

        // Build the induced W-graph and decompose (with size tiers).
        let weights = vec![1u32; g.rank];
        let cg = CellGraph::from_relkl(g, &weights, &rk.input);
        let ind = decompose_tiered(g, &cg, &rk.perms, tier_direct, tier_tau);

        // For each component: emit its star orbit (+ the w0-image's orbit).
        for ii in &ind {
            // First: the component itself.
            if tot < order && !ii.xrep.iter().any(|x| celms.contains(x)) {
                cr1.push(ii.clone());
                expand_orbit(
                    g,
                    ii,
                    all_cells,
                    &mut nc,
                    &mut celms,
                    &mut tot,
                    id_ce.clone(),
                );
            }
            // Then: the w0-image (PyCox 12268–12287).
            if tot < order {
                let ii0 = ii.cell_w0(g);
                if !ii0.xrep.iter().any(|x| celms.contains(x)) {
                    cr1.push(ii0.clone());
                    expand_orbit(
                        g,
                        &ii0,
                        all_cells,
                        &mut nc,
                        &mut celms,
                        &mut tot,
                        id_ce.clone(),
                    );
                }
            }
        }
        i += 1;
    }

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

/// Expand one cell's star orbit, appending each orbit member's cell words to
/// `nc`, registering involution coxelms into `celms`, and advancing `tot`.
///
/// `all_cells` controls the inverse-closure filter:
/// - `true`: every orbit element's word is emitted.
/// - `false`: only elements whose inverse is also in the orbit.
#[allow(clippy::too_many_arguments)]
fn expand_orbit(
    g: &CoxeterGroup,
    cell: &CellGraph,
    all_cells: bool,
    nc: &mut Vec<Vec<Word>>,
    celms: &mut HashSet<CoxElm>,
    tot: &mut u128,
    id_ce: CoxElm,
) {
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
        nc.push(words);

        // Register involution coxelms.
        for e in o {
            if e.then(e).coxelm_sr(sr) == id_ce {
                celms.insert(e.coxelm_sr(sr));
            }
        }
        *tot += o.len() as u128;
    }
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
