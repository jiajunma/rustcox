//! Golden + cross-check tests for the `klcells` driver (Task P5).
//!
//! Four families:
//!   1. `cells_golden_match` — byte-for-byte match against the PyCox golden
//!      `cells_*` files (full nested cell equality, `ncells`, `nstarreps`).
//!   2. `cells_match_full_table` — the klcells partition equals the Phase-1
//!      full-table `CellData::lcells` partition (as canonical word-sets).
//!   3. `recursion_depth` + synthetic tier test — exercise the parabolic
//!      recursion and pin the size-tier pre-partition soundness.
//!   4. `allcells_false_inverse_closure` — with `all_cells=false`, each output
//!      cell is inverse-closed and a subset of the corresponding full cell.

use std::collections::BTreeSet;

use rustcox_core::{
    element::Word,
    group::CoxeterGroup,
    kl::{klcells, klcells_with_tiers, klpolynomials_seq, CellData, CellsOpts, KlOpts, KlTable},
};

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// All `cells_*` golden groups covered by the local suite.
const GROUPS: &[&str] = &["A3", "B3", "A4", "D4", "B4", "H3", "F4"];

fn group_from_golden(name: &str) -> CoxeterGroup {
    let g = common::golden(&format!("cells_{name}"));
    let comps = common::components_of(&g);
    CoxeterGroup::from_components(&comps).unwrap()
}

/// Parse the golden `"cells"` field into `Vec<Vec<Word>>`.
fn golden_cells(g: &serde_json::Value) -> Vec<Vec<Word>> {
    g["cells"]
        .as_array()
        .unwrap()
        .iter()
        .map(|cell| {
            cell.as_array()
                .unwrap()
                .iter()
                .map(|w| {
                    w.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap() as u8)
                        .collect::<Word>()
                })
                .collect::<Vec<Word>>()
        })
        .collect()
}

/// Set of canonical-word-sets (each cell sorted) for partition equality.
fn cell_word_sets(cells: &[Vec<Word>]) -> BTreeSet<Vec<Word>> {
    cells
        .iter()
        .map(|c| {
            let mut v = c.clone();
            v.sort();
            v
        })
        .collect()
}

fn build_table(g: &CoxeterGroup) -> KlTable {
    klpolynomials_seq(g, &KlOpts::equal(g.rank)).unwrap()
}

/// Full-table left cells as canonical-word-sets.
fn full_table_word_sets(t: &KlTable) -> BTreeSet<Vec<Word>> {
    CellData::from_table(t)
        .lcells
        .iter()
        .map(|cell| {
            let mut ws: Vec<Word> = cell
                .iter()
                .map(|&e| t.elms.elms[e as usize].clone())
                .collect();
            ws.sort();
            ws
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test 1: byte-for-byte golden match
// ---------------------------------------------------------------------------

#[test]
fn cells_golden_match() {
    for name in GROUPS {
        // F4 is the slow one; still run it in debug (the suite is small).
        let g = group_from_golden(name);
        let golden = common::golden(&format!("cells_{name}"));

        let res = klcells(&g, &CellsOpts::default()).unwrap();

        // Exact nested-cell equality (cells already canonicalized like golden).
        let want_cells = golden_cells(&golden);
        assert_eq!(
            res.cells, want_cells,
            "{name}: cells nested-list must match golden EXACTLY"
        );

        let want_ncells = golden["ncells"].as_u64().unwrap() as usize;
        assert_eq!(res.cells.len(), want_ncells, "{name}: ncells mismatch");

        let want_nstarreps = golden["nstarreps"].as_u64().unwrap() as usize;
        assert_eq!(
            res.n_star_reps, want_nstarreps,
            "{name}: nstarreps mismatch"
        );
        assert_eq!(
            res.star_reps.len(),
            want_nstarreps,
            "{name}: star_reps length mismatch"
        );

        // Structural invariant: Σ|cell| == order.
        let tot: usize = res.cells.iter().map(|c| c.len()).sum();
        assert_eq!(tot as u128, g.order, "{name}: Σ|cell| != |W|");
    }
}

// ---------------------------------------------------------------------------
// Test 2: cross-check vs the full-table CellData partition
// ---------------------------------------------------------------------------

#[test]
fn cells_match_full_table() {
    for name in GROUPS {
        let g = group_from_golden(name);
        let res = klcells(&g, &CellsOpts::default()).unwrap();
        let got = cell_word_sets(&res.cells);

        let t = build_table(&g);
        let want = full_table_word_sets(&t);

        assert_eq!(
            got, want,
            "{name}: klcells partition must equal full-table CellData.lcells"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: recursion depth + synthetic tier soundness
// ---------------------------------------------------------------------------

/// B4 exercises the parabolic recursion B4 → B3 → B2 → A1 → rank-0; covered
/// implicitly by `cells_golden_match`.  This test additionally pins the
/// size-tier pre-partition: forcing tiny tier thresholds (so the descent-set
/// and generalised-tau branches activate on a tiny group) must produce an
/// IDENTICAL result to the default-tier run.  If the right-descent or
/// generalised-tau pre-partition ever split a left cell, this would diverge.
#[test]
fn recursion_depth_and_tier_soundness() {
    // B4 drives the recursion (B4 → B3 → B2 → A1 → rank-0); H3 carries a rich mu
    // pool.  Both must survive extreme tier thresholds unchanged.
    for name in ["B4", "H3"] {
        let g = group_from_golden(name);

        let default_run = klcells(&g, &CellsOpts::default()).unwrap();

        // tier_direct = 1, tier_tau = 3: ANY induced set with > 1 element takes
        // the pre-partition path, and any with > 3 uses the generalised-tau key.
        // This forces both pre-partition branches on essentially every step.
        let tiered = klcells_with_tiers(&g, &CellsOpts::default(), 1, 3).unwrap();
        assert_eq!(
            tiered.cells, default_run.cells,
            "{name}: tiny-tier pre-partition must reproduce the default-tier cells exactly"
        );
        assert_eq!(
            tiered.n_star_reps, default_run.n_star_reps,
            "{name}: tiny-tier nstarreps must match default"
        );

        // Force ONLY the descent-set tier (tier_tau huge so the tau branch never
        // fires) — still identical.
        let descent_only = klcells_with_tiers(&g, &CellsOpts::default(), 1, 1_000_000).unwrap();
        assert_eq!(
            descent_only.cells, default_run.cells,
            "{name}: descent-set-only pre-partition must reproduce default cells"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: parallel klcells == sequential klcells (Task P6 determinism)
// ---------------------------------------------------------------------------

/// The multiset of star-rep vertex-word sets (each rep's `x`, sorted), for an
/// order-insensitive set-equality comparison of the star reps.
fn star_rep_word_sets(res: &rustcox_core::kl::KlCellsResult) -> BTreeSet<Vec<Word>> {
    res.star_reps
        .iter()
        .map(|g| {
            let mut v = g.x.clone();
            v.sort();
            v
        })
        .collect()
}

/// `klcells` must produce IDENTICAL `cells` and star-class reps whether the
/// relative-KL wavefront runs sequentially or on 2 / 4 threads.  This pins the
/// two-phase deterministic interning of the parallel `relklpols` (Task P6):
/// only *when* a block is computed may change, never the cell partition or the
/// pool order that the W-graphs are built from.
#[test]
fn parallel_cells_equal_sequential() {
    use rustcox_core::kl::klcells;

    for name in ["B4", "H3", "F4"] {
        let g = group_from_golden(name);

        // Sequential baseline (threads = Some(1)).
        let seq = klcells(
            &g,
            &CellsOpts {
                all_cells: true,
                threads: Some(1),
            },
        )
        .unwrap();
        let seq_star_sets = star_rep_word_sets(&seq);

        for &t in &[2usize, 4] {
            let par = klcells(
                &g,
                &CellsOpts {
                    all_cells: true,
                    threads: Some(t),
                },
            )
            .unwrap();

            // (a) the canonicalized cell partition is byte-identical.
            assert_eq!(
                par.cells, seq.cells,
                "{name} (threads={t}): cells must equal the sequential partition EXACTLY"
            );
            // (b) the number of star-class reps matches.
            assert_eq!(
                par.n_star_reps, seq.n_star_reps,
                "{name} (threads={t}): n_star_reps mismatch"
            );
            assert_eq!(
                par.star_reps.len(),
                seq.star_reps.len(),
                "{name} (threads={t}): star_reps length mismatch"
            );
            // (c) the star reps' vertex-word sets match as a set.
            assert_eq!(
                star_rep_word_sets(&par),
                seq_star_sets,
                "{name} (threads={t}): star-rep vertex sets must match the sequential reps"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 5: all_cells = false inverse closure + subset of full cells
// ---------------------------------------------------------------------------

/// With `all_cells=false`, each output cell keeps only elements whose inverse
/// is also in that cell.  Assert:
///   (a) every cell is inverse-closed (word's inverse-word also present);
///   (b) each reduced cell is a SUBSET of the matching full (all_cells=true)
///       cell — matched by the rep word (first element) that they share.
#[test]
fn allcells_false_inverse_closure() {
    let g = group_from_golden("B3");

    let full = klcells(&g, &CellsOpts::default()).unwrap();
    let opts_false = CellsOpts {
        all_cells: false,
        threads: None,
    };
    let reduced = klcells(&g, &opts_false).unwrap();

    // Build full cells as canonical-word-sets for the subset check; index each
    // full cell by EVERY word it contains so we can locate the parent cell of
    // any reduced-cell element.
    let full_sets: Vec<BTreeSet<Word>> = full
        .cells
        .iter()
        .map(|c| c.iter().cloned().collect())
        .collect();

    // Helper: inverse of a reduced word, re-canonicalized to a reduced word.
    let inverse_word = |w: &Word| -> Word {
        let p = g.word_to_perm(w).inverse();
        g.perm_to_word(&p)
    };

    assert_eq!(
        reduced.cells.len(),
        full.cells.len(),
        "B3: all_cells=false must yield the same NUMBER of cells"
    );

    for cell in &reduced.cells {
        // (a) inverse closure.
        let words: BTreeSet<Word> = cell.iter().cloned().collect();
        for w in cell {
            let inv = inverse_word(w);
            assert!(
                words.contains(&inv),
                "B3 all_cells=false: cell {cell:?} not inverse-closed (missing inv of {w:?} = {inv:?})"
            );
        }

        // (b) subset of exactly one full cell.
        let any = cell.first().expect("reduced cell is non-empty");
        let parent = full_sets
            .iter()
            .find(|s| s.contains(any))
            .unwrap_or_else(|| panic!("B3: no full cell contains reduced element {any:?}"));
        for w in cell {
            assert!(
                parent.contains(w),
                "B3 all_cells=false: reduced word {w:?} not in its full cell"
            );
        }
    }

    // The union of reduced cells equals the set of self-inverse-closed
    // elements; in particular Σ|reduced cell| ≤ |W|.
    let tot: usize = reduced.cells.iter().map(|c| c.len()).sum();
    assert!(
        tot <= g.order as usize,
        "B3 all_cells=false: too many elements"
    );
    assert!(tot > 0, "B3 all_cells=false: produced no elements");
}
