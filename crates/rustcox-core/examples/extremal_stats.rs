//! Extremal-pair statistics for full KL tables (du Cloux compression analysis).
//!
//! **Definition.** For a Bruhat-comparable pair `y <_B w` (strict), `y` is
//! *extremal* for `w` iff `L(w) ⊆ L(y)` and `R(w) ⊆ R(y)`, where `L`/`R` are
//! the left/right descent sets.
//!
//! **Reduction identity.** For any descent `s ∈ L(w)` with `s·y > y`:
//! `P_{y,w} = P_{sy,w}`.  Similarly for right descents.  Repeatedly pushing `y`
//! upward through descents of `w` on both sides until fixpoint yields an extremal
//! pair `y'` with the same polynomial.  The distinct-polynomial value sets over
//! all comparable pairs and over extremal pairs alone are therefore equal.
//!
//! This tool:
//!   1. Builds the full KL table for the given TYPE.
//!   2. On B3 (or the requested type if it is B3): verifies the reduction
//!      identity empirically — for every comparable pair, pushes `y` to the
//!      extremal closure and asserts the polynomial is unchanged.
//!   3. Counts comparable pairs, extremal pairs, distinct polynomials overall,
//!      and distinct polynomials over extremal pairs; asserts the two distinct-
//!      polynomial sets are equal (would falsify the reduction if violated).
//!   4. Prints one summary line and exits.
//!
//! Permitted types: A4, B3, B4, D4, H3, F4 (and smaller).
//!
//! Usage: `cargo run --release --example extremal_stats -- B4`

use std::collections::HashSet;

use rustcox_core::{
    element::{ElmIdx, Perm},
    group::CoxeterGroup,
    kl::{desc_mask, is_extremal, klpolynomials, push_to_extremal, KlOpts, PushCtx},
    laurent::Laurent,
};

/// Groups this tool is permitted to run on. The last four (H4, D6, B6, E6)
/// build full KL tables needing GBs of RAM — run those on an HPC node only.
const ALLOWED: &[&str] = &[
    "A1", "A2", "A3", "B2", "B3", "A4", "B4", "D4", "H3", "F4", "H4", "D6", "B6", "E6",
];

fn main() {
    let typ = std::env::args()
        .nth(1)
        .unwrap_or_else(|| panic!("usage: extremal_stats <{}>", ALLOWED.join("|")));
    if !ALLOWED.contains(&typ.as_str()) {
        panic!(
            "group {typ:?} is out of scope; allowed: {}",
            ALLOWED.join(", ")
        );
    }

    run(&typ);
}

// ---------------------------------------------------------------------------
// Core logic
// ---------------------------------------------------------------------------

fn run(typ: &str) {
    let group = CoxeterGroup::from_type(typ).expect("build group");
    let opts = KlOpts::equal(group.rank);
    let table = klpolynomials(&group, &opts).expect("kl table");

    let n = table.n() as ElmIdx;

    // Precompute perms (needed for descent-set queries).
    let perms: Vec<Perm> = (0..n)
        .map(|i| group.word_to_perm(&table.elms.elms[i as usize]))
        .collect();

    // Precompute left and right descent sets for every element.
    // left_desc[i]  = bitmask of generators in L(elms[i])
    // right_desc[i] = bitmask of generators in R(elms[i])
    // (rank ≤ 8 for all permitted types, so u64 is more than enough)
    let rank = group.rank;
    let left_desc: Vec<u64> = (0..n as usize)
        .map(|i| desc_mask(&group.left_descents(&perms[i])))
        .collect();
    let right_desc: Vec<u64> = (0..n as usize)
        .map(|i| desc_mask(&group.right_descents(&perms[i])))
        .collect();

    // The element table's `lft` gives left multiplication: lft(w, s) = s·w index.
    // We need right multiplication: rft(w, s) = w·s.
    // rft(w, s) = inva[ lft( inva[w], s ) ]
    // because: w·s = (s^{-1} · w^{-1})^{-1} = (s · w^{-1})^{-1}
    // (s is an involution, so s^{-1}=s)
    // Precompute a flat right-mult table: rft[w * rank + s] = index of w·s.
    let rft: Vec<ElmIdx> = {
        let inva = &table.elms.inva;
        let lft = &table.elms.lft;
        (0..n as usize)
            .flat_map(|w| {
                let w_inv = inva[w] as usize;
                (0..rank).map(move |s| {
                    let sw_inv = lft[w_inv * rank + s];
                    inva[sw_inv as usize]
                })
            })
            .collect()
    };

    let push_ctx = PushCtx {
        left_desc: &left_desc,
        right_desc: &right_desc,
        lft: &table.elms.lft,
        rft: &rft,
        rank,
    };

    // --- B3 reduction-identity verification ---
    let is_b3 = typ == "B3";
    if is_b3 {
        eprintln!("B3: verifying reduction identity for all comparable pairs...");
        let mut checked = 0u64;
        for w in 1..n {
            for y in 0..w {
                if !table.bruhat_leq(y, w) {
                    continue;
                }
                // Compute the extremal closure of y w.r.t. w by push-up.
                let y_ext = push_to_extremal(y, w, &push_ctx);
                // Verify the polynomial is unchanged.
                let pol_orig = table.pol(y, w).expect("comparable => pol present");
                let pol_ext = table.pol(y_ext, w).expect("extremal y' must be <= w");
                assert_eq!(
                    pol_orig, pol_ext,
                    "REDUCTION IDENTITY FAILED for typ=B3 y={y} w={w} y'={y_ext}: \
                     P_{{y,w}}={pol_orig:?} but P_{{y',w}}={pol_ext:?}",
                );
                // Verify y_ext is indeed extremal.
                assert!(
                    is_extremal(y_ext, w, &left_desc, &right_desc),
                    "push_to_extremal returned non-extremal y'={y_ext} for y={y} w={w}"
                );
                checked += 1;
            }
        }
        eprintln!("B3: reduction identity verified on {checked} comparable pairs. PASS.");
    }

    // --- Full statistics pass ---
    let mut n_comparable: u64 = 0;
    let mut n_extremal: u64 = 0;
    let mut all_pols: HashSet<Laurent> = HashSet::new();
    let mut extremal_pols: HashSet<Laurent> = HashSet::new();

    for w in 1..n {
        for y in 0..w {
            if !table.bruhat_leq(y, w) {
                continue;
            }
            n_comparable += 1;
            let pol = table.pol(y, w).expect("comparable => pol present").clone();
            all_pols.insert(pol.clone());

            if is_extremal(y, w, &left_desc, &right_desc) {
                n_extremal += 1;
                extremal_pols.insert(pol);
            }
        }
    }

    // Sanity: every polynomial value is achieved at an extremal pair.
    if all_pols != extremal_pols {
        eprintln!(
            "ASSERTION FAILED for {typ}: distinct-polynomial sets differ!\n\
             all_pols has {} entries, extremal_pols has {} entries.\n\
             This FALSIFIES the reduction identity — investigation required.",
            all_pols.len(),
            extremal_pols.len()
        );
        std::process::exit(1);
    }

    let frac = if n_comparable > 0 {
        n_extremal as f64 / n_comparable as f64
    } else {
        0.0
    };

    println!(
        "EXTREMAL {typ} order={order} comparable={c} extremal={e} frac={frac:.4} \
         distinct_pols={p} distinct_pols_extremal={pe}",
        order = group.order,
        c = n_comparable,
        e = n_extremal,
        frac = frac,
        p = all_pols.len(),
        pe = extremal_pols.len(),
    );
}

// Helpers (`desc_mask`, `is_extremal`, `push_to_extremal`, `PushCtx`) live in
// `rustcox_core::kl::extremal` (see issue #12 / plan
// `docs/superpowers/plans/2026-06-18-memory-compression-relkl.md`) so the
// `relklpols` memory compression refactor can reuse them.
