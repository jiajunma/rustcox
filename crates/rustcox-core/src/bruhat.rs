//! Bruhat–Chevalley order on a finite Coxeter group.
//!
//! # Algorithm
//!
//! This module implements the iterative descent-stripping algorithm from
//! PyCox (`bruhatperm`, lines 3622–3653).
//!
//! ## Loop invariant
//!
//! At the start of every iteration we have elements `x` and `y` with
//! `lx = l(x)`, `ly = l(y)`, and `lx ≤ ly`.  The claim `x ≤ y` in the
//! original Bruhat order is equivalent to `x ≤ y` after the reductions.
//!
//! Each pass of the loop:
//! 1. Picks a left descent `s` of `y` (a simple root with `y[s] ≥ N`).
//! 2. Replaces `y ← s·y`  (one positive root becomes negative; `ly` drops by 1).
//! 3. If `s` is also a left descent of `x`, replaces `x ← s·x`; `lx` drops by 1.
//!
//! By the standard Bruhat recursion:
//!   `x ≤ y  ⟺  s·x ≤ s·y`  (when `l(s·x) = l(x) − 1`)
//!   `x ≤ y  ⟺  x ≤ s·y`    (when `l(s·x) = l(x) + 1`)
//!
//! Termination: `ly` decreases by 1 on every iteration, so the loop
//! terminates after at most `ly` steps.
//!
//! Terminal condition: `lx == 0` means `x` is the identity, which is
//! below every element; otherwise we need `lx == ly && x == y`.
//!
//! ## Allocation profile
//!
//! Each left-multiplication allocates exactly one `Perm` (a `Vec<u32>` of
//! length `2N`).  Per call, this is O(l(y)) allocations, each of size O(N).
//! For groups up to F₄ (N = 24) this is negligible.

use crate::{element::Perm, group::CoxeterGroup};

/// Test whether `x ≤ y` in the Bruhat–Chevalley order on `group`.
///
/// Both `x` and `y` must be full permutations (length `2 * group.n_pos`)
/// in the same group.
///
/// The algorithm is the iterative descent-stripping method of PyCox
/// (`bruhatperm`).  See the module-level docs for the loop invariant.
pub fn leq(group: &CoxeterGroup, x: &Perm, y: &Perm) -> bool {
    let n = group.n_pos as usize;

    // Fast-path: identity ≤ everything; x ≤ x.
    // PyCox: `if x == tuple(range(2*W.N)) or x == y: return True`
    if *x == Perm::identity(2 * n) || x == y {
        return true;
    }
    // Fast-path: nothing is ≤ identity except identity itself.
    // PyCox: `elif y == tuple(range(2*W.N)): return x == y`
    if *y == Perm::identity(2 * n) {
        // x ≠ identity (handled above), so return false
        return false;
    }

    let mut lx = group.perm_length(x);
    let mut ly = group.perm_length(y);

    // If x is already longer than y it cannot be below y.
    if lx > ly {
        return false;
    }

    // Work on owned (mutable) copies.
    let mut cx = x.clone();
    let mut cy = y.clone();

    // Main loop — invariant: cx ≤ cy in the original order iff cx ≤ cy now.
    // Terminates because ly decreases by 1 each iteration.
    while lx < ly && lx != 0 && ly != 0 {
        // Pick the smallest left descent s of cy.
        // s is a left descent of cy iff cy[simple_root[s]] ≥ N,
        // which by the mapping simple_root[s] == s for irreducible groups
        // reduces to cy[s] ≥ N.  For generality we use the PyCox approach:
        // find the first position in 0..N with cy[position] ≥ N, then
        // identify which simple root that is.
        //
        // PyCox: `s = 0; while y[s] < W.N: s += 1`
        // This scans positions 0, 1, 2, … and stops at the first i where
        // cy[i] ≥ N.  Because simple roots are 0..rank and lie in 0..N,
        // the first such i IS the index of the leftmost left-descent generator.
        // (Positive roots are 0..N; negative roots are N..2N.)
        let s = (0..n)
            .find(|&i| cy.0[i] as usize >= n)
            .expect("ly > 0 implies at least one left descent");

        // If s is also a left descent of cx, strip cx too.
        // PyCox: `if x[s]>=W.N: x = tuple([x[r] for r in W.permgens[s]]); lx-=1`
        if cx.0[s] as usize >= n {
            // Left-multiply cx by the generator whose permgen maps position s.
            // We need to identify which generator s corresponds to.
            // By PyCox convention, the permgen scan walks positions 0..N and
            // position i is the simple root of generator i.  But for reducible
            // groups simple_root[gen] may not equal gen.
            //
            // However, the PyCox scan `while y[s] < W.N: s += 1` finds a
            // *position index* s (not a generator index).  In PyCox,
            // `W.permgens[s]` is the permgen of generator s, and simple root s
            // is at position s (PyCox always has simple root s at position s
            // in the root list — for all built-in types).
            //
            // Our Rust group guarantees simple_root[s] == s for irreducible
            // groups; for reducible groups simple_root may differ.  To be safe
            // we find the generator g whose simple_root[g] == s.
            let g = group
                .simple_root
                .iter()
                .position(|&sr| sr == s)
                .expect("every position in 0..rank is a simple root");

            cx = group.permgens[g].then(&cx);
            lx -= 1;
        }

        // Always strip cy.
        // PyCox: `y = tuple([y[r] for r in W.permgens[s]]); ly -= 1`
        let g = group
            .simple_root
            .iter()
            .position(|&sr| sr == s)
            .expect("every position in 0..rank is a simple root");
        cy = group.permgens[g].then(&cy);
        ly -= 1;
    }

    // PyCox: `return lx == 0 or (lx == ly and x == y)`
    lx == 0 || (lx == ly && cx == cy)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{enumerate::ElementTable, group::CoxeterGroup};
    use std::collections::HashSet;

    // -----------------------------------------------------------------------
    // Test 1: identity ≤ everything in B3
    // -----------------------------------------------------------------------

    /// The identity element is the unique minimum of the Bruhat order.
    /// For every element `w` in B3, `leq(id, w)` must be true.
    #[test]
    fn identity_below_everything() {
        let group = CoxeterGroup::from_type("B3").unwrap();
        let table = ElementTable::build(&group);
        let id = group.id_perm();

        for (i, word) in table.elms.iter().enumerate() {
            let w = group.word_to_perm(word);
            assert!(
                leq(&group, &id, &w),
                "identity should be ≤ every element, failed at index {i} word={word:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: length is monotone for strict Bruhat comparisons in B2
    // -----------------------------------------------------------------------

    /// If `x < y` (i.e. `x ≤ y` and `x ≠ y`) then `l(x) < l(y)`.
    ///
    /// Checked over all 8 × 8 = 64 pairs in B2.
    #[test]
    fn length_monotone() {
        let group = CoxeterGroup::from_type("B2").unwrap();
        let table = ElementTable::build(&group);

        let perms: Vec<Perm> = table.elms.iter().map(|w| group.word_to_perm(w)).collect();

        for i in 0..table.len() {
            for j in 0..table.len() {
                if leq(&group, &perms[i], &perms[j]) && perms[i] != perms[j] {
                    assert!(
                        table.lengths[i] < table.lengths[j],
                        "length_monotone violated: l({i})={} not < l({j})={} but {i} < {j} in Bruhat",
                        table.lengths[i],
                        table.lengths[j]
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helper: build the subword-closure set for a given element
    //
    // For word `y_word` of length `l`, enumerate all 2^l subsequences.
    // For each subsequence compute its element (via word_to_perm) and
    // collect the CoxElm into a HashSet.
    // Then `x ≤ y` iff `coxelm(x)` is in that set.
    // -----------------------------------------------------------------------
    fn subword_closure(group: &CoxeterGroup, y_word: &[u8]) -> HashSet<Perm> {
        let l = y_word.len();
        let mut set = HashSet::new();
        for mask in 0u64..(1u64 << l) {
            let subword: Vec<u8> = (0..l)
                .filter(|&i| (mask >> i) & 1 == 1)
                .map(|i| y_word[i])
                .collect();
            let p = group.word_to_perm(&subword);
            set.insert(p);
        }
        set
    }

    // -----------------------------------------------------------------------
    // Test 3: brute-force subword cross-check in A3 and B3
    //
    // For every ordered pair (x, y):
    //   leq(x, y) == (perm(x) ∈ subword_closure(y))
    //
    // A3: 24 × 24 = 576 pairs, max l(y) = 6 → max 64 subsequences per y.
    // B3: 48 × 48 = 2304 pairs, max l(y) = 9 → max 512 subsequences per y.
    // -----------------------------------------------------------------------

    fn cross_check_group(type_str: &str) {
        let group = CoxeterGroup::from_type(type_str).unwrap();
        let table = ElementTable::build(&group);
        let order = table.len();

        // Precompute perms and subword closures for each element.
        let perms: Vec<Perm> = table.elms.iter().map(|w| group.word_to_perm(w)).collect();

        // Cache: for each y, the set of perms that are ≤ y via subword.
        let closures: Vec<HashSet<Perm>> = table
            .elms
            .iter()
            .map(|y_word| subword_closure(&group, y_word))
            .collect();

        for i in 0..order {
            for j in 0..order {
                let bruhat_result = leq(&group, &perms[i], &perms[j]);
                let subword_result = closures[j].contains(&perms[i]);
                assert_eq!(
                    bruhat_result,
                    subword_result,
                    "{type_str}: leq({i},{j}) = {bruhat_result} but subword says {subword_result}; x={:?} y={:?}",
                    table.elms[i],
                    table.elms[j]
                );
            }
        }
    }

    #[test]
    fn brute_force_subword_crosscheck_a3() {
        cross_check_group("A3");
    }

    #[test]
    fn brute_force_subword_crosscheck_b3() {
        cross_check_group("B3");
    }

    // -----------------------------------------------------------------------
    // Test 4: antisymmetry in B2
    //
    // leq(x, y) && leq(y, x) ⟺ x == y
    // -----------------------------------------------------------------------

    #[test]
    fn antisymmetry() {
        let group = CoxeterGroup::from_type("B2").unwrap();
        let table = ElementTable::build(&group);
        let perms: Vec<Perm> = table.elms.iter().map(|w| group.word_to_perm(w)).collect();
        let n = table.len();
        for i in 0..n {
            for j in 0..n {
                let both = leq(&group, &perms[i], &perms[j]) && leq(&group, &perms[j], &perms[i]);
                let equal = perms[i] == perms[j];
                assert_eq!(
                    both,
                    equal,
                    "antisymmetry violated at i={i}, j={j}: leq(i,j)={}, leq(j,i)={}, equal={equal}",
                    leq(&group, &perms[i], &perms[j]),
                    leq(&group, &perms[j], &perms[i])
                );
            }
        }
    }
}
