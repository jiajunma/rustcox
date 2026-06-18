//! Extremal-pair helpers for Kazhdan–Lusztig polynomial compression.
//!
//! A pair `(y, w)` with `y ≤_B w` is *extremal* iff
//! `L(w) ⊆ L(y)` and `R(w) ⊆ R(y)`, where `L` and `R` denote left and right
//! descent sets respectively.
//!
//! For every non-extremal pair the KL polynomial equals the polynomial of an
//! extremal pair, by the reduction identity:
//!
//! - if `s ∈ L(w)` with `s·y > y` (i.e. `s ∉ L(y)`), then
//!   `P_{y,w} = P_{sy,w}`;
//! - if `s ∈ R(w)` with `y·s > y` (i.e. `s ∉ R(y)`), then
//!   `P_{y,w} = P_{ys,w}`.
//!
//! Repeatedly applying these pushes drives `y` to its canonical extremal
//! closure for `w`, returned by [`push_to_extremal`].
//!
//! The reduction is exact (no information loss): the set of distinct KL
//! polynomial values over all comparable pairs equals the set over extremal
//! pairs alone, as verified empirically by
//! [`examples/extremal_stats.rs`](../../../../examples/extremal_stats.rs) on
//! H4 / D6 / E6, and by a brute-force per-pair check on B3.
//!
//! Atlas of Lie Groups uses the same compression under the name *primitive*
//! (`sources/stand-alone/KLread.cpp:162–224`); the encoded row stores a
//! bitmap of primitive positions plus one polynomial id per primitive entry,
//! and reconstructs non-primitive entries on demand via the dual of
//! [`push_to_extremal`].  See plan
//! [`docs/superpowers/plans/2026-06-18-memory-compression-relkl.md`] and
//! tracking issue #12.
//!
//! This module is intentionally storage-agnostic: the helpers take
//! precomputed `lft`/`rft` multiplication tables and descent bitmasks so they
//! can be reused by both the full-table `klpolynomials` path and the
//! parabolic-induction `relklpols` path.

use crate::element::ElmIdx;

/// Pack a sorted slice of generator indices `[s_0, s_1, …]` into a `u64`
/// bitmask with bit `s_i` set.
///
/// Generators come from a finite Coxeter group of rank ≤ 64; all currently
/// supported types have rank ≤ 8, so a `u64` is far more than enough.
#[inline]
pub fn desc_mask(gens: &[u8]) -> u64 {
    gens.iter().fold(0u64, |acc, &g| acc | (1u64 << g))
}

/// Return `true` iff `(y, w)` is extremal: `L(w) ⊆ L(y)` and `R(w) ⊆ R(y)`.
///
/// `left_desc[i]` and `right_desc[i]` must be the descent bitmasks of element
/// `i`.  Bruhat-comparability of `(y, w)` is **not** checked — the helper is
/// `O(1)` and only inspects descent masks.  Callers that need extremality
/// only for comparable pairs should filter first.
#[inline]
pub fn is_extremal(y: ElmIdx, w: ElmIdx, left_desc: &[u64], right_desc: &[u64]) -> bool {
    let lw = left_desc[w as usize];
    let rw = right_desc[w as usize];
    let ly = left_desc[y as usize];
    let ry = right_desc[y as usize];
    // L(w) ⊆ L(y)  ⟺  lw & ~ly == 0
    // R(w) ⊆ R(y)  ⟺  rw & ~ry == 0
    (lw & !ly == 0) && (rw & !ry == 0)
}

/// Shared context for [`push_to_extremal`].
///
/// All fields are borrowed; callers precompute the multiplication tables and
/// descent masks once and reuse them.  `lft[w * rank + s] = s·w` and
/// `rft[w * rank + s] = w·s` are flat row-major u32 tables of length
/// `n_elements * rank`.
pub struct PushCtx<'a> {
    pub left_desc: &'a [u64],
    pub right_desc: &'a [u64],
    pub lft: &'a [ElmIdx],
    pub rft: &'a [ElmIdx],
    pub rank: usize,
}

/// Push `y` upward through descents of `w` until the extremal fixpoint.
///
/// At each step:
/// - if `s ∈ L(w) \ L(y)` exists, replace `y ← s·y` (= `lft[y · rank + s]`).
///   The smallest such `s` is chosen (deterministic, lex-order tie-break).
/// - else if `s ∈ R(w) \ R(y)` exists, replace `y ← y·s`.
/// - else `y` is extremal, return it.
///
/// The result `y'` satisfies:
/// - `y' ≤_B w` whenever `y ≤_B w` (each push strictly raises `y`, and stays
///   `≤ w` by the KL reduction lemma — see e.g. Geck–Pfeiffer §6.4);
/// - `y'` is extremal for `w`;
/// - `P_{y, w} = P_{y', w}` (the KL reduction identity).
///
/// Termination: each push strictly increases `len(y)`; the chain is bounded
/// by `len(w)`.
pub fn push_to_extremal(mut y: ElmIdx, w: ElmIdx, ctx: &PushCtx<'_>) -> ElmIdx {
    let lw = ctx.left_desc[w as usize];
    let rw = ctx.right_desc[w as usize];
    loop {
        // Left push: pick smallest s ∈ L(w) with s ∉ L(y).
        let lmask = lw & !ctx.left_desc[y as usize];
        if lmask != 0 {
            let s = lmask.trailing_zeros() as usize;
            let sy = ctx.lft[y as usize * ctx.rank + s];
            debug_assert!(
                sy > y,
                "push_to_extremal: lft says sy={sy} ≤ y={y} for s={s} (table bug)"
            );
            y = sy;
            continue;
        }
        // Right push: pick smallest s ∈ R(w) with s ∉ R(y).
        let rmask = rw & !ctx.right_desc[y as usize];
        if rmask != 0 {
            let s = rmask.trailing_zeros() as usize;
            let ys = ctx.rft[y as usize * ctx.rank + s];
            debug_assert!(
                ys > y,
                "push_to_extremal: rft says ys={ys} ≤ y={y} for s={s} (table bug)"
            );
            y = ys;
            continue;
        }
        // No further push available — y is extremal for w.
        return y;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// Behaviour is pinned against precomputed B3 / A3 / B2 element tables.  These
// groups are small enough that we can enumerate every comparable pair and
// verify two properties:
//
//   1. `push_to_extremal(y, w)` lands on an extremal pair.
//   2. The result is invariant under permuting the order of generators that
//      qualify for a push at any given step (no — we pick the smallest, so
//      the result is deterministic; but the *extremality* of the result
//      doesn't depend on the choice).
//
// Larger groups (D4, B4, F4, …) are exercised by the examples binary and by
// the future tests that compare `klmat` over extremal-only against the dense
// reference.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::CoxeterGroup;
    use crate::kl::{klpolynomials, KlOpts};

    /// Build `(left_desc, right_desc, lft, rft, rank)` for the group's full
    /// element table.  This duplicates the setup done in
    /// `examples/extremal_stats.rs`, kept inline so this test does not depend
    /// on the example binary.
    fn build_ctx(typ: &str) -> (Vec<u64>, Vec<u64>, Vec<ElmIdx>, Vec<ElmIdx>, usize) {
        let group = CoxeterGroup::from_type(typ).expect("build group");
        let opts = KlOpts::equal(group.rank);
        let table = klpolynomials(&group, &opts).expect("kl table");
        let n = table.n() as ElmIdx;
        let rank = group.rank;

        let perms: Vec<_> = (0..n)
            .map(|i| group.word_to_perm(&table.elms.elms[i as usize]))
            .collect();
        let left_desc: Vec<u64> = (0..n as usize)
            .map(|i| desc_mask(&group.left_descents(&perms[i])))
            .collect();
        let right_desc: Vec<u64> = (0..n as usize)
            .map(|i| desc_mask(&group.right_descents(&perms[i])))
            .collect();

        // rft(w, s) = inva[ lft( inva[w], s ) ] (s is an involution).
        let inva = &table.elms.inva;
        let lft_src = &table.elms.lft;
        let lft: Vec<ElmIdx> = lft_src.to_vec();
        let rft: Vec<ElmIdx> = (0..n as usize)
            .flat_map(|w| {
                let w_inv = inva[w] as usize;
                (0..rank).map(move |s| {
                    let sw_inv = lft_src[w_inv * rank + s];
                    inva[sw_inv as usize]
                })
            })
            .collect();

        (left_desc, right_desc, lft, rft, rank)
    }

    #[test]
    fn is_extremal_trivial_top() {
        // For w = identity (index 0), L(w) = R(w) = empty, so every y is
        // trivially extremal.  In particular (0, 0) is extremal.
        let (ld, rd, _, _, _) = build_ctx("B3");
        assert!(is_extremal(0, 0, &ld, &rd));
    }

    #[test]
    fn is_extremal_self_pair() {
        // (w, w) is always extremal: L(w) ⊆ L(w), R(w) ⊆ R(w).
        let (ld, rd, _, _, _) = build_ctx("B3");
        for w in 0..ld.len() as ElmIdx {
            assert!(is_extremal(w, w, &ld, &rd), "(w,w) must be extremal, w={w}");
        }
    }

    #[test]
    fn push_to_extremal_is_extremal_b3() {
        // For every comparable pair (y, w) in B3, push_to_extremal must land
        // on an extremal pair.
        let (ld, rd, lft, rft, rank) = build_ctx("B3");
        let ctx = PushCtx {
            left_desc: &ld,
            right_desc: &rd,
            lft: &lft,
            rft: &rft,
            rank,
        };
        let n = ld.len() as ElmIdx;
        for w in 1..n {
            for y in 0..w {
                let y_ext = push_to_extremal(y, w, &ctx);
                assert!(
                    is_extremal(y_ext, w, &ld, &rd),
                    "push landed on non-extremal: y={y} w={w} y'={y_ext}"
                );
            }
        }
    }

    #[test]
    fn push_to_extremal_idempotent_a3() {
        // push_to_extremal must be idempotent: pushing an already-extremal
        // pair returns the same y.
        let (ld, rd, lft, rft, rank) = build_ctx("A3");
        let ctx = PushCtx {
            left_desc: &ld,
            right_desc: &rd,
            lft: &lft,
            rft: &rft,
            rank,
        };
        let n = ld.len() as ElmIdx;
        for w in 1..n {
            for y in 0..w {
                let y1 = push_to_extremal(y, w, &ctx);
                let y2 = push_to_extremal(y1, w, &ctx);
                assert_eq!(
                    y1, y2,
                    "push not idempotent: y={y} w={w} y1={y1} y2={y2}"
                );
            }
        }
    }

    #[test]
    fn push_to_extremal_preserves_polynomial_b3() {
        // The reduction identity: P_{y,w} == P_{push(y,w), w}.  Verified
        // pair-by-pair on B3 (same shape as the B3 brute-force check inside
        // examples/extremal_stats.rs, but now exercised as a library test).
        let group = CoxeterGroup::from_type("B3").unwrap();
        let opts = KlOpts::equal(group.rank);
        let table = klpolynomials(&group, &opts).unwrap();
        let n = table.n() as ElmIdx;
        let rank = group.rank;

        let perms: Vec<_> = (0..n)
            .map(|i| group.word_to_perm(&table.elms.elms[i as usize]))
            .collect();
        let left_desc: Vec<u64> = (0..n as usize)
            .map(|i| desc_mask(&group.left_descents(&perms[i])))
            .collect();
        let right_desc: Vec<u64> = (0..n as usize)
            .map(|i| desc_mask(&group.right_descents(&perms[i])))
            .collect();
        let inva = &table.elms.inva;
        let lft_src = &table.elms.lft;
        let lft: Vec<ElmIdx> = lft_src.to_vec();
        let rft: Vec<ElmIdx> = (0..n as usize)
            .flat_map(|w| {
                let w_inv = inva[w] as usize;
                (0..rank).map(move |s| {
                    let sw_inv = lft_src[w_inv * rank + s];
                    inva[sw_inv as usize]
                })
            })
            .collect();
        let ctx = PushCtx {
            left_desc: &left_desc,
            right_desc: &right_desc,
            lft: &lft,
            rft: &rft,
            rank,
        };

        let mut checked = 0u64;
        for w in 1..n {
            for y in 0..w {
                if !table.bruhat_leq(y, w) {
                    continue;
                }
                let y_ext = push_to_extremal(y, w, &ctx);
                let pol_orig = table.pol(y, w).expect("comparable => pol present");
                let pol_ext = table.pol(y_ext, w).expect("extremal y' must be ≤ w");
                assert_eq!(
                    pol_orig, pol_ext,
                    "REDUCTION IDENTITY FAILED on B3: y={y} w={w} y'={y_ext}"
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "B3 must have at least one comparable pair");
    }

    #[test]
    fn desc_mask_basic() {
        assert_eq!(desc_mask(&[]), 0);
        assert_eq!(desc_mask(&[0]), 0b1);
        assert_eq!(desc_mask(&[0, 2]), 0b101);
        assert_eq!(desc_mask(&[3, 1]), 0b1010);
        // Duplicates collapse via OR.
        assert_eq!(desc_mask(&[2, 2]), 0b100);
    }
}
