//! Canonical element table for a finite Coxeter group.
//!
//! Builds the complete set of group elements in canonical order
//! (sorted by `(length, canonical-word lex)`), together with the
//! precomputed maps `lft`, `inva`, and `aw0`.
//!
//! # Composition convention (same as `group.rs`)
//!
//! `then(p, q)[i] = q[p[i]]` — apply `p` first, then `q`.
//!
//! For LEFT multiplication by generator `s`:
//! `perm(s · w) = permgens[s].then(&perm_w)`
//!
//! For RIGHT multiplication by `w0`:
//! `perm(w · w0) = perm_w.then(longest_perm)`
//! → `aw0[w]` = index of `w · w0`
//!
//! # Left-multiplication verification (B2 unit test below)
//!
//! `lft(w, s) < w  ⟺  s ∈ left_descents(w)` — because the table is
//! sorted by increasing length, shorter elements always have smaller
//! indices than longer ones, and equal-length elements are never each
//! other's s-image (that would require l(s·w) = l(w), impossible for
//! simple s).
//!
//! # lft storage layout
//!
//! `lft` is a flat row-major `Vec<ElmIdx>` of length `|W| * rank`.
//! Use `self.lft(w, s)` to access entry `(w, s)`.  Flat layout
//! avoids pointer-chasing in the KL hot path.

use std::collections::HashMap;

use crate::{
    element::{CoxElm, ElmIdx, Perm, Word},
    group::CoxeterGroup,
};

// ---------------------------------------------------------------------------
// Public type
// ---------------------------------------------------------------------------

/// Canonical element table for a finite Coxeter group.
///
/// Elements are stored in canonical order: sorted by `(length, lex)` where
/// `lex` compares canonical reduced words.  This matches PyCox's `allwords`
/// output order and is the index base for all KL data.
pub struct ElementTable {
    /// Canonical reduced words, one per element, in canonical order.
    pub elms: Vec<Word>,
    /// CoxElms (rank-length image tuples), parallel to `elms`.
    pub coxelms: Vec<CoxElm>,
    /// Map from `CoxElm` back to its canonical index.
    pub index: HashMap<CoxElm, ElmIdx>,
    /// `lengths[i]` = length of `elms[i]`.
    pub lengths: Vec<u32>,
    /// `inva[i]` = canonical index of `elms[i]^{-1}`.
    pub inva: Vec<ElmIdx>,
    /// `aw0[i]` = canonical index of `elms[i] · w₀` (RIGHT mult by w0).
    ///
    /// Equivalently: `perm(elms[i]) · longest_perm`.
    /// Invariant: `lengths[aw0[i]] == N − lengths[i]`.
    pub aw0: Vec<ElmIdx>,
    /// Flat row-major left-multiplication table of length `|W| * rank`.
    ///
    /// Access via `self.lft(w, s)` = index of `s · elms[w]`.
    /// Invariant: `lft(i, s) < i  ⟺  s ∈ left_descents(elms[i])`.
    pub lft: Vec<ElmIdx>,
    /// Number of generators (rank of the Coxeter group).
    pub rank: usize,
}

// ---------------------------------------------------------------------------
// impl ElementTable
// ---------------------------------------------------------------------------

impl ElementTable {
    /// Return `lft(w, s)` = canonical index of `s · elms[w]`.
    ///
    /// Flat row-major accessor: `lft[w * rank + s]`.
    #[inline]
    pub fn lft(&self, w: ElmIdx, s: usize) -> ElmIdx {
        self.lft[w as usize * self.rank + s]
    }

    /// Build the complete element table for `group`.
    ///
    /// ## Algorithm
    ///
    /// 1. **BFS by LEFT multiplication** from the identity, using a
    ///    `HashSet<CoxElm>` for deduplication within each level.  For
    ///    length `k+1`: extend every element `p` of level `k` by each
    ///    generator `s` that is NOT a left descent of `p`.
    ///
    /// 2. **w0 symmetry trick** (mirrors PyCox `allcoxelms` ≈3925–3971):
    ///    levels with index `> N/2` are derived as `p · w0` from the
    ///    mirror level at index `N − i − 1`.  This halves BFS work.
    ///
    /// 3. **Canonical sort**: for each perm, compute the canonical word via
    ///    `perm_to_word`, then sort all elements by `(length, word lex)`.
    ///
    /// 4. **Map construction**: build `index`, `inva`, `aw0`, `lft`.
    pub fn build(group: &CoxeterGroup) -> Self {
        let rank = group.rank;
        let n = group.n_pos as usize; // positive root count = N
        let order = group.order as usize;
        let longest = group.longest_perm();

        // ---------------------------------------------------------------
        // Phase 1: BFS enumerate all elements as (CoxElm, full Perm) pairs
        // ---------------------------------------------------------------
        //
        // We store (CoxElm, Perm) for all elements.  CoxElm is used for
        // deduplication; Perm is needed for descent checks, `lft`, `inva`,
        // and `aw0` construction.
        //
        // Peak memory (Phase 1 + flat): 2 × order × (rank + 2N) × 4 bytes.
        // F4 → 2 × 1152 × (4+48) × 4 ≈ 478 KB.  H4 → 2 × 14400 × 244 × 4 ≈ 28 MB.

        // Each level: Vec of (CoxElm, Perm)
        let mut levels: Vec<Vec<(CoxElm, Perm)>> = Vec::new();

        // Level 0: identity
        let id_perm = group.id_perm();
        let id_coxelm = id_perm.coxelm_sr(&group.simple_root);
        levels.push(vec![(id_coxelm, id_perm)]);

        let half = n / 2; // floor(N/2)

        for i in 0..n {
            if i < half || (n % 2 == 1 && i == half) {
                // BFS: extend current level by left-multiplying by non-left-descents
                let prev = &levels[i];
                // Upper bound on next level size: prev.len() * rank
                let mut seen: HashMap<CoxElm, Perm> = HashMap::with_capacity(prev.len() * rank);
                for (_, p) in prev {
                    for s in 0..rank {
                        // s is a left descent iff p[simple_root[s]] >= n
                        if p.0[group.simple_root[s]] as usize >= n {
                            continue; // skip: left descent → length would decrease
                        }
                        // Left-multiply: perm(s · w) = permgens[s].then(p)
                        let np = group.permgens[s].then(p);
                        let ce = np.coxelm_sr(&group.simple_root);
                        seen.entry(ce).or_insert(np);
                    }
                }
                // Convert to sorted vec (sort by CoxElm for determinism, but order
                // within a level doesn't matter for correctness — only canonical
                // sort happens later).
                let mut next_level: Vec<(CoxElm, Perm)> = seen.into_iter().collect();
                next_level.sort_by(|a, b| a.0 .0.cmp(&b.0 .0));
                levels.push(next_level);
            } else {
                // w0 symmetry: level[i] = { p · w0 : p ∈ level[N - i - 1] }
                let mirror_idx = n - i - 1;
                let mirror = &levels[mirror_idx];
                let next_level: Vec<(CoxElm, Perm)> = mirror
                    .iter()
                    .map(|(_, p)| {
                        let np = p.then(longest);
                        let ce = np.coxelm_sr(&group.simple_root);
                        (ce, np)
                    })
                    .collect();

                // debug_assert: count mirrors the BFS half
                debug_assert_eq!(
                    next_level.len(),
                    mirror.len(),
                    "w0 symmetry: level {i} count ({}) != mirror level {mirror_idx} count ({})",
                    next_level.len(),
                    mirror.len()
                );

                levels.push(next_level);
            }
        }

        // Flatten all (CoxElm, Perm) pairs with their lengths
        let mut flat: Vec<(Word, CoxElm, Perm, u32)> = Vec::with_capacity(order);
        for (len_idx, level) in levels.iter().enumerate() {
            let l = len_idx as u32;
            for (ce, p) in level {
                // Compute canonical word (lex-min reduced word)
                let word = group.perm_to_word(p);
                debug_assert_eq!(
                    word.len() as u32,
                    l,
                    "perm_length mismatch at BFS level {l}: word={word:?} perm_length={}",
                    group.perm_length(p)
                );
                flat.push((word, ce.clone(), p.clone(), l));
            }
        }

        debug_assert_eq!(
            flat.len(),
            order,
            "BFS produced {} elements but expected {}",
            flat.len(),
            order
        );

        // ---------------------------------------------------------------
        // Phase 2: Canonical sort by (length, word lex)
        // ---------------------------------------------------------------
        flat.sort_by(|a, b| a.3.cmp(&b.3).then_with(|| a.0.cmp(&b.0)));

        // ---------------------------------------------------------------
        // Phase 3: Single-pass extraction into pre-capacitied Vecs
        //
        // Consume `flat` in one pass, eliminating clone-per-Perm and
        // halving peak memory vs. collecting each field separately.
        // ---------------------------------------------------------------
        let mut elms: Vec<Word> = Vec::with_capacity(order);
        let mut coxelms: Vec<CoxElm> = Vec::with_capacity(order);
        let mut perms: Vec<Perm> = Vec::with_capacity(order);
        let mut lengths: Vec<u32> = Vec::with_capacity(order);

        for (word, ce, p, l) in flat {
            elms.push(word);
            coxelms.push(ce);
            perms.push(p);
            lengths.push(l);
        }

        // Build the CoxElm → index map
        let mut index: HashMap<CoxElm, ElmIdx> = HashMap::with_capacity(order);
        for (i, ce) in coxelms.iter().enumerate() {
            index.insert(ce.clone(), i as ElmIdx);
        }

        // ---------------------------------------------------------------
        // Phase 4: inva — index of w^{-1}
        // ---------------------------------------------------------------
        let inva: Vec<ElmIdx> = perms
            .iter()
            .map(|p| {
                let inv_perm = p.inverse();
                let inv_ce = inv_perm.coxelm_sr(&group.simple_root);
                *index.get(&inv_ce).expect("inverse element not in table")
            })
            .collect();

        // ---------------------------------------------------------------
        // Phase 5: aw0 — index of w · w0 (RIGHT multiplication by w0)
        //
        // perm(w · w0) = perm_w.then(longest_perm)
        // CoxElm = images of simple roots under the composition.
        //
        // Invariant: lengths[aw0[i]] == N - lengths[i]
        // ---------------------------------------------------------------
        let aw0: Vec<ElmIdx> = perms
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let np = p.then(longest);
                let ce = np.coxelm_sr(&group.simple_root);
                let idx = *index.get(&ce).expect("w·w0 element not in table");
                debug_assert_eq!(
                    lengths[idx as usize],
                    n as u32 - lengths[i],
                    "aw0 length invariant violated at i={i}: lengths[aw0[{i}]]={}, N-lengths[{i}]={}",
                    lengths[idx as usize],
                    n as u32 - lengths[i]
                );
                idx
            })
            .collect();

        // ---------------------------------------------------------------
        // Phase 6: lft — flat row-major left multiplication by generators
        //
        // lft[i * rank + s] = index of s · w_i
        // perm(s · w_i) = permgens[s].then(&perm_w_i)
        //
        // Invariant: lft(i, s) < i  ⟺  s ∈ left_descents(perm_w_i)
        // ---------------------------------------------------------------
        let mut lft: Vec<ElmIdx> = Vec::with_capacity(order * rank);
        for (i, p) in perms.iter().enumerate() {
            for s in 0..rank {
                let np = group.permgens[s].then(p);
                let ce = np.coxelm_sr(&group.simple_root);
                let idx = *index.get(&ce).expect("s·w element not in table");
                debug_assert!(
                    (idx < i as ElmIdx) == (p.0[group.simple_root[s]] as usize >= n),
                    "lft invariant violated: lft({i},{s})={idx}, left_desc={}",
                    p.0[group.simple_root[s]] as usize >= n
                );
                lft.push(idx);
            }
        }

        ElementTable {
            elms,
            coxelms,
            index,
            lengths,
            inva,
            aw0,
            lft,
            rank,
        }
    }

    /// Compute `L(w_i) = Σ weights[s]` over all generators in the canonical
    /// word of `w_i`, for each element `i`.
    ///
    /// The returned vector has length `self.len()`.
    pub fn lweights(&self, weights: &[u32]) -> Vec<u32> {
        self.elms
            .iter()
            .map(|word| word.iter().map(|&s| weights[s as usize]).sum())
            .collect()
    }

    /// Number of elements in the table.
    #[inline]
    pub fn len(&self) -> usize {
        self.elms.len()
    }

    /// Whether the table is empty (only possible for a trivial group).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elms.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::Gen;

    /// LEFT-multiplication side test: in B2, left-mult of word [1] by gen 0
    /// must equal word [0, 1].
    ///
    /// word_to_perm([1]) has perm P_1.
    /// perm(0 · [1]) = permgens[0].then(P_1).
    /// We expect that to have CoxElm == word_to_perm([0,1]).coxelm_sr(simple_root).
    #[test]
    fn b2_left_mult_side_check() {
        let group = CoxeterGroup::from_type("B2").unwrap();
        let sr = &group.simple_root;
        let p1 = group.word_to_perm(&[1]);
        let p01 = group.word_to_perm(&[0, 1]);
        // Left-multiply perm of [1] by generator 0: permgens[0].then(p1)
        let left_result = group.permgens[0].then(&p1);
        assert_eq!(
            left_result.coxelm_sr(sr),
            p01.coxelm_sr(sr),
            "permgens[0].then(perm([1])) should equal perm([0,1])"
        );
        // Sanity: the other side should NOT equal p01
        let right_result = p1.then(&group.permgens[0]);
        // [1].then([0]) corresponds to word [1,0], not [0,1]
        let p10 = group.word_to_perm(&[1, 0]);
        assert_eq!(
            right_result.coxelm_sr(sr),
            p10.coxelm_sr(sr),
            "perm([1]).then(permgens[0]) should equal perm([1,0])"
        );
    }

    /// Verify the full B2 element table.
    #[test]
    fn b2_element_table() {
        let group = CoxeterGroup::from_type("B2").unwrap();
        let table = ElementTable::build(&group);

        // B2 has 8 elements
        assert_eq!(table.len(), 8);

        // Canonical element list (matches kl_B2_w1.json "elms")
        let expected_elms: Vec<Vec<u8>> = vec![
            vec![],
            vec![0],
            vec![1],
            vec![0, 1],
            vec![1, 0],
            vec![0, 1, 0],
            vec![1, 0, 1],
            vec![0, 1, 0, 1],
        ];
        assert_eq!(table.elms, expected_elms, "B2 canonical element list");

        // Lengths
        let expected_lengths = vec![0u32, 1, 1, 2, 2, 3, 3, 4];
        assert_eq!(table.lengths, expected_lengths);

        // inva of [0,1] (index 3) should be [1,0] (index 4)
        assert_eq!(
            table.inva[3], 4,
            "inva of [0,1] (idx 3) should be [1,0] (idx 4)"
        );
        assert_eq!(
            table.inva[4], 3,
            "inva of [1,0] (idx 4) should be [0,1] (idx 3)"
        );

        // inva∘inva == id
        for i in 0..8 {
            assert_eq!(
                table.inva[table.inva[i] as usize] as usize, i,
                "inva∘inva != id at i={i}"
            );
        }

        // aw0 length mirror: lengths[aw0[i]] == N - lengths[i]  (N=4 for B2)
        let n = group.n_pos;
        for i in 0..8 {
            assert_eq!(
                table.lengths[table.aw0[i] as usize],
                n - table.lengths[i],
                "aw0 length mirror violated at i={i}"
            );
        }

        // lft invariant: lft(i, s) < i  ⟺  s ∈ left_descents(perm_i)
        for i in 0..8 {
            let perm = group.word_to_perm(&table.elms[i]);
            let left_desc = group.left_descents(&perm);
            for s in 0..group.rank {
                let lft_idx = table.lft(i as ElmIdx, s);
                let is_left_desc = left_desc.contains(&(s as Gen));
                assert_eq!(
                    lft_idx < i as u32,
                    is_left_desc,
                    "lft invariant violated at i={i}, s={s}: lft={lft_idx}, is_left_desc={is_left_desc}"
                );
            }
        }
    }

    /// Test `lweights` on B2 with equal weights (all 1).
    #[test]
    fn b2_lweights_equal() {
        let group = CoxeterGroup::from_type("B2").unwrap();
        let table = ElementTable::build(&group);
        let weights = vec![1u32; group.rank];
        let lw = table.lweights(&weights);
        // lweights should equal lengths for all-1 weights
        assert_eq!(lw, table.lengths);
    }

    /// Verify A1 (trivial: 2 elements [[], [0]]).
    #[test]
    fn a1_element_table() {
        let group = CoxeterGroup::from_type("A1").unwrap();
        let table = ElementTable::build(&group);
        assert_eq!(table.len(), 2);
        assert_eq!(table.elms[0], Vec::<u8>::new());
        assert_eq!(table.elms[1], vec![0u8]);
        assert_eq!(table.inva[0], 0); // id^{-1} = id
        assert_eq!(table.inva[1], 1); // s0 is an involution
        assert_eq!(table.aw0[0], 1); // id · w0 = w0
        assert_eq!(table.aw0[1], 0); // w0 · w0 = id
    }

    /// Verify A3: table size 24, inva∘inva = id, lft invariant, aw0 mirror.
    /// Also checks: aw0[aw0[i]] == i (involution assertion).
    #[test]
    fn a3_invariants() {
        let group = CoxeterGroup::from_type("A3").unwrap();
        let table = ElementTable::build(&group);
        let n_total = group.order as usize;
        let n = group.n_pos;

        assert_eq!(table.len(), n_total);

        // inva∘inva == id
        for i in 0..n_total {
            assert_eq!(
                table.inva[table.inva[i] as usize] as usize, i,
                "A3 inva∘inva != id at i={i}"
            );
        }

        // aw0[aw0[i]] == i  (right multiplication by w0 is an involution)
        for i in 0..n_total {
            assert_eq!(
                table.aw0[table.aw0[i] as usize] as usize, i,
                "A3 aw0∘aw0 != id at i={i}"
            );
        }

        // aw0 length mirror
        for i in 0..n_total {
            assert_eq!(
                table.lengths[table.aw0[i] as usize],
                n - table.lengths[i],
                "A3 aw0 length mirror violated at i={i}"
            );
        }

        // lft invariant
        for i in 0..n_total {
            let perm = group.word_to_perm(&table.elms[i]);
            let left_desc = group.left_descents(&perm);
            for s in 0..group.rank {
                let lft_idx = table.lft(i as ElmIdx, s);
                let is_left_desc = left_desc.contains(&(s as Gen));
                assert_eq!(
                    lft_idx < i as u32,
                    is_left_desc,
                    "A3 lft invariant violated at i={i}, s={s}"
                );
            }
        }
    }

    /// Reducible group A2 × A1 (order 12).
    ///
    /// Poincaré polynomial: (1+v)(1+v+v²)(1+v) = (1+v)²(1+v+v²).
    /// Expand: (1+2v+v²)(1+v+v²) = 1+3v+4v²+3v³+v⁴.
    /// Length histogram: [1, 3, 4, 3, 1].
    ///
    /// Checks: order 12, histogram, inva involution, aw0 involution, lft invariant.
    #[test]
    fn a2xa1_table() {
        use crate::cartan::Series;

        // Build A2 × A1 via from_components: A2 (rank 2) + A1 (rank 1).
        let group = CoxeterGroup::from_components(&[(Series::A, 2), (Series::A, 1)]).unwrap();
        let table = ElementTable::build(&group);

        // Order must be 12
        assert_eq!(table.len(), 12, "A2×A1 order should be 12");

        // Length histogram: [1, 3, 4, 3, 1]
        let max_len = *table.lengths.iter().max().unwrap_or(&0) as usize;
        let mut hist = vec![0usize; max_len + 1];
        for &l in &table.lengths {
            hist[l as usize] += 1;
        }
        assert_eq!(
            hist,
            vec![1, 3, 4, 3, 1],
            "A2×A1 length histogram should be [1,3,4,3,1]"
        );

        let n_total = table.len();

        // inva∘inva == id
        for i in 0..n_total {
            assert_eq!(
                table.inva[table.inva[i] as usize] as usize, i,
                "A2×A1 inva∘inva != id at i={i}"
            );
        }

        // aw0∘aw0 == id
        for i in 0..n_total {
            assert_eq!(
                table.aw0[table.aw0[i] as usize] as usize, i,
                "A2×A1 aw0∘aw0 != id at i={i}"
            );
        }

        // lft invariant
        for i in 0..n_total {
            let perm = group.word_to_perm(&table.elms[i]);
            let left_desc = group.left_descents(&perm);
            for s in 0..group.rank {
                let lft_idx = table.lft(i as ElmIdx, s);
                let is_left_desc = left_desc.contains(&(s as Gen));
                assert_eq!(
                    lft_idx < i as u32,
                    is_left_desc,
                    "A2×A1 lft invariant violated at i={i}, s={s}: lft={lft_idx}"
                );
            }
        }

        // aw0 length mirror
        let n_pos = group.n_pos;
        for i in 0..n_total {
            assert_eq!(
                table.lengths[table.aw0[i] as usize],
                n_pos - table.lengths[i],
                "A2×A1 aw0 length mirror violated at i={i}"
            );
        }
    }
}
