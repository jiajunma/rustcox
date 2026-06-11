//! Coxeter group construction and element calculus.
//!
//! This module builds a finite Coxeter group from a type string or a list of
//! irreducible components, and provides the fundamental element-calculus
//! operations on permutations of the root set.
//!
//! # Composition convention
//!
//! **`then(p, q)[i] = q[p[i]]`** — "apply p first, then q".  This matches
//! PyCox's `permmult(p, q)`.  All operations in this module are documented
//! with which side they multiply on.
//!
//! # Key operations
//!
//! | Operation | PyCox fn | Composition side |
//! |-----------|----------|-----------------|
//! | `word_to_perm(w)` | `wordtoperm` | left-fold: acc = then(acc, P_si) for si in w |
//! | `perm_to_word(p)` | `permtoword` | strips smallest left descent; sets p ← then(P_s, p) |
//! | `perm_length(p)` | `permlength` | #{i < N : p\[i\] ≥ N} |
//! | `left_descents(p)` | `leftdescentsetperm` | {s : p\[s\] ≥ N} |
//! | `right_descents(p)` | `rightdescentsetperm` | left descents of p⁻¹ |
//! | `longest_perm()` | `longestperm` | left-multiply by s until all s are descents |

use std::sync::OnceLock;

use crate::{
    cartan::{
        cartan_mat, coxeter_mat_from_cartan, degrees_of, order_from_degrees, parse_type, Error,
        Series,
    },
    element::{Gen, Perm, Word},
    roots,
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One irreducible factor of a (possibly reducible) Coxeter group.
#[derive(Debug)]
pub struct TypeComponent {
    /// Dynkin series of this factor.
    pub series: Series,
    /// The generator indices (into the full group's generator list) belonging
    /// to this factor, in the order 0, 1, …, rank_k − 1 within the factor.
    pub indices: Vec<usize>,
}

/// A finite Coxeter group.
///
/// All permutations in `permgens` act on a root set of size `2 * n_pos`.
///
/// ## Invariant: simple root indexing
///
/// `simple_root[s]` gives the global root index of the s-th simple root.
/// For irreducible groups this is always `s` (simple roots are roots 0..rank).
/// For reducible groups it may differ when component n_pos > component rank.
/// All descent-check functions use this map so they work correctly for
/// both irreducible and reducible groups.
#[derive(Debug)]
pub struct CoxeterGroup {
    /// Number of simple generators (= total rank).
    pub rank: usize,
    /// Total number of positive roots (sum over all components).
    pub n_pos: u32,
    /// Group order |W|.
    pub order: u128,
    /// Reflection degrees, sorted ascending, concatenated over components.
    pub degrees: Vec<u32>,
    /// Full rank×rank Coxeter matrix.  Diagonal entries are 1; off-diagonal
    /// entries between generators from different components are 2.
    pub coxmat: Vec<Vec<u32>>,
    /// Generator permutations on the full 2N root set.
    pub permgens: Vec<Perm>,
    /// Irreducible factors.
    pub components: Vec<TypeComponent>,
    /// Integer coordinate vectors for all 2N roots.
    /// `Some` iff every component is crystallographic (non-Golden).
    pub roots_int: Option<Vec<Vec<i64>>>,
    /// `simple_root[s]` = global positive-root index of the s-th simple root.
    ///
    /// For irreducible groups always `simple_root[s] == s`.  For reducible
    /// groups with unequal component sizes (e.g. A2×A1) the values differ
    /// for generators beyond the first component.
    pub simple_root: Vec<usize>,
    /// Cached longest element (computed lazily).
    longest: OnceLock<Perm>,
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl CoxeterGroup {
    /// Build a Coxeter group from a type string (e.g. `"B2"`, `"A2xA1"`).
    pub fn from_type(s: &str) -> Result<Self, Error> {
        Self::from_components(&parse_type(s)?)
    }

    /// Build a Coxeter group from a slice of `(series, rank)` pairs.
    ///
    /// For a product group the positive roots of component k occupy a
    /// contiguous block `[offset_k, offset_k + N_k)` in the global root list.
    /// Generator indices are assigned consecutively: component 0 gets generators
    /// 0..rank_0, component 1 gets rank_0..rank_0+rank_1, etc.
    /// Build the trivial rank-0 Coxeter group (the only element is the
    /// identity).
    ///
    /// This is the base of the `klcells` parabolic recursion: dropping the last
    /// generator of a rank-1 group yields the empty parabolic `W_∅`.  PyCox's
    /// `reflectionsubgroup(W, [])` builds exactly this group.  All
    /// element-calculus methods degenerate sensibly: the identity perm is empty,
    /// `word_to_perm(&[])` is the empty perm, and `coxelm_sr(&[])` is the empty
    /// `CoxElm`.
    pub fn rank_zero() -> Self {
        CoxeterGroup {
            rank: 0,
            n_pos: 0,
            order: 1,
            degrees: Vec::new(),
            coxmat: Vec::new(),
            permgens: Vec::new(),
            components: Vec::new(),
            roots_int: Some(Vec::new()),
            simple_root: Vec::new(),
            longest: OnceLock::new(),
        }
    }

    pub fn from_components(comps: &[(Series, usize)]) -> Result<Self, Error> {
        if comps.is_empty() {
            return Err(Error::ParseError(
                String::new(),
                "empty component list".to_string(),
            ));
        }

        // Gen is u8; generators are indexed 0..rank, so rank must fit in u8.
        let total_rank: usize = comps.iter().map(|&(_, r)| r).sum();
        if total_rank > 255 {
            return Err(Error::RankOutOfRange {
                series: "product".to_string(),
                rank: total_rank,
            });
        }

        // Build each component's root system.
        struct CompData {
            series: Series,
            rank: usize,
            n_pos: u32,
            permgens: Vec<Perm>, // local (2*n_pos-length perms)
            degrees: Vec<u32>,
            coxmat: Vec<Vec<u32>>,
            roots_int: Option<Vec<Vec<i64>>>,
        }

        let mut comp_data: Vec<CompData> = Vec::with_capacity(comps.len());
        for &(series, rank) in comps {
            let cmat = cartan_mat(series, rank)?;
            let coxmat = coxeter_mat_from_cartan(&cmat);
            let rs = roots::build(&cmat);
            let mut degs = degrees_of(series, rank)?;
            degs.sort_unstable();
            comp_data.push(CompData {
                series,
                rank,
                n_pos: rs.n_pos,
                permgens: rs.permgens,
                degrees: degs,
                coxmat,
                roots_int: rs.roots_int,
            });
        }

        // Total N (positive roots).
        let n_pos: u32 = comp_data.iter().map(|c| c.n_pos).sum();
        let n2 = 2 * n_pos as usize;
        let rank: usize = comp_data.iter().map(|c| c.rank).sum();

        // Build global permgens by embedding each component's local permgen
        // into the global 2N root space.
        //
        // Global root indexing:
        //   positives: [comp0_pos | comp1_pos | ...]  indices 0..N
        //   negatives: [comp0_neg | comp1_neg | ...]  indices N..2N
        //
        // For component k with local offset_k (in positive roots):
        //   global positive root j of comp k → global index offset_k + j
        //   global negative root j of comp k → global index N + offset_k + j
        //
        // A local perm entry local_perm[i] for i < local_n_pos maps to
        // global root offset_k + local_perm[i] (if local_perm[i] < local_n_pos)
        // or N + offset_k + (local_perm[i] - local_n_pos) (if negative).

        // Compute per-component offsets in the global positive root block.
        let mut pos_offsets: Vec<u32> = Vec::with_capacity(comp_data.len());
        let mut running = 0u32;
        for cd in &comp_data {
            pos_offsets.push(running);
            running += cd.n_pos;
        }

        let mut permgens: Vec<Perm> = Vec::with_capacity(rank);
        for (k, cd) in comp_data.iter().enumerate() {
            let off = pos_offsets[k];
            let local_n = cd.n_pos as usize;
            for local_perm in &cd.permgens {
                // Build a global perm that acts as local_perm on component k's
                // roots and as identity on all other components' roots.
                let mut global: Vec<u32> = (0..n2 as u32).collect(); // identity
                                                                     // Map each local root index to a global root index.
                let local_to_global = |local_idx: usize| -> usize {
                    if local_idx < local_n {
                        // positive root of component k
                        off as usize + local_idx
                    } else {
                        // negative root of component k
                        n_pos as usize + off as usize + (local_idx - local_n)
                    }
                };
                for i in 0..(2 * local_n) {
                    let global_i = local_to_global(i);
                    let local_image = local_perm.0[i] as usize;
                    let global_image = local_to_global(local_image);
                    global[global_i] = global_image as u32;
                }
                permgens.push(Perm(global.into_boxed_slice()));
            }
        }

        // Build full rank×rank Coxeter matrix.
        // - Within component k: use that component's coxmat (offset by gen_offset_k).
        // - Between generators of different components: 2.
        let mut gen_offsets: Vec<usize> = Vec::with_capacity(comp_data.len());
        let mut gen_running = 0usize;
        for cd in &comp_data {
            gen_offsets.push(gen_running);
            gen_running += cd.rank;
        }

        let mut coxmat: Vec<Vec<u32>> = vec![vec![2u32; rank]; rank];
        // Diagonal must be 1 (m_ss = 1 for all s).
        for (s, row) in coxmat.iter_mut().enumerate() {
            row[s] = 1;
        }
        for (k, cd) in comp_data.iter().enumerate() {
            let go = gen_offsets[k];
            for s in 0..cd.rank {
                for t in 0..cd.rank {
                    coxmat[go + s][go + t] = cd.coxmat[s][t];
                }
            }
        }

        // Degrees: sorted ascending over all components concatenated.
        let mut degrees: Vec<u32> = comp_data
            .iter()
            .flat_map(|cd| cd.degrees.iter().copied())
            .collect();
        degrees.sort_unstable();

        let order = order_from_degrees(&degrees);

        // Roots int: Some iff all components are crystallographic.
        let all_crystallographic = comp_data.iter().all(|cd| cd.roots_int.is_some());
        let roots_int = if all_crystallographic {
            // Build block-diagonal coordinate vectors.
            // Global positive roots: comp0 positives, comp1 positives, ...
            // Global negative roots: comp0 negatives, comp1 negatives, ...
            //
            // Each component k has coordinate dimension rank_k.  Global coords
            // for component k are embedded in a vector of length `rank` with
            // zeros in positions outside [gen_offset_k, gen_offset_k+rank_k).
            let total_rank = rank;
            let mut all_roots: Vec<Vec<i64>> = Vec::with_capacity(n2);
            // First pass: positives in global order.
            for (k, cd) in comp_data.iter().enumerate() {
                let go = gen_offsets[k];
                let local_roots = cd.roots_int.as_ref().unwrap();
                let local_n = cd.n_pos as usize;
                for local_coord in &local_roots[..local_n] {
                    let mut global_coord = vec![0i64; total_rank];
                    for (j, &v) in local_coord.iter().enumerate() {
                        global_coord[go + j] = v;
                    }
                    all_roots.push(global_coord);
                }
            }
            // Second pass: negatives in global order.
            for (k, cd) in comp_data.iter().enumerate() {
                let go = gen_offsets[k];
                let local_roots = cd.roots_int.as_ref().unwrap();
                let local_n = cd.n_pos as usize;
                for local_coord in &local_roots[local_n..] {
                    let mut global_coord = vec![0i64; total_rank];
                    for (j, &v) in local_coord.iter().enumerate() {
                        global_coord[go + j] = v;
                    }
                    all_roots.push(global_coord);
                }
            }
            Some(all_roots)
        } else {
            None
        };

        // Build TypeComponent list.
        let components: Vec<TypeComponent> = comp_data
            .iter()
            .enumerate()
            .map(|(k, cd)| TypeComponent {
                series: cd.series,
                indices: (gen_offsets[k]..(gen_offsets[k] + cd.rank)).collect(),
            })
            .collect();

        // Build simple_root: simple_root[gen_offset_k + s] = pos_offset_k + s
        // for s in 0..rank_k, for each component k.
        // For irreducible groups this is always simple_root[s] = s.
        let mut simple_root: Vec<usize> = vec![0; rank];
        for (k, cd) in comp_data.iter().enumerate() {
            let go = gen_offsets[k];
            let po = pos_offsets[k] as usize;
            for s in 0..cd.rank {
                simple_root[go + s] = po + s;
            }
        }

        Ok(CoxeterGroup {
            rank,
            n_pos,
            order,
            degrees,
            coxmat,
            permgens,
            components,
            roots_int,
            simple_root,
            longest: OnceLock::new(),
        })
    }

    // -------------------------------------------------------------------------
    // Element calculus
    // -------------------------------------------------------------------------

    /// Return the identity permutation (length 2N).
    ///
    /// Composition note: `identity.then(p) == p` and `p.then(identity) == p`.
    #[inline]
    pub fn id_perm(&self) -> Perm {
        Perm::identity(2 * self.n_pos as usize)
    }

    /// Convert a word to its permutation.
    ///
    /// Replicates PyCox `wordtoperm`:
    /// `reduce(permmult, [id, P_s1, ..., P_sk])`
    /// i.e. acc = then(acc, P_si) for each si in word order.
    ///
    /// Result is the product P_sk ∘ … ∘ P_s1 as functions (right-to-left).
    pub fn word_to_perm(&self, w: &[Gen]) -> Perm {
        let mut acc = self.id_perm();
        for &s in w {
            acc = acc.then(&self.permgens[s as usize]);
        }
        acc
    }

    /// Convert a permutation to its canonical reduced word.
    ///
    /// Replicates PyCox `permtoword`: repeatedly find the smallest `s` with
    /// `p[simple_root[s]] >= N`, set `p ← then(P_s, p)` (left-multiply by s),
    /// and append s.  This strips one left descent at a time, yielding the
    /// lex-smallest reduced word.
    ///
    /// Uses `simple_root[s]` so this works correctly for both irreducible and
    /// reducible groups.
    ///
    /// Composition note: `then(P_s, p)[i] = p[P_s[i]]`, which corresponds to
    /// the PyCox expression `[p[i] for i in permgens[s]]`.
    pub fn perm_to_word(&self, p: &Perm) -> Word {
        let n = self.n_pos as usize;
        let mut cur = p.clone();
        let mut word = Vec::new();
        loop {
            // Find smallest s with cur[simple_root[s]] >= n  (s is a left descent).
            let s = (0..self.rank).find(|&s| cur.0[self.simple_root[s]] as usize >= n);
            match s {
                None => break,
                Some(s) => {
                    // Left-multiply by generator s:
                    // new_cur[i] = cur[permgens[s][i]]
                    // = then(permgens[s], cur)
                    cur = self.permgens[s].then(&cur);
                    // Safe: from_components guards rank ≤ 255 so s fits in Gen (u8).
                    word.push(s as Gen);
                }
            }
        }
        word
    }

    /// Convert a word to a `CoxElm` (images of simple roots under the permutation).
    ///
    /// Replicates PyCox `wordtocoxelm`: computes the full 2N permutation via
    /// `word_to_perm` and extracts the simple-root images via `coxelm_sr`.
    pub fn word_to_coxelm(&self, w: &[Gen]) -> crate::element::CoxElm {
        self.word_to_perm(w).coxelm_sr(&self.simple_root)
    }

    /// Return the length of a permutation.
    ///
    /// `length = #{i < N : p[i] >= N}` — the number of positive roots that map
    /// to negative roots under this element.
    #[inline]
    pub fn perm_length(&self, p: &Perm) -> u32 {
        let n = self.n_pos as usize;
        p.0.iter().take(n).filter(|&&x| x as usize >= n).count() as u32
    }

    /// Return the left descent set of p.
    ///
    /// `s` is a left descent iff `p[simple_root[s]] >= N`.
    pub fn left_descents(&self, p: &Perm) -> Vec<Gen> {
        let n = self.n_pos as usize;
        (0..self.rank)
            .filter(|&s| p.0[self.simple_root[s]] as usize >= n)
            // Safe: from_components guards rank ≤ 255 so s fits in Gen (u8).
            .map(|s| s as Gen)
            .collect()
    }

    /// Return the right descent set of p.
    ///
    /// `s` is a right descent iff the simple root `simple_root[s]` maps to a
    /// negative root under `p⁻¹`, i.e. the preimage of `simple_root[s]` under
    /// `p` lies in the negative block `[N, 2N)`.
    pub fn right_descents(&self, p: &Perm) -> Vec<Gen> {
        let n = self.n_pos as usize;
        (0..self.rank)
            .filter(|&s| {
                let r = self.simple_root[s];
                (n..2 * n).any(|j| p.0[j] as usize == r)
            })
            // Safe: from_components guards rank ≤ 255 so s fits in Gen (u8).
            .map(|s| s as Gen)
            .collect()
    }

    /// Return a reference to the longest element w₀.
    ///
    /// Computed lazily and cached.  Replicates PyCox `longestperm`:
    /// start from the identity; while ∃ s ∈ rank with p\[simple_root\[s\]\] < N,
    /// set `p ← then(P_s, p)` (left-multiply by s).
    ///
    /// Composition note: `then(P_s, p)[i] = p[P_s[i]]`, matching PyCox
    /// `[p[i] for i in permgens[J[s]]]`.
    pub fn longest_perm(&self) -> &Perm {
        self.longest.get_or_init(|| {
            let n = self.n_pos as usize;
            let rank = self.rank;
            let mut p = self.id_perm();
            loop {
                // Find any s ∈ 0..rank with p[simple_root[s]] < n.
                let s = (0..rank).find(|&s| (p.0[self.simple_root[s]] as usize) < n);
                match s {
                    None => break,
                    Some(s) => {
                        // p ← then(permgens[s], p): new_p[i] = p[permgens[s][i]]
                        p = self.permgens[s].then(&p);
                    }
                }
            }
            p
        })
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b2_element_calculus() {
        let w = CoxeterGroup::from_type("B2").unwrap();
        assert_eq!((w.rank, w.n_pos, w.order), (2, 4, 8));

        // word_to_perm / perm_length
        let p = w.word_to_perm(&[0, 1, 0]);
        assert_eq!(w.perm_length(&p), 3);

        // perm_to_word roundtrip
        assert_eq!(w.perm_to_word(&p), vec![0, 1, 0]);

        // Non-reduced word reduces to canonical form
        assert_eq!(
            w.perm_to_word(&w.word_to_perm(&[1, 0, 1, 1, 0])),
            vec![1],
            "s1·s0·s1·s1·s0 should reduce to s1"
        );

        // Left / right descents of s0·s1·s0
        assert_eq!(w.left_descents(&p), vec![0]);
        assert_eq!(w.right_descents(&p), vec![0]);

        // Longest element
        let w0 = w.longest_perm();
        assert_eq!(w.perm_length(w0), 4);
        assert_eq!(w.perm_to_word(w0), vec![0, 1, 0, 1]);
    }

    #[test]
    fn identity_roundtrip() {
        let w = CoxeterGroup::from_type("A3").unwrap();
        let id = w.id_perm();
        assert_eq!(
            w.perm_to_word(&id),
            vec![] as Vec<Gen>,
            "identity should give empty word"
        );
        assert_eq!(w.perm_length(&id), 0);
        assert_eq!(w.word_to_perm(&[]), id, "empty word should give identity");
    }

    #[test]
    fn a3_inverse_roundtrip() {
        // For a reduced word w = [s1, s2, s3], word_to_perm(w).inverse() should
        // equal word_to_perm(w_reversed).
        let w = CoxeterGroup::from_type("A3").unwrap();
        let word: &[Gen] = &[0, 1, 2, 0, 1]; // some word of length 5
        let p = w.word_to_perm(word);
        let p_inv = p.inverse();

        // The inverse perm should give the reversed word (for a REDUCED word).
        // word_to_perm(rev(w)) = P_w1 ∘ P_w2 ∘ ... = inverse of P_w_k ∘ ... ∘ P_w1
        // Both should have the same length.
        assert_eq!(w.perm_length(&p_inv), w.perm_length(&p));
        // Roundtrip: perm_to_word(p_inv) should reduce correctly.
        let inv_word = w.perm_to_word(&p_inv);
        let p_back = w.word_to_perm(&inv_word);
        assert_eq!(
            p_back, p_inv,
            "perm_to_word/word_to_perm roundtrip for inverse"
        );
    }

    #[test]
    fn a1xa1_product_group() {
        let w = CoxeterGroup::from_components(&[(Series::A, 1), (Series::A, 1)]).unwrap();
        assert_eq!(w.rank, 2);
        assert_eq!(w.n_pos, 2);
        assert_eq!(w.order, 4);
        // Coxeter matrix: [[1,2],[2,1]] (2 between different components)
        assert_eq!(w.coxmat, vec![vec![1, 2], vec![2, 1]]);
        // Longest element: both generators are descents, length = N = 2.
        let w0 = w.longest_perm();
        assert_eq!(w.perm_length(w0), 2);
        assert_eq!(w.perm_to_word(w0), vec![0, 1]);
    }

    #[test]
    fn b2_coxelm() {
        let w = CoxeterGroup::from_type("B2").unwrap();
        let ce = w.word_to_coxelm(&[0, 1]);
        // CoxElm has rank=2 entries.
        assert_eq!(ce.0.len(), 2);
    }

    /// Property test: right_descents(p) == left_descents(p.inverse()) for 50
    /// deterministic words in B3.  Verifies the allocation-free formula
    /// p[N+s] < N against the reference inverse()-based definition.
    #[test]
    fn b3_right_descents_matches_inverse_based() {
        let w = CoxeterGroup::from_type("B3").unwrap();
        // 50 deterministic words built by cycling through generators 0,1,2.
        let words: Vec<Vec<Gen>> = (0u32..50)
            .map(|i| {
                (0..=(i % 8))
                    .map(|j| ((i + j) % w.rank as u32) as Gen)
                    .collect()
            })
            .collect();
        for word in &words {
            let p = w.word_to_perm(word);
            let direct = w.right_descents(&p);
            let via_inv = w.left_descents(&p.inverse());
            assert_eq!(
                direct, via_inv,
                "right_descents mismatch for word {word:?}: direct={direct:?} via_inv={via_inv:?}"
            );
        }
    }

    /// A1×A1: generators 0 and 1 commute (they act on disjoint root systems).
    #[test]
    fn a1xa1_generators_commute() {
        let w = CoxeterGroup::from_components(&[(Series::A, 1), (Series::A, 1)]).unwrap();
        assert_eq!(
            w.word_to_perm(&[0, 1]),
            w.word_to_perm(&[1, 0]),
            "generators of A1×A1 should commute"
        );
    }

    /// Rank guard: a product group whose total rank exceeds 255 must be rejected.
    #[test]
    fn rank_guard_rejects_overflow() {
        // 86 copies of A3 = rank 3×86 = 258 > 255.
        let comps: Vec<(Series, usize)> = vec![(Series::A, 3); 86];
        assert!(
            matches!(
                CoxeterGroup::from_components(&comps),
                Err(Error::RankOutOfRange { .. })
            ),
            "expected RankOutOfRange for total rank 258"
        );
    }
}
