//! Parabolic subgroups, sub-Cartan classification, and distinguished left
//! coset representatives.
//!
//! This module ports two PyCox facilities needed by the cell machinery:
//!
//! - **`reflectionsubgroup(W, J)`** (parabolic path, `pycox_ref.py` ≈3800–3842):
//!   given a subset `J` of simple generators, build the parabolic subgroup
//!   `W_J` as a standalone [`CoxeterGroup`].  PyCox classifies the restricted
//!   Cartan matrix back to named types; we do the same with
//!   [`classify_cartan_sub`].
//! - **`redleftcosetreps(W, J)`** (`pycox_ref.py` ≈3974–4010): the distinguished
//!   left coset representatives of `W_J` in `W` — the minimal-length elements of
//!   the cosets `w·W_J`.  See [`red_left_coset_reps`].
//!
//! # Why classification is needed
//!
//! [`CoxeterGroup`] retains the Coxeter matrix and component list but *not* the
//! Cartan matrix.  To build `W_J` we reconstruct `W`'s full Cartan matrix from
//! its components (each component's block is `cartan_mat(series, rank)`),
//! restrict it to `J`, split it into connected sub-diagrams, and recognise each
//! sub-diagram as a named `(Series, ordering)` by exact matrix comparison
//! against `cartan_mat(series, n)` under candidate orderings.
//!
//! The Cartan-level data matters: `B_n` and `C_n` share a Coxeter matrix but
//! differ in their Cartan matrices (the `−2` sits on opposite sides).  A
//! parabolic of `B_n` whose restricted Cartan carries the `B`-side `−2` must be
//! recognised as `B`, not `C`, so the generated permutation calculus matches the
//! honest restriction.
//!
//! # Generator mapping
//!
//! [`Parabolic`] stores `gen_map`, mapping each *local* `W_J` generator to the
//! *global* `W` generator it represents.  This makes `W_J` words translate to
//! `W` words by `gen_map[s]`, and guarantees
//! `W_J.coxmat[s][t] == W.coxmat[gen_map[s]][gen_map[t]]`.

use std::collections::HashSet;

use thiserror::Error;

use crate::{
    cartan::{cartan_mat, CartanMat, Series},
    element::{CoxElm, Gen, Word},
    group::CoxeterGroup,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by parabolic-subgroup construction.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    /// A generator index in `J` is outside `0..W.rank`.
    #[error("generator {gen} is out of range for a group of rank {rank}")]
    GenOutOfRange { gen: usize, rank: usize },

    /// A connected sub-diagram could not be matched to any named series.
    #[error("could not classify the sub-Cartan on generators {indices:?}")]
    Unclassified { indices: Vec<Gen> },

    /// An error bubbled up from Cartan-data construction.
    #[error(transparent)]
    Cartan(#[from] crate::cartan::Error),
}

// ---------------------------------------------------------------------------
// Cartan entry abstraction
// ---------------------------------------------------------------------------

/// A single Cartan-matrix entry, unified across the three number rings so that
/// restricted sub-matrices can be compared for exact equality.
///
/// Equality normalises across rings: a `Golden`/`Cyc` value that happens to be a
/// plain integer compares equal to the corresponding `Int`.  This matters
/// because a parabolic of `H_n` (golden ring) can have integer-only sub-blocks
/// (e.g. the A-chain left after removing the golden-edge generator), which must
/// match the integer canonical `cartan_mat(A, n)`.
#[derive(Clone, Debug)]
enum CEntry {
    Int(i64),
    Golden(crate::ring::GoldenInt),
    Cyc(crate::ring::CycInt),
}

impl CEntry {
    /// Return the plain integer value if this entry is integral (in any ring).
    fn as_int(&self) -> Option<i64> {
        match self {
            CEntry::Int(x) => Some(*x),
            CEntry::Golden(g) => (g.b == 0).then_some(g.a),
            // A `CycInt` is a plain integer iff its reduced polynomial has at
            // most a constant term (coeffs trimmed, low degree first).
            CEntry::Cyc(c) => match c.coeffs() {
                [] => Some(0),
                [c0] => Some(*c0),
                _ => None,
            },
        }
    }
}

impl PartialEq for CEntry {
    fn eq(&self, other: &Self) -> bool {
        // If both reduce to plain integers, compare integers (cross-ring).
        if let (Some(a), Some(b)) = (self.as_int(), other.as_int()) {
            return a == b;
        }
        match (self, other) {
            (CEntry::Golden(a), CEntry::Golden(b)) => a == b,
            (CEntry::Cyc(a), CEntry::Cyc(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for CEntry {}

/// A Cartan matrix with entries unified to [`CEntry`].
type CMat = Vec<Vec<CEntry>>;

/// Convert a [`CartanMat`] to a uniform [`CMat`].
fn to_cmat(m: &CartanMat) -> CMat {
    match m {
        CartanMat::Int(rows) => rows
            .iter()
            .map(|r| r.iter().map(|&x| CEntry::Int(x)).collect())
            .collect(),
        CartanMat::Golden(rows) => rows
            .iter()
            .map(|r| r.iter().map(|&x| CEntry::Golden(x)).collect())
            .collect(),
        CartanMat::Cyc(rows) => rows
            .iter()
            .map(|r| r.iter().map(|x| CEntry::Cyc(x.clone())).collect())
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Full-Cartan reconstruction
// ---------------------------------------------------------------------------

/// Reconstruct `W`'s full `rank × rank` Cartan matrix from its components.
///
/// Each irreducible component contributes the block `cartan_mat(series, rank_k)`
/// placed at its global generator indices; cross-component entries are `0`.
fn full_cartan(w: &CoxeterGroup) -> Result<CMat, Error> {
    let n = w.rank;
    let mut mat: CMat = vec![vec![CEntry::Int(0); n]; n];
    for comp in &w.components {
        let local = cartan_mat(comp.series, comp.indices.len())?;
        let cm = to_cmat(&local);
        for (li, &gi) in comp.indices.iter().enumerate() {
            for (lj, &gj) in comp.indices.iter().enumerate() {
                mat[gi][gj] = cm[li][lj].clone();
            }
        }
    }
    Ok(mat)
}

/// Restrict a [`CMat`] to the rows/columns indexed by `idx`, in `idx` order.
fn restrict(mat: &CMat, idx: &[usize]) -> CMat {
    idx.iter()
        .map(|&i| idx.iter().map(|&j| mat[i][j].clone()).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// Connected components of a sub-diagram
// ---------------------------------------------------------------------------

/// Split generators `j` into connected sub-diagrams.
///
/// Two generators are adjacent iff their Coxeter-matrix entry is `≥ 3`.  Each
/// returned group is a sorted list of `W`-generator indices.
fn connected_components(w: &CoxeterGroup, j: &[Gen]) -> Vec<Vec<Gen>> {
    let mut remaining: Vec<Gen> = {
        let mut v: Vec<Gen> = j.to_vec();
        v.sort_unstable();
        v.dedup();
        v
    };
    let mut comps: Vec<Vec<Gen>> = Vec::new();
    while let Some(&seed) = remaining.first() {
        // BFS from `seed` within `remaining`.
        let mut comp: Vec<Gen> = vec![seed];
        let mut frontier: Vec<Gen> = vec![seed];
        remaining.retain(|&x| x != seed);
        while let Some(cur) = frontier.pop() {
            let adjacent: Vec<Gen> = remaining
                .iter()
                .copied()
                .filter(|&other| w.coxmat[cur as usize][other as usize] >= 3)
                .collect();
            for a in adjacent {
                comp.push(a);
                frontier.push(a);
                remaining.retain(|&x| x != a);
            }
        }
        comp.sort_unstable();
        comps.push(comp);
    }
    comps
}

// ---------------------------------------------------------------------------
// classify_cartan_sub
// ---------------------------------------------------------------------------

/// Classify the parabolic sub-diagram on generator subset `j`.
///
/// Returns one `(Series, indices)` pair per connected sub-diagram, where
/// `indices` is the list of `W`-generator indices ordered so that the restricted
/// Cartan matrix on `indices` equals `cartan_mat(series, indices.len())`
/// exactly.
///
/// # Approach
///
/// Reconstruct `W`'s full Cartan matrix, split `j` into connected components,
/// and for each component search candidate `(series, ordering)` pairs until the
/// restricted Cartan matches a canonical `cartan_mat`.  Orderings are generated
/// by backtracking that respects the diagram structure (rank ≤ 8, diagrams are
/// paths/trees), so the search is small.
pub fn classify_cartan_sub(w: &CoxeterGroup, j: &[Gen]) -> Result<Vec<(Series, Vec<Gen>)>, Error> {
    for &g in j {
        if g as usize >= w.rank {
            return Err(Error::GenOutOfRange {
                gen: g as usize,
                rank: w.rank,
            });
        }
    }
    let full = full_cartan(w)?;
    let comps = connected_components(w, j);
    let mut out: Vec<(Series, Vec<Gen>)> = Vec::with_capacity(comps.len());
    for comp in comps {
        out.push(classify_component(&full, &comp)?);
    }
    Ok(out)
}

/// Candidate series for a connected sub-diagram of the given rank.
///
/// Order matters only for ties between Coxeter-isomorphic types (`B`/`C`):
/// `B` is tried before `C`, so a `−2` on the `B` side classifies as `B`.
fn candidate_series(rank: usize) -> Vec<Series> {
    match rank {
        1 => vec![Series::A],
        2 => vec![
            Series::A,
            Series::B,
            Series::C,
            Series::G,
            Series::I(5),
            // Other dihedral I(m) handled separately via the Coxeter order.
        ],
        3 => vec![Series::A, Series::B, Series::C, Series::D, Series::H],
        4 => vec![
            Series::A,
            Series::B,
            Series::C,
            Series::D,
            Series::F,
            Series::H,
        ],
        n => {
            let mut v = vec![Series::A, Series::B, Series::C, Series::D];
            if (6..=8).contains(&n) {
                v.push(Series::E);
            }
            v
        }
    }
}

/// Classify a single connected sub-diagram.
fn classify_component(full: &CMat, comp: &[Gen]) -> Result<(Series, Vec<Gen>), Error> {
    let n = comp.len();
    let idx: Vec<usize> = comp.iter().map(|&g| g as usize).collect();
    let sub = restrict(full, &idx);

    // Rank-2 dihedral I(m): if no canonical series matches, derive m from the
    // Coxeter order of the single edge.  Handled inside the candidate loop by
    // also trying I(m) for the actual bond.
    let mut candidates = candidate_series(n);
    if n == 2 {
        if let Some(m) = dihedral_param(&sub) {
            if !matches!(m, 3 | 4 | 6) {
                candidates.push(Series::I(m));
            }
        }
    }

    for series in candidates {
        let canon = match cartan_mat(series, n) {
            Ok(c) => to_cmat(&c),
            Err(_) => continue,
        };
        if let Some(order) = find_ordering(&sub, &canon) {
            // Map the local positions back to global generator indices.
            let mapped: Vec<Gen> = order.iter().map(|&p| comp[p]).collect();
            return Ok((series, mapped));
        }
    }
    Err(Error::Unclassified {
        indices: comp.to_vec(),
    })
}

/// Derive the dihedral parameter `m` from a rank-2 sub-Cartan, if the edge is a
/// recognised bond; otherwise `None`.
///
/// Uses the product `c[0][1]·c[1][0]` of the off-diagonal entries when integer:
/// `0→2, 1→3, 2→4, 3→6`.  Golden edges (`m = 5`) and higher cyclotomic edges are
/// recognised directly by matrix matching, so this only needs the integer case.
fn dihedral_param(sub: &CMat) -> Option<u32> {
    let (a, b) = match (&sub[0][1], &sub[1][0]) {
        (CEntry::Int(a), CEntry::Int(b)) => (*a, *b),
        _ => return None,
    };
    match a * b {
        0 => Some(2),
        1 => Some(3),
        2 => Some(4),
        3 => Some(6),
        _ => None,
    }
}

/// Find an ordering `p` of `0..n` such that `sub[p[i]][p[j]] == canon[i][j]` for
/// all `i, j`.  Returns the ordering, or `None` if no permutation matches.
///
/// Backtracking with two prunes: diagonal entries must already match (they are
/// equal here), and partial assignments are checked against `canon` as they are
/// extended.
fn find_ordering(sub: &CMat, canon: &CMat) -> Option<Vec<usize>> {
    let n = sub.len();
    let mut assign: Vec<usize> = Vec::with_capacity(n);
    let mut used = vec![false; n];
    backtrack(sub, canon, &mut assign, &mut used).then_some(assign)
}

/// Recursive backtracking helper for [`find_ordering`].
///
/// `assign[i]` (for `i < assign.len()`) is the `sub` index placed at canonical
/// position `i`.  Returns `true` once a full consistent assignment is found,
/// leaving it in `assign`.
fn backtrack(sub: &CMat, canon: &CMat, assign: &mut Vec<usize>, used: &mut [bool]) -> bool {
    let pos = assign.len();
    let n = sub.len();
    if pos == n {
        return true;
    }
    for cand in 0..n {
        if used[cand] {
            continue;
        }
        // Check consistency of placing `cand` at canonical position `pos`
        // against all already-placed positions (both directions plus diagonal).
        if sub[cand][cand] != canon[pos][pos] {
            continue;
        }
        let consistent = (0..pos).all(|k| {
            let prev = assign[k];
            sub[cand][prev] == canon[pos][k] && sub[prev][cand] == canon[k][pos]
        });
        if !consistent {
            continue;
        }
        assign.push(cand);
        used[cand] = true;
        if backtrack(sub, canon, assign, used) {
            return true;
        }
        used[cand] = false;
        assign.pop();
    }
    false
}

// ---------------------------------------------------------------------------
// Parabolic
// ---------------------------------------------------------------------------

/// A parabolic subgroup `W_J` of a Coxeter group `W`.
///
/// `group` is the standalone [`CoxeterGroup`] for `W_J`, built from the
/// classified components.  `sub_j` is the generator subset `J` (global indices).
/// `gen_map[s]` is the global `W` generator that local `W_J` generator `s`
/// represents.
#[derive(Debug)]
pub struct Parabolic {
    /// The parabolic subgroup as a standalone group, `W_J`.
    pub group: CoxeterGroup,
    /// The defining generator subset `J` (global `W`-indices, as supplied).
    pub sub_j: Vec<Gen>,
    /// `gen_map[local W_J generator] = global W generator index`.
    gen_map: Vec<Gen>,
}

impl Parabolic {
    /// Build the parabolic subgroup `W_J` for generator subset `j`.
    ///
    /// The empty subset `j = []` yields the trivial rank-0 parabolic `W_∅`
    /// (matching PyCox `reflectionsubgroup(W, [])`), the base of the `klcells`
    /// recursion.
    pub fn new(w: &CoxeterGroup, j: &[Gen]) -> Result<Self, Error> {
        if j.is_empty() {
            return Ok(Parabolic {
                group: CoxeterGroup::rank_zero(),
                sub_j: Vec::new(),
                gen_map: Vec::new(),
            });
        }
        let classified = classify_cartan_sub(w, j)?;

        // `from_components` concatenates components in the order given and
        // relabels generators 0,1,2,… consecutively.  Build `gen_map` so that
        // local generator k corresponds to the right global W generator.
        let comps: Vec<(Series, usize)> =
            classified.iter().map(|(s, idx)| (*s, idx.len())).collect();
        let gen_map: Vec<Gen> = classified
            .iter()
            .flat_map(|(_, idx)| idx.iter().copied())
            .collect();

        let group = CoxeterGroup::from_components(&comps)?;

        Ok(Parabolic {
            group,
            sub_j: j.to_vec(),
            gen_map,
        })
    }

    /// Map a `W_J` word to the corresponding `W` word via `gen_map`.
    pub fn word_to_w(&self, w1_word: &[Gen]) -> Word {
        w1_word.iter().map(|&s| self.gen_map[s as usize]).collect()
    }

    /// The local-to-global generator map.  `gen_map()[s]` is the global `W`
    /// generator that local `W_J` generator `s` represents.
    pub fn gen_map(&self) -> &[Gen] {
        &self.gen_map
    }
}

// ---------------------------------------------------------------------------
// red_left_coset_reps
// ---------------------------------------------------------------------------

/// Distinguished left coset representatives of `W_J` in `W`.
///
/// Ports PyCox `redleftcosetreps` (`pycox_ref.py` ≈3974–4010).  An element `w`
/// is a representative iff it has minimal length in `w·W_J`, equivalently
/// `l(ws) = l(w)+1` for every `s ∈ J` (no `J`-generator is a right descent).
///
/// # Algorithm
///
/// BFS over coxelms from the identity by **left** multiplication.  For each
/// element `w` of the current layer and each simple generator `s` that is not a
/// left descent of `w` (so `s·w` is longer), the candidate `s·w` is accepted
/// unless it coincides with `w·u` for some `u ∈ J` — the exact PyCox acceptance
/// test, which (together with the layer dedup) keeps only minimal-length reps.
///
/// Returns canonical reduced words, sorted by `(length, lex)`.  For groups built
/// via [`CoxeterGroup::from_type`] this needs only the reps and their frontier —
/// it never enumerates `W`, so it is cheap even for `E7`.
pub fn red_left_coset_reps(w: &CoxeterGroup, j: &[Gen]) -> Vec<Word> {
    let n = w.n_pos as usize;
    let sr = &w.simple_root;

    // Identity perm and its coxelm seed the BFS.
    let id = w.id_perm();
    let id_ce = id.coxelm_sr(sr);

    // We track full perms (for descent checks and multiplications) keyed by
    // their coxelm for dedup.
    let mut all_coxelms: HashSet<CoxElm> = HashSet::new();
    all_coxelms.insert(id_ce.clone());

    let mut reps_perms: Vec<crate::element::Perm> = vec![id];
    let mut frontier: Vec<crate::element::Perm> = vec![reps_perms[0].clone()];

    while !frontier.is_empty() {
        let mut next: Vec<crate::element::Perm> = Vec::new();
        for wp in &frontier {
            for s in 0..w.rank {
                // `s` is a left descent of `wp` iff wp[simple_root[s]] >= N.
                if wp.0[sr[s]] as usize >= n {
                    continue; // left descent: s·w would be shorter — skip.
                }
                // nw = s · wp (left multiply): then(permgens[s], wp).
                let nw = w.permgens[s].then(wp);
                let nw_ce = nw.coxelm_sr(sr);

                // Acceptance test: reject if nw == coxelm(wp · u) for some u ∈ J.
                let rejected = j.iter().any(|&u| {
                    let wu = wp.then(&w.permgens[u as usize]);
                    wu.coxelm_sr(sr) == nw_ce
                });
                if rejected {
                    continue;
                }
                // Dedup across all discovered reps.
                if all_coxelms.insert(nw_ce) {
                    next.push(nw.clone());
                }
            }
        }
        for p in &next {
            reps_perms.push(p.clone());
        }
        frontier = next;
    }

    // Convert to canonical words and sort by (length, lex).
    let mut words: Vec<Word> = reps_perms.iter().map(|p| w.perm_to_word(p)).collect();
    words.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    words
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enumerate::ElementTable;

    /// Hand-computed |W| / |W_J| for every single-generator removal, to assert
    /// `|reps| * |W_J| == |W|` and that `Parabolic::new` succeeds.
    #[test]
    fn coset_counts() {
        let types = ["A4", "B4", "D4", "F4", "H4", "E6"];
        for t in types {
            let w = CoxeterGroup::from_type(t).unwrap();
            for rem in 0..w.rank {
                let j: Vec<Gen> = (0..w.rank as Gen).filter(|&s| s as usize != rem).collect();
                let par = Parabolic::new(&w, &j)
                    .unwrap_or_else(|e| panic!("{t} remove {rem}: Parabolic::new failed: {e}"));
                let reps = red_left_coset_reps(&w, &j);
                assert_eq!(
                    reps.len() as u128 * par.group.order,
                    w.order,
                    "{t} remove {rem}: |reps|({}) * |W_J|({}) != |W|({})",
                    reps.len(),
                    par.group.order,
                    w.order
                );
            }
        }
    }

    /// E7: removing the last generator (index 6) yields E6 with 56 coset reps;
    /// removing generator 0 yields D6 with 126 reps.  Verified against PyCox
    /// (the oracle); the original plan prose had the generators swapped.
    ///
    /// Uses `from_type("E7")` which builds roots/permgens only — no element
    /// table — so the BFS stays cheap.
    #[test]
    fn e7_parabolics_without_enumeration() {
        let w = CoxeterGroup::from_type("E7").unwrap();

        // Remove last generator (6) → E6, 56 reps.
        let j6: Vec<Gen> = (0..6).collect();
        let par6 = Parabolic::new(&w, &j6).unwrap();
        let reps6 = red_left_coset_reps(&w, &j6);
        assert_eq!(reps6.len(), 56, "E7 remove gen 6: |X1| should be 56");
        assert_eq!(
            par6.group.components.len(),
            1,
            "E7 remove gen 6: W_J should be irreducible"
        );
        assert_eq!(
            par6.group.components[0].series,
            Series::E,
            "E7 remove gen 6: W_J should be type E (E6)"
        );
        assert_eq!(
            par6.group.rank, 6,
            "E7 remove gen 6: W_J should have rank 6"
        );

        // Remove generator 0 → D6, 126 reps.
        let j0: Vec<Gen> = (1..7).collect();
        let par0 = Parabolic::new(&w, &j0).unwrap();
        let reps0 = red_left_coset_reps(&w, &j0);
        assert_eq!(reps0.len(), 126, "E7 remove gen 0: |X1| should be 126");
        assert_eq!(
            par0.group.components[0].series,
            Series::D,
            "E7 remove gen 0: W_J should be type D (D6)"
        );
        assert_eq!(par0.group.rank, 6);
    }

    /// `gen_map` consistency: for all local generators s,t,
    /// `W_J.coxmat[s][t] == W.coxmat[gen_map[s]][gen_map[t]]`.
    #[test]
    fn coxmat_compat() {
        let types = ["A4", "B4", "D4", "F4", "H4", "E6"];
        for t in types {
            let w = CoxeterGroup::from_type(t).unwrap();
            for rem in 0..w.rank {
                let j: Vec<Gen> = (0..w.rank as Gen).filter(|&s| s as usize != rem).collect();
                let par = Parabolic::new(&w, &j).unwrap();
                let gm = par.gen_map();
                let r = par.group.rank;
                for s in 0..r {
                    for tt in 0..r {
                        assert_eq!(
                            par.group.coxmat[s][tt], w.coxmat[gm[s] as usize][gm[tt] as usize],
                            "{t} remove {rem}: coxmat mismatch at local ({s},{tt})"
                        );
                    }
                }
            }
        }
    }

    /// B-vs-C distinction.  In B4 the `−2` Cartan entry sits on the `B` side of
    /// the 0–1 bond.  Removing the LAST generator keeps {0,1,2} with that bond,
    /// which must classify as B3 (not C3).  Removing generator 0 leaves the
    /// straight A3 chain {1,2,3}.
    #[test]
    fn b_vs_c_distinction() {
        let w = CoxeterGroup::from_type("B4").unwrap();

        // Remove last generator (3): {0,1,2} → B3.
        let j_last: Vec<Gen> = vec![0, 1, 2];
        let cls = classify_cartan_sub(&w, &j_last).unwrap();
        assert_eq!(cls.len(), 1, "B4 remove 3 should be irreducible");
        assert_eq!(cls[0].0, Series::B, "B4 remove 3 must be B3, not C3");
        assert_eq!(cls[0].1.len(), 3);

        // Remove generator 0: {1,2,3} → A3.
        let j0: Vec<Gen> = vec![1, 2, 3];
        let cls0 = classify_cartan_sub(&w, &j0).unwrap();
        assert_eq!(cls0.len(), 1);
        assert_eq!(cls0[0].0, Series::A, "B4 remove 0 must be A3");
    }

    /// `red_left_coset_reps` returns exactly the length-minimal coset reps.
    ///
    /// Brute-force oracle from the full element table: `x` is a rep iff no
    /// `j ∈ J` is a RIGHT descent of `x` (`l(xj) > l(x)` for all `j ∈ J`).
    /// Checked on A3 (J={0,1}) and B3 (J={0,1}).
    #[test]
    fn reps_are_minimal() {
        for t in ["A3", "B3"] {
            let w = CoxeterGroup::from_type(t).unwrap();
            let j: Vec<Gen> = vec![0, 1];
            let table = ElementTable::build(&w);

            // Oracle: filter table elements with no right descent in J.
            let jset: HashSet<Gen> = j.iter().copied().collect();
            let mut oracle: Vec<Word> = table
                .elms
                .iter()
                .filter(|word| {
                    let p = w.word_to_perm(word);
                    let rd = w.right_descents(&p);
                    !rd.iter().any(|s| jset.contains(s))
                })
                .cloned()
                .collect();
            oracle.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));

            let reps = red_left_coset_reps(&w, &j);
            assert_eq!(reps, oracle, "{t} J={{0,1}}: reps differ from oracle");
        }

        // Spot-check the explicit A3 J={0,1} answer from PyCox.
        let w = CoxeterGroup::from_type("A3").unwrap();
        let reps = red_left_coset_reps(&w, &[0, 1]);
        let expected: Vec<Word> = vec![vec![], vec![2], vec![1, 2], vec![0, 1, 2]];
        assert_eq!(reps, expected, "A3 J={{0,1}} explicit reps");
    }

    /// H4: removing the last generator yields H3, and there are 14400/120 = 120
    /// coset reps.  Uses `from_type("H4")` (roots only, no element table).
    #[test]
    fn h4_reps_from_h3() {
        let w = CoxeterGroup::from_type("H4").unwrap();
        let j: Vec<Gen> = vec![0, 1, 2];
        let par = Parabolic::new(&w, &j).unwrap();
        assert_eq!(par.group.components[0].series, Series::H);
        assert_eq!(par.group.rank, 3);
        assert_eq!(par.group.order, 120, "W_J should be H3 (order 120)");
        let reps = red_left_coset_reps(&w, &j);
        assert_eq!(reps.len(), 120, "H4 remove 3: |X1| should be 120");
    }

    /// End-to-end embedding check: every `W_J` element, mapped into `W` via
    /// `word_to_w`, must keep its length.  A standard parabolic embedding is an
    /// isometry, so this verifies that classification, the local ordering, and
    /// `gen_map` are jointly correct — a stronger guarantee than `coxmat_compat`.
    #[test]
    fn embedding_is_length_preserving() {
        for t in ["B4", "F4", "H4", "D4"] {
            let w = CoxeterGroup::from_type(t).unwrap();
            for rem in 0..w.rank {
                let j: Vec<Gen> = (0..w.rank as Gen).filter(|&s| s as usize != rem).collect();
                let par = Parabolic::new(&w, &j).unwrap();
                let w1 = &par.group;
                // Enumerate all W_J elements (small) and compare lengths.
                let table = ElementTable::build(w1);
                for word in &table.elms {
                    let l_w1 = w1.perm_length(&w1.word_to_perm(word));
                    let mapped = par.word_to_w(word);
                    let l_w = w.perm_length(&w.word_to_perm(&mapped));
                    assert_eq!(
                        l_w1, l_w,
                        "{t} remove {rem}: length not preserved for W_J word {word:?} -> {mapped:?}"
                    );
                }
            }
        }
    }

    /// `word_to_w` translates W_J words to W words via `gen_map`.
    #[test]
    fn word_to_w_maps_generators() {
        let w = CoxeterGroup::from_type("B4").unwrap();
        // Remove gen 0 → A3 on {1,2,3}; gen_map = [1,2,3].
        let par = Parabolic::new(&w, &[1, 2, 3]).unwrap();
        assert_eq!(par.word_to_w(&[0, 1, 2]), vec![1, 2, 3]);
        assert_eq!(par.word_to_w(&[2, 0]), vec![3, 1]);
    }

    /// Spot-check F4 parabolic classification against PyCox oracle:
    /// - F4 minus gen 0 (sub {1,2,3}): the restricted Cartan matches C3 exactly.
    /// - F4 minus gen 3 (sub {0,1,2}): the restricted Cartan matches B3.
    ///
    /// Verified with `reflectionsubgroup(coxeter("F",4), [1,2,3]).cartantype` → C3
    /// and `reflectionsubgroup(coxeter("F",4), [0,1,2]).cartantype` → B3 (PyCox).
    #[test]
    fn f4_series_spot_check() {
        let w = CoxeterGroup::from_type("F4").unwrap();
        // Remove gen 0 → {1,2,3}: sub-Cartan matches C3.
        let cls1 = classify_cartan_sub(&w, &[1, 2, 3]).unwrap();
        assert_eq!(cls1.len(), 1);
        assert_eq!(cls1[0].0, Series::C, "F4 remove 0 must be C3, not B3");
        // Remove gen 3 → {0,1,2}: sub-Cartan matches B3.
        let cls2 = classify_cartan_sub(&w, &[0, 1, 2]).unwrap();
        assert_eq!(cls2.len(), 1);
        assert_eq!(cls2[0].0, Series::B, "F4 remove 3 must be B3, not C3");
    }
}
