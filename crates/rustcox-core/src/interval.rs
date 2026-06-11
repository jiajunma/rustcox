//! Bruhat-interval machinery and digraph isomorphism for the combinatorial
//! invariance experiment (task Q3).
//!
//! Given a finite Coxeter group and its full KL table, for a comparable pair
//! `y <_B w` we extract the order interval `I = {z : y ≤ z ≤ w}` and build two
//! directed graphs on it:
//!
//! - the **poset of covers** (Hasse diagram of Bruhat order restricted to `I`):
//!   an edge `z1 → z2` whenever `z1 < z2`, `l(z2) = l(z1) + 1`, and `z1 ≤_B z2`.
//!   In a Bruhat interval all covers are length-1 steps.
//! - the **full Bruhat graph**: an edge `z1 → z2` for ALL `z1 < z2` in `I` with
//!   `z2 = t · z1` for a reflection `t` (LEFT-reflection convention: the edge
//!   exists iff `z2 · z1⁻¹` is a reflection and `l(z2) > l(z1)`).  The cover
//!   graph is a subgraph of the Bruhat graph.
//!
//! Vertices carry a **relative length level** `l(z) − l(y)` so that the
//! isomorphism test is length-shift invariant: intervals of the same shape in
//! different length windows are identified.
//!
//! ## Combinatorial invariance conjecture (Lusztig/Dyer)
//!
//! The KL polynomial `P_{y,w}` depends only on the isomorphism type of the
//! Bruhat interval `[y, w]`.  The standard form uses the poset; a graph form
//! uses the Bruhat graph.  This module provides the structures and a canonical
//! classifier so the conjecture can be tested empirically on small groups.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    element::{ElmIdx, Perm},
    group::CoxeterGroup,
    kl::table::KlTable,
};

/// Hard cap on isomorphism-backtracking node expansions.  A pathological
/// blow-up should panic loudly rather than silently spin.
pub const ISO_NODE_CAP: u64 = 10_000_000;

// ---------------------------------------------------------------------------
// Reflections
// ---------------------------------------------------------------------------

/// Compute the set of all reflections of `group` as full permutations.
///
/// A reflection is any conjugate `u · s · u⁻¹` of a simple reflection `s`.
/// We compute the conjugacy closure of `{permgens[s]}` by BFS: starting from
/// the simple reflections, repeatedly conjugate every known reflection `t` by
/// every simple generator `s` (`t ↦ permgens[s] · t · permgens[s]`, since a
/// simple reflection is its own inverse).  The closure has exactly `N`
/// elements where `N = n_pos`, which the caller can assert.
pub fn reflections(group: &CoxeterGroup) -> HashSet<Perm> {
    let mut set: HashSet<Perm> = HashSet::new();
    let mut queue: VecDeque<Perm> = VecDeque::new();

    for s in 0..group.rank {
        let t = group.permgens[s].clone();
        if set.insert(t.clone()) {
            queue.push_back(t);
        }
    }

    while let Some(t) = queue.pop_front() {
        for s in 0..group.rank {
            let g = &group.permgens[s];
            // conj = g · t · g  (g is an involution so g⁻¹ = g).
            // With `then(p, q)[i] = q[p[i]]` (apply p first), the product
            // g · t · g as functions is g.then(&t).then(g).
            let conj = g.then(&t).then(g);
            if set.insert(conj.clone()) {
                queue.push_back(conj);
            }
        }
    }

    set
}

// ---------------------------------------------------------------------------
// Interval extraction
// ---------------------------------------------------------------------------

/// Extract the Bruhat order interval `I = {z : y ≤ z ≤ w}` as a sorted list of
/// element indices (ascending by canonical index, hence by length).
///
/// Uses the KL table's `bruhat_leq` flags.  Because canonical order is sorted
/// by `(length, lex)` and `y ≤_B z` forces `l(y) ≤ l(z)`, every interval member
/// has index in `y..=w`, so we only scan that range.  `O(|w − y|)` per call.
///
/// Requires `y <= w` by canonical index and `y ≤_B w` (caller guarantees the
/// pair is comparable).  Always includes both `y` and `w`.
pub fn extract_interval(table: &KlTable, y: ElmIdx, w: ElmIdx) -> Vec<ElmIdx> {
    debug_assert!(y <= w, "extract_interval: y={y} > w={w}");
    let mut members = Vec::new();
    for z in y..=w {
        // y ≤ z by index (since z ≥ y); z ≤ w by index (since z ≤ w).
        if table.bruhat_leq(y, z) && table.bruhat_leq(z, w) {
            members.push(z);
        }
    }
    members
}

// ---------------------------------------------------------------------------
// Interval digraphs
// ---------------------------------------------------------------------------

/// A directed graph on the vertices of a Bruhat interval, with per-vertex
/// relative-length levels.
///
/// Vertices are `0..n` (local indices into `members`).  `level[i]` is the
/// relative length `l(member_i) − l(y)`; vertex 0 is always `y` (level 0) and
/// the last vertex is always `w`.  Edges are directed *upward* (from lower to
/// higher length).  `out[i]` and `in_[i]` are adjacency lists.
#[derive(Debug, Clone)]
pub struct IntervalGraph {
    /// Number of vertices.
    pub n: usize,
    /// `level[i]` = relative length `l(member_i) − l(y)`.
    pub level: Vec<u32>,
    /// Out-adjacency: `out[i]` lists targets `j` with edge `i → j`.
    pub out: Vec<Vec<usize>>,
    /// In-adjacency: `in_[i]` lists sources `j` with edge `j → i`.
    pub in_: Vec<Vec<usize>>,
}

impl IntervalGraph {
    /// Total number of directed edges.
    pub fn edge_count(&self) -> usize {
        self.out.iter().map(Vec::len).sum()
    }

    /// Add a directed edge `i → j` (no dedup; caller must avoid duplicates).
    fn add_edge(&mut self, i: usize, j: usize) {
        self.out[i].push(j);
        self.in_[j].push(i);
    }
}

/// Which interval graph flavour to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphKind {
    /// Hasse diagram: covers only (length-1 steps).
    Covers,
    /// Full Bruhat graph: all reflection edges.
    Bruhat,
}

/// Build an [`IntervalGraph`] of the requested kind on the interval `members`
/// (output of [`extract_interval`], ascending by index).
///
/// `perms[k]` must be the full permutation of `members[k]`; `lengths[k]` its
/// length.  `reflset` is the reflection set from [`reflections`] (only used for
/// [`GraphKind::Bruhat`]).
///
/// Cover edges: `z1 → z2` with `l(z2) = l(z1) + 1` and `z1 ≤_B z2`.
/// Bruhat edges: `z1 → z2` with `l(z2) > l(z1)` and `z2 · z1⁻¹ ∈ reflset`.
pub fn build_graph(
    members: &[ElmIdx],
    perms: &[Perm],
    lengths: &[u32],
    kind: GraphKind,
    reflset: &HashSet<Perm>,
) -> IntervalGraph {
    let n = members.len();
    debug_assert_eq!(perms.len(), n);
    debug_assert_eq!(lengths.len(), n);

    let base = lengths[0]; // l(y)
    let level: Vec<u32> = lengths.iter().map(|&l| l - base).collect();

    let mut g = IntervalGraph {
        n,
        level,
        out: vec![Vec::new(); n],
        in_: vec![Vec::new(); n],
    };

    // Precompute inverses for the reflection test `z2 · z1⁻¹ ∈ reflections`.
    let invs: Vec<Perm> = perms.iter().map(Perm::inverse).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            // members ascend by index ⇒ lengths[j] ≥ lengths[i].
            let li = lengths[i];
            let lj = lengths[j];
            if lj <= li {
                continue; // equal length ⇒ never an upward edge
            }
            // Both flavours share the reflection test: an upward Bruhat-graph
            // edge i → j exists iff `members[j] · members[i]⁻¹` is a reflection
            // (LEFT convention).  A cover is exactly such an edge with a
            // length gap of 1 — in a Bruhat interval every length-1 reflection
            // edge is a cover, and every cover is a length-1 reflection edge.
            let is_refl_edge = reflset.contains(&perms[j].then(&invs[i]));
            let edge = match kind {
                GraphKind::Covers => lj == li + 1 && is_refl_edge,
                GraphKind::Bruhat => is_refl_edge,
            };
            if edge {
                g.add_edge(i, j);
            }
        }
    }

    g
}

// ---------------------------------------------------------------------------
// Cheap canonical key
// ---------------------------------------------------------------------------

/// A cheap, order-invariant fingerprint of an [`IntervalGraph`].
///
/// Two non-isomorphic graphs may share a key (collisions resolved by the exact
/// test), but isomorphic graphs ALWAYS share a key.  Components:
/// - vertex count `n`;
/// - sorted multiset of level sizes (by relative level);
/// - edge count;
/// - sorted multiset of stable Weisfeiler–Leman colors (color refinement).
///
/// The WL color multiset is a much stronger invariant than raw degree
/// sequences: it folds in the iterated neighbourhood structure, so distinct
/// interval shapes rarely collide and the exact test's buckets stay tiny.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphKey {
    pub n: usize,
    pub level_sizes: Vec<(u32, u32)>,
    pub edges: usize,
    /// Sorted multiset of stable, cross-graph-comparable WL signatures.
    pub color_profile: Vec<u64>,
}

/// Compute stable Weisfeiler–Leman (1-WL / color-refinement) signatures that are
/// **comparable across graphs**: vertices with equal signatures in two graphs
/// occupy structurally identical positions (necessary, not sufficient, for an
/// isomorphism to map one to the other).
///
/// The initial signature of a vertex is a hash of `(level, in_degree,
/// out_degree)` — already comparable across graphs.  Each round replaces every
/// vertex's signature by a hash of `(old_sig, sorted multiset of in-neighbour
/// sigs, sorted multiset of out-neighbour sigs)`, iterating until the induced
/// partition stabilizes (at most `n` rounds).  Because every round mixes only
/// content (never opaque per-graph ids), the resulting `u64` labels mean the
/// same thing in any graph.  Direction is preserved (in/out kept separate).
pub fn wl_signatures(g: &IntervalGraph) -> Vec<u64> {
    use std::hash::{Hash, Hasher};

    fn hash_of<T: Hash>(t: &T) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        t.hash(&mut h);
        h.finish()
    }

    // Initial cross-graph-comparable signature.
    let mut sig: Vec<u64> = (0..g.n)
        .map(|i| hash_of(&(g.level[i], g.in_[i].len() as u32, g.out[i].len() as u32)))
        .collect();

    // Number of distinct signatures; refinement stops when this stabilizes.
    let distinct = |s: &[u64]| {
        let set: std::collections::HashSet<u64> = s.iter().copied().collect();
        set.len()
    };
    let mut prev_distinct = distinct(&sig);

    for _ in 0..g.n {
        let new_sig: Vec<u64> = (0..g.n)
            .map(|i| {
                let mut in_sigs: Vec<u64> = g.in_[i].iter().map(|&j| sig[j]).collect();
                let mut out_sigs: Vec<u64> = g.out[i].iter().map(|&j| sig[j]).collect();
                in_sigs.sort_unstable();
                out_sigs.sort_unstable();
                hash_of(&(sig[i], in_sigs, out_sigs))
            })
            .collect();
        let d = distinct(&new_sig);
        sig = new_sig;
        if d == prev_distinct {
            break; // partition no longer refines
        }
        prev_distinct = d;
    }
    sig
}

/// Compute the cheap [`GraphKey`] for a graph.
pub fn graph_key(g: &IntervalGraph) -> GraphKey {
    let mut level_counts: HashMap<u32, u32> = HashMap::new();
    for &l in &g.level {
        *level_counts.entry(l).or_insert(0) += 1;
    }
    let mut level_sizes: Vec<(u32, u32)> = level_counts.into_iter().collect();
    level_sizes.sort_unstable();

    // Cross-graph-comparable WL signatures → sorted multiset.  Two isomorphic
    // graphs always produce the same sorted signature vector; non-isomorphic
    // graphs usually differ here, keeping exact-test buckets tiny.
    let sigs = wl_signatures(g);
    let mut color_profile: Vec<u64> = sigs.clone();
    color_profile.sort_unstable();

    GraphKey {
        n: g.n,
        level_sizes,
        edges: g.edge_count(),
        color_profile,
    }
}

// ---------------------------------------------------------------------------
// Exact level-respecting digraph isomorphism
// ---------------------------------------------------------------------------

/// Test whether two interval graphs are isomorphic by a level-preserving
/// digraph isomorphism (vertices map only within equal relative level, edges
/// and directions preserved).
///
/// Backtracking with degree-signature pruning.  Panics if the node-expansion
/// budget [`ISO_NODE_CAP`] is exceeded so pathological cases are visible.
pub fn is_isomorphic(a: &IntervalGraph, b: &IntervalGraph) -> bool {
    if a.n != b.n || a.edge_count() != b.edge_count() {
        return false;
    }

    // Adjacency sets for O(1) edge checks during backtracking.
    let a_out: Vec<HashSet<usize>> = a.out.iter().map(|v| v.iter().copied().collect()).collect();
    let b_out: Vec<HashSet<usize>> = b.out.iter().map(|v| v.iter().copied().collect()).collect();

    // Cross-graph-comparable WL signatures: a candidate `j` in `b` for source
    // `i` in `a` must have an equal signature.  This is far stronger than raw
    // degrees and keeps candidate sets tiny even for highly symmetric intervals.
    let sig_a = wl_signatures(a);
    let sig_b = wl_signatures(b);

    // Candidate target vertices in `b` grouped by signature.
    let mut b_by_sig: HashMap<u64, Vec<usize>> = HashMap::new();
    for (j, &s) in sig_b.iter().enumerate() {
        b_by_sig.entry(s).or_default().push(j);
    }
    // If any `a`-signature has a different candidate count than the matching
    // `b`-group, the partitions disagree → not isomorphic.
    {
        let mut a_counts: HashMap<u64, usize> = HashMap::new();
        for &s in &sig_a {
            *a_counts.entry(s).or_insert(0) += 1;
        }
        for (s, c) in &a_counts {
            if b_by_sig.get(s).map_or(0, Vec::len) != *c {
                return false;
            }
        }
        if a_counts.len() != b_by_sig.keys().filter(|k| a_counts.contains_key(k)).count() {
            return false;
        }
    }

    // Order source vertices by ascending candidate-set size (most constrained
    // first) to prune early.
    let mut order: Vec<usize> = (0..a.n).collect();
    order.sort_by_key(|&i| b_by_sig.get(&sig_a[i]).map_or(0, Vec::len));

    let mut mapping: Vec<Option<usize>> = vec![None; a.n];
    let mut used: Vec<bool> = vec![false; b.n];
    let mut nodes: u64 = 0;

    #[allow(clippy::too_many_arguments)]
    fn backtrack(
        depth: usize,
        order: &[usize],
        a_out: &[HashSet<usize>],
        b_out: &[HashSet<usize>],
        b_by_sig: &HashMap<u64, Vec<usize>>,
        sig_a: &[u64],
        mapping: &mut [Option<usize>],
        used: &mut [bool],
        nodes: &mut u64,
    ) -> bool {
        if depth == order.len() {
            return true;
        }
        *nodes += 1;
        if *nodes > ISO_NODE_CAP {
            panic!(
                "is_isomorphic: exceeded ISO_NODE_CAP ({ISO_NODE_CAP}) node expansions — \
                 pathological interval; investigate"
            );
        }
        let i = order[depth];
        let Some(cands) = b_by_sig.get(&sig_a[i]) else {
            return false;
        };
        'cand: for &j in cands {
            if used[j] {
                continue;
            }
            // Check consistency against already-mapped neighbours (both
            // directions), for the edges among already-placed vertices.
            for k in &order[..depth] {
                let mk = mapping[*k].expect("mapped");
                // edge i→k in a must match mi→mk in b
                let a_ik = a_out[i].contains(k);
                let b_jk = b_out[j].contains(&mk);
                if a_ik != b_jk {
                    continue 'cand;
                }
                let a_ki = a_out[*k].contains(&i);
                let b_kj = b_out[mk].contains(&j);
                if a_ki != b_kj {
                    continue 'cand;
                }
            }
            mapping[i] = Some(j);
            used[j] = true;
            if backtrack(
                depth + 1,
                order,
                a_out,
                b_out,
                b_by_sig,
                sig_a,
                mapping,
                used,
                nodes,
            ) {
                return true;
            }
            mapping[i] = None;
            used[j] = false;
        }
        false
    }

    backtrack(
        0,
        &order,
        &a_out,
        &b_out,
        &b_by_sig,
        &sig_a,
        &mut mapping,
        &mut used,
        &mut nodes,
    )
}

#[cfg(test)]
mod tests;
