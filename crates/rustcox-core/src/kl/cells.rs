//! Left / right / two-sided cells, Duflo involutions, and the cell order.
//!
//! This module ports `pycox-ref/pycox_ref.py::klpolynomials` lines
//! ≈10385–10466 — the post-recursion block that, from the completed KL table,
//! derives:
//!
//! 1. **arrows** (`pp`): the pairs `(w, y)` such that `C_y` occurs in
//!    `C_s C_w` for some simple reflection `s`.
//! 2. **adelta / ndelta**: from `p = v^{−L(w)} · P̃_{0,w}`, the (negated)
//!    degree and leading coefficient — the `a`-invariant and sign `n_d`.
//! 3. **lcells**: left cells, the strongly-connected components of the arrow
//!    digraph (PyCox computes mutual-reachability classes; an SCC partition is
//!    the identical result with better asymptotics).
//! 4. **duflo**: the distinguished involution of each left cell.
//! 5. **lorder**: the partial order on left cells (reachability in the
//!    condensation DAG, reflexive).
//! 6. **rcells**: right cells = `{inva[w] : w ∈ left cell}`.
//! 7. **tcells**: two-sided cells = connected components of the relation
//!    "same left cell OR same right cell".
//! 8. **checks_ok**: PyCox's sanity checks (`n_d ∈ {±1}`, the `a`-invariant
//!    minimum is attained at a unique element, and the cell order is
//!    compatible with the `a`-invariant).
//!
//! ## Canonicalisation
//!
//! Because the [`KlTable`] is already in canonical element order, every index
//! produced here is already canonical (PyCox's `sigma` is the identity for us).
//! [`CellData::from_table`] therefore canonicalises purely by sorting: each
//! cell ascending, and each cell *list* lexicographically — with `duflo` and
//! `lorder` permuted by the same permutation as `lcells`.  This reproduces
//! `gen_golden.py::canon_cells` exactly.

use crate::{element::ElmIdx, kl::table::KlTable};

// ---------------------------------------------------------------------------
// CellData
// ---------------------------------------------------------------------------

/// All cell-theoretic data derived from a completed [`KlTable`].
///
/// Every field is canonical: cells are sorted ascending, cell lists are sorted
/// lexicographically, and `duflo` / `lorder` are aligned with the canonical
/// `lcells` order.
#[derive(Clone, Debug, PartialEq)]
pub struct CellData {
    /// Sorted list of arrows `(w, y)`.
    pub arrows: Vec<(ElmIdx, ElmIdx)>,
    /// Left cells: each sorted ascending; the list sorted lexicographically.
    pub lcells: Vec<Vec<ElmIdx>>,
    /// Duflo involutions `(d, a(d), n_d)`, aligned with `lcells`.
    pub duflo: Vec<(ElmIdx, i32, i64)>,
    /// Left-cell order incidence matrix, aligned with `lcells`.
    pub lorder: Vec<Vec<bool>>,
    /// Right cells, canonicalised independently.
    pub rcells: Vec<Vec<ElmIdx>>,
    /// Two-sided cells, canonicalised independently.
    pub tcells: Vec<Vec<ElmIdx>>,
    /// Whether all PyCox sanity checks pass.
    pub checks_ok: bool,
}

impl CellData {
    /// Derive all cell data from a completed KL table.
    ///
    /// The returned struct is fully canonical (see the module docs).
    pub fn from_table(t: &KlTable) -> Self {
        let n = t.n();
        let rank = t.rank();
        let elms = &t.elms;

        // -------------------------------------------------------------------
        // 1. arrows (pp)
        // -------------------------------------------------------------------
        let arrows = build_arrows(t, n, rank);

        // -------------------------------------------------------------------
        // 2. adelta / ndelta
        // -------------------------------------------------------------------
        let (adelta, ndelta) = build_deltas(t, n);

        // -------------------------------------------------------------------
        // 3. left cells = SCCs of the arrow digraph (edges w → y)
        // -------------------------------------------------------------------
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &(w, y) in &arrows {
            adj[w as usize].push(y);
        }
        // PyCox's pp0[w] always begins with [w] (the self-loop), so every
        // node reaches itself; Tarjan handles isolated nodes as singleton SCCs.
        let (comp_of, num_comp) = tarjan_scc(&adj, n);

        // Members of each raw SCC (ascending by construction: we iterate
        // 0..n).  These are the pre-canonical left cells.
        let mut raw_cells: Vec<Vec<ElmIdx>> = vec![Vec::new(); num_comp];
        for w in 0..n {
            raw_cells[comp_of[w]].push(w as ElmIdx);
        }

        // -------------------------------------------------------------------
        // 4. duflo per cell + checks
        // -------------------------------------------------------------------
        let mut checks_ok = true;
        let mut raw_duflo: Vec<(ElmIdx, i32, i64)> = Vec::with_capacity(num_comp);
        for cell in &raw_cells {
            let (d, ok) = duflo_of_cell(cell, &adelta, &ndelta);
            checks_ok &= ok;
            raw_duflo.push((d, adelta[d as usize], ndelta[d as usize]));
        }

        // -------------------------------------------------------------------
        // 5. lorder: reachability between duflo elements in the digraph.
        //    Equivalently, condensation-DAG reachability comp(c1) →* comp(c2),
        //    reflexive.  Compute per-cell BFS on the condensation.
        // -------------------------------------------------------------------
        // Condensation adjacency (deduplicated component edges).
        let cond_adj = condensation_adj(&adj, &comp_of, num_comp, n);
        // comp_reach[c1][c2] = comp c2 reachable from comp c1 (reflexive).
        let comp_reach = condensation_reachability(&cond_adj, num_comp);

        // raw_duflo[i] belongs to raw_cells[i], whose component is
        // comp_of[member]; map cell index → component index.
        let cell_comp: Vec<usize> = raw_cells.iter().map(|c| comp_of[c[0] as usize]).collect();
        let mut raw_lorder: Vec<Vec<bool>> = vec![vec![false; num_comp]; num_comp];
        for c1 in 0..num_comp {
            for c2 in 0..num_comp {
                raw_lorder[c1][c2] = comp_reach[cell_comp[c1]][cell_comp[c2]];
            }
        }

        // Final lorder/a-invariant compatibility check (PyCox ≈10440–10443).
        for c1 in 0..num_comp {
            for c2 in 0..num_comp {
                if c1 != c2 && raw_lorder[c1][c2] && raw_duflo[c1].1 >= raw_duflo[c2].1 {
                    checks_ok = false;
                }
            }
        }

        // -------------------------------------------------------------------
        // 6. rcells = {inva[w] for w in cell}; tcells via union-find.
        // -------------------------------------------------------------------
        let raw_rcells: Vec<Vec<ElmIdx>> = raw_cells
            .iter()
            .map(|c| c.iter().map(|&w| elms.inva[w as usize]).collect())
            .collect();
        let raw_tcells = build_tcells(&raw_cells, &raw_rcells, n);

        // -------------------------------------------------------------------
        // Canonicalisation (matches gen_golden.py::canon_cells).
        // -------------------------------------------------------------------
        let (lcells, lperm) = canon_cells(raw_cells);
        let (rcells, _) = canon_cells(raw_rcells);
        let (tcells, _) = canon_cells(raw_tcells);

        // duflo permuted by lperm.
        let duflo: Vec<(ElmIdx, i32, i64)> = lperm.iter().map(|&i| raw_duflo[i]).collect();
        // lorder: rows AND columns permuted by lperm.
        let lorder: Vec<Vec<bool>> = lperm
            .iter()
            .map(|&i| lperm.iter().map(|&j| raw_lorder[i][j]).collect())
            .collect();

        CellData {
            arrows,
            lcells,
            duflo,
            lorder,
            rcells,
            tcells,
            checks_ok,
        }
    }
}

// ---------------------------------------------------------------------------
// 1. arrows
// ---------------------------------------------------------------------------

/// Build the sorted arrow list `pp` (PyCox ≈10385–10394).
///
/// For each `w`, each generator `s`:
/// - emit `(w, lft(w, s))` when `weights[s] == 0`, OR when
///   `lft(w, s) > w` and `weights[s] > 0`.
///
/// For each `w`, each `y < w` with `(w, y)` Bruhat-comparable:
/// - emit `(w, y)` when some generator `s` has `weights[s] > 0`,
///   `lft(y, s) < y`, `lft(w, s) > w`, and `μ^s_{y,w} ≠ 0`.
fn build_arrows(t: &KlTable, n: usize, rank: usize) -> Vec<(ElmIdx, ElmIdx)> {
    let elms = &t.elms;
    let weights = &t.weights;
    let mut pp: Vec<(ElmIdx, ElmIdx)> = Vec::new();

    for w in 0..n {
        let wu = w as ElmIdx;
        for (s, &weight) in weights.iter().enumerate().take(rank) {
            let sw = elms.lft(wu, s);
            if weight == 0 || (sw > wu && weight > 0) {
                pp.push((wu, sw));
            }
        }
        for y in 0..w {
            let yu = y as ElmIdx;
            if !t.bruhat_leq(yu, wu) {
                continue;
            }
            let has_arrow = (0..rank).any(|s| {
                weights[s] > 0
                    && elms.lft(yu, s) < yu
                    && elms.lft(wu, s) > wu
                    && t.mu_is_nonzero(s, yu, wu)
            });
            if has_arrow {
                pp.push((wu, yu));
            }
        }
    }

    pp.sort_unstable();
    pp
}

// ---------------------------------------------------------------------------
// 2. adelta / ndelta
// ---------------------------------------------------------------------------

/// Compute `(adelta, ndelta)` for every element (PyCox ≈10396–10405).
///
/// `p = v^{−L(w)} · P̃_{0,w}` (the row's `y = 0` polynomial shifted by the
/// element's weighted length).  If `p == 0` (possible with weight-0
/// generators), `adelta[w] = −1`, `ndelta[w] = 0`.  Otherwise
/// `adelta[w] = −deg(p)`, `ndelta[w] = leading_coeff(p)`.
fn build_deltas(t: &KlTable, n: usize) -> (Vec<i32>, Vec<i64>) {
    let mut adelta = Vec::with_capacity(n);
    let mut ndelta = Vec::with_capacity(n);
    for w in 0..n {
        // P̃_{0,w} is pol(0, w); for w == 0 this is P_{e,e} = 1.
        let p_raw = t
            .pol(0, w as ElmIdx)
            .expect("P_{0,w}: identity is Bruhat-below every element");
        let shift = -(t.lweights[w] as i32);
        let p = p_raw.shifted(shift);
        if p.is_zero() {
            adelta.push(-1);
            ndelta.push(0);
        } else {
            adelta.push(-p.degree().expect("non-zero polynomial has a degree"));
            ndelta.push(p.leading_coeff());
        }
    }
    (adelta, ndelta)
}

// ---------------------------------------------------------------------------
// 4. duflo
// ---------------------------------------------------------------------------

/// Find the Duflo involution of a cell and run PyCox's per-cell checks
/// (≈10423–10433).  `cell` is the ascending member list.
///
/// Returns `(d, checks_ok)`.
fn duflo_of_cell(cell: &[ElmIdx], adelta: &[i32], ndelta: &[i64]) -> (ElmIdx, bool) {
    // i0 = first index with ndelta != 0.
    let i0 = cell
        .iter()
        .position(|&w| ndelta[w as usize] != 0)
        .expect("every left cell contains an element with n_d != 0");
    let mut d = cell[i0];
    for &w in &cell[i0..] {
        if ndelta[w as usize] != 0 && adelta[w as usize] < adelta[d as usize] {
            d = w;
        }
    }

    let mut ok = true;
    // The a-invariant minimum must be attained at a unique element of the
    // cell (counted over ALL members, not just n_d != 0).
    let count_min = cell
        .iter()
        .filter(|&&w| adelta[w as usize] == adelta[d as usize])
        .count();
    if count_min > 1 {
        ok = false;
    }
    // n_d must be ±1.
    let nd = ndelta[d as usize];
    if nd != 1 && nd != -1 {
        ok = false;
    }
    (d, ok)
}

// ---------------------------------------------------------------------------
// 6. tcells via union-find
// ---------------------------------------------------------------------------

/// Build two-sided cells: connected components of the relation
/// "same left cell OR same right cell" over all elements (PyCox ≈10445–10465).
///
/// Implemented with union-find: union every pair within a shared left cell and
/// within a shared right cell, then collect components.
fn build_tcells(lcells: &[Vec<ElmIdx>], rcells: &[Vec<ElmIdx>], n: usize) -> Vec<Vec<ElmIdx>> {
    let mut uf = UnionFind::new(n);
    for cell in lcells.iter().chain(rcells.iter()) {
        if let Some((&first, rest)) = cell.split_first() {
            for &w in rest {
                uf.union(first as usize, w as usize);
            }
        }
    }
    // Group by representative; preserve ascending member order.
    let mut groups: std::collections::BTreeMap<usize, Vec<ElmIdx>> =
        std::collections::BTreeMap::new();
    for w in 0..n {
        groups.entry(uf.find(w)).or_default().push(w as ElmIdx);
    }
    groups.into_values().collect()
}

// ---------------------------------------------------------------------------
// Canonicalisation
// ---------------------------------------------------------------------------

/// Sort each cell ascending and the list of cells lexicographically.
///
/// Returns `(sorted_cells, perm)` where `perm[new] = old`, i.e. the new cell
/// at position `i` was originally at `perm[i]` (matches `gen_golden.py`'s
/// `order` permutation, used to permute `duflo` and `lorder`).
fn canon_cells(cells: Vec<Vec<ElmIdx>>) -> (Vec<Vec<ElmIdx>>, Vec<usize>) {
    let mut sorted: Vec<Vec<ElmIdx>> = cells
        .into_iter()
        .map(|mut c| {
            c.sort_unstable();
            c
        })
        .collect();
    let mut perm: Vec<usize> = (0..sorted.len()).collect();
    perm.sort_by(|&a, &b| sorted[a].cmp(&sorted[b]));
    // Reorder `sorted` by `perm` without cloning: drain into Options, then
    // pull each slot out in the new order.
    let mut slots: Vec<Option<Vec<ElmIdx>>> = sorted.drain(..).map(Some).collect();
    let reordered: Vec<Vec<ElmIdx>> = perm
        .iter()
        .map(|&i| slots[i].take().expect("each cell taken exactly once"))
        .collect();
    (reordered, perm)
}

// ---------------------------------------------------------------------------
// Tarjan SCC
// ---------------------------------------------------------------------------

/// Tarjan's strongly-connected-components algorithm (iterative, no recursion
/// to avoid deep-stack blowups).  Returns `(comp_of, num_comp)` where
/// `comp_of[v]` is the component id of vertex `v`.
fn tarjan_scc(adj: &[Vec<u32>], n: usize) -> (Vec<usize>, usize) {
    const UNVISITED: u32 = u32::MAX;

    let mut index = vec![UNVISITED; n];
    let mut lowlink = vec![0u32; n];
    let mut on_stack = vec![false; n];
    let mut comp_of = vec![usize::MAX; n];
    let mut stack: Vec<u32> = Vec::new();
    let mut next_index: u32 = 0;
    let mut num_comp = 0usize;

    // Explicit DFS stack: each frame is (vertex, next-child-cursor).
    let mut call: Vec<(u32, usize)> = Vec::new();

    for start in 0..n {
        if index[start] != UNVISITED {
            continue;
        }
        call.push((start as u32, 0));
        while let Some(&(v, ci)) = call.last() {
            let vu = v as usize;
            if ci == 0 {
                index[vu] = next_index;
                lowlink[vu] = next_index;
                next_index += 1;
                stack.push(v);
                on_stack[vu] = true;
            }
            if ci < adj[vu].len() {
                // Advance the cursor before recursing.
                call.last_mut().unwrap().1 = ci + 1;
                let w = adj[vu][ci];
                let wu = w as usize;
                if index[wu] == UNVISITED {
                    call.push((w, 0));
                } else if on_stack[wu] {
                    lowlink[vu] = lowlink[vu].min(index[wu]);
                }
            } else {
                // Done with v: if it's a root, pop an SCC.
                if lowlink[vu] == index[vu] {
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack[w as usize] = false;
                        comp_of[w as usize] = num_comp;
                        if w == v {
                            break;
                        }
                    }
                    num_comp += 1;
                }
                call.pop();
                // Propagate lowlink to parent.
                if let Some(&(parent, _)) = call.last() {
                    let pu = parent as usize;
                    lowlink[pu] = lowlink[pu].min(lowlink[vu]);
                }
            }
        }
    }

    (comp_of, num_comp)
}

// ---------------------------------------------------------------------------
// Condensation reachability
// ---------------------------------------------------------------------------

/// Build deduplicated condensation adjacency: `out[c]` lists distinct
/// successor components of component `c` (excluding self).
fn condensation_adj(
    adj: &[Vec<u32>],
    comp_of: &[usize],
    num_comp: usize,
    n: usize,
) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); num_comp];
    for v in 0..n {
        let cv = comp_of[v];
        for &w in &adj[v] {
            let cw = comp_of[w as usize];
            if cw != cv {
                out[cv].push(cw);
            }
        }
    }
    for row in &mut out {
        row.sort_unstable();
        row.dedup();
    }
    out
}

/// Reachability over the condensation DAG, reflexive.
///
/// `reach[c1][c2]` is `true` iff `c2` is reachable from `c1` (including
/// `c1 == c2`).  Per-component BFS; the condensation is a DAG so each BFS is
/// linear in the component subgraph.
fn condensation_reachability(cond_adj: &[Vec<usize>], num_comp: usize) -> Vec<Vec<bool>> {
    let mut reach = vec![vec![false; num_comp]; num_comp];
    let mut queue: Vec<usize> = Vec::new();
    for start in 0..num_comp {
        let row = &mut reach[start];
        row[start] = true;
        queue.clear();
        queue.push(start);
        let mut head = 0;
        while head < queue.len() {
            let c = queue[head];
            head += 1;
            for &nc in &cond_adj[c] {
                if !row[nc] {
                    row[nc] = true;
                    queue.push(nc);
                }
            }
        }
    }
    reach
}

// ---------------------------------------------------------------------------
// Union-find
// ---------------------------------------------------------------------------

/// Disjoint-set union with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Path compression.
        let mut cur = x;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::CoxeterGroup;
    use crate::kl::{klpolynomials_seq, KlOpts};

    fn cells_for(ty: &str, weights: Vec<u32>) -> CellData {
        let group = CoxeterGroup::from_type(ty).unwrap();
        let opts = KlOpts {
            weights,
            threads: None,
            layer_chunk: None,
        };
        let table = klpolynomials_seq(&group, &opts).unwrap();
        CellData::from_table(&table)
    }

    /// A3 cell-size multiset and various counts (plan §11 pins).
    #[test]
    fn a3_cell_sizes_and_counts() {
        let rank = 3;
        let cd = cells_for("A3", vec![1; rank]);

        let mut sizes: Vec<usize> = cd.lcells.iter().map(|c| c.len()).collect();
        sizes.sort_unstable();
        assert_eq!(
            sizes,
            vec![1, 1, 2, 2, 3, 3, 3, 3, 3, 3],
            "A3 left-cell size multiset"
        );

        assert_eq!(cd.arrows.len(), 54, "A3 arrow count");

        let true_count: usize = cd
            .lorder
            .iter()
            .map(|row| row.iter().filter(|&&b| b).count())
            .sum();
        assert_eq!(true_count, 39, "A3 lorder true-count");

        assert!(cd.checks_ok, "A3 checks_ok");
    }

    /// B2 with weights [2,1]: pinned lcells and a specific duflo entry.
    #[test]
    fn b2_w2_1_cells() {
        let cd = cells_for("B2", vec![2, 1]);
        assert_eq!(
            cd.lcells,
            vec![vec![0], vec![1, 4], vec![2], vec![3, 6], vec![5], vec![7],],
            "B2[2,1] lcells"
        );
        assert!(
            cd.duflo.contains(&(5, 3, -1)),
            "B2[2,1] duflo contains (5,3,-1); got {:?}",
            cd.duflo
        );
        assert!(cd.checks_ok, "B2[2,1] checks_ok");
    }

    /// checks_ok holds for several known-good groups.
    #[test]
    fn checks_ok_known_groups() {
        assert!(cells_for("A3", vec![1; 3]).checks_ok, "A3");
        assert!(cells_for("B3", vec![1; 3]).checks_ok, "B3");
        assert!(cells_for("B2", vec![2, 1]).checks_ok, "B2[2,1]");
        assert!(cells_for("H3", vec![1; 3]).checks_ok, "H3");
    }

    /// Duflo elements are distinct and exactly one per left cell.
    #[test]
    fn duflo_one_per_cell() {
        let cd = cells_for("A3", vec![1; 3]);
        assert_eq!(cd.duflo.len(), cd.lcells.len());
        // Each duflo element belongs to its aligned cell.
        for (cell, &(d, _, _)) in cd.lcells.iter().zip(cd.duflo.iter()) {
            assert!(cell.contains(&d), "duflo {d} must lie in its cell {cell:?}");
        }
    }
}
