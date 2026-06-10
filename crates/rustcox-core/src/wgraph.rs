//! Minimal W-graphs extracted from a completed KL table.
//!
//! A W-graph is a combinatorial object encoding a representation of the
//! generic Hecke algebra.  For a left cell (or any element subset), each
//! vertex carries its left-descent set I(x) and edges carry per-generator
//! mu-coefficients.
//!
//! Reference: `pycox-ref/pycox_ref.py` class `wgraph` (≈9698–10051) and its
//! `decompose` method (≈9940–9971).  We implement a **minimal** version:
//! - `WGraph::of_cell` – build the W-graph for an element subset.
//! - `WGraph::decompose` – split into strongly-connected components using
//!   the standard W-graph arrow condition.
//!
//! ## Arrow directions in `decompose`
//!
//! The directed arrows used for SCC computation come from two sources,
//! matching `cells.rs::build_arrows`:
//!
//! **Type 1 — mu-based:**  For each stored edge `(i, j)` (with `i < j`) and
//! each generator `s` with non-zero `μ^s`:
//! - If `s ∈ I(vertices[i])` and `s ∉ I(vertices[j])`: arrow `j → i`.
//! - If `s ∈ I(vertices[j])` and `s ∉ I(vertices[i])`: arrow `i → j`.
//!
//! **Type 2 — direct `s·u`:** For each vertex `u` at position `pos` and
//! each generator `s` with `s ∉ I(u)`, if `s·u` is also in `self.vertices`,
//! add arrow `pos → pos_of(s·u)`.  This mirrors `build_arrows`'s
//! `(wu, sw)` emission when `sw > wu`.

use std::collections::HashMap;

use crate::{
    element::{ElmIdx, Gen},
    kl::KlTable,
    laurent::Laurent,
};

use crate::kl::scc::tarjan_scc;

// ---------------------------------------------------------------------------
// WGraph
// ---------------------------------------------------------------------------

/// Minimal W-graph of a left cell (or any element subset) extracted from a
/// KL table.
///
/// ## Public fields
///
/// - `vertices`: ascending element indices (into the canonical element order).
/// - `isets`: left-descent set of each vertex (parallel to `vertices`).
/// - `edges`: keyed by `(i, j)` with `i < j` (positions in `vertices`);
///   the value is `[μ^0_{y,w}, μ^1_{y,w}, …]` where `(y, w)` is the
///   Bruhat-ordered pair (`y = vertices[i] < vertices[j] = w`).  Only
///   inserted when at least one μ-value is non-zero.
#[derive(Clone, Debug)]
pub struct WGraph {
    /// Element indices in ascending order.
    pub vertices: Vec<ElmIdx>,
    /// Left-descent set of each vertex (same ordering as `vertices`).
    pub isets: Vec<Vec<Gen>>,
    /// Mu-value vectors keyed by `(i, j)`, `i < j` (vertex positions).
    ///
    /// Value = `vec![μ^0_{y,w}, μ^1_{y,w}, …]` for the Bruhat-ordered pair
    /// `(y = vertices[i], w = vertices[j])`.  Only present when at least one
    /// generator has a non-zero μ.
    pub edges: HashMap<(u32, u32), Vec<Laurent>>,

    /// Private: for each `(vertex_pos * rank + s)`, the position of `s·vertex`
    /// within `self.vertices`, or `u32::MAX` if not present.  Used by
    /// `decompose` to compute type-2 (direct `s·u`) arrows without needing
    /// a back-reference to the KL table.
    lft_cell: Vec<u32>,
    /// Number of generators (rank).
    rank: usize,
}

impl WGraph {
    /// Build the W-graph of `cell` (element indices) from the KL table.
    ///
    /// For each unordered pair `{i, j}` of vertex positions (with `i < j`),
    /// find the Bruhat-comparable ordering `(y, w)` of the corresponding
    /// element indices.  If Bruhat-comparable, collect
    /// `mus = [table.mu(s, y, w) for s in 0..rank]`; insert the edge
    /// `(i, j) → mus` when at least one μ-value is non-zero.
    ///
    /// Left-descent sets are derived from the element table via
    /// `lft(w, s) < w`.
    pub fn of_cell(t: &KlTable, cell: &[ElmIdx]) -> Self {
        let mut vertices: Vec<ElmIdx> = cell.to_vec();
        vertices.sort_unstable();

        let n = vertices.len();
        let rank = t.rank();

        // Build a fast lookup from element index to position in vertices.
        let pos_of: HashMap<ElmIdx, u32> = vertices
            .iter()
            .enumerate()
            .map(|(pos, &v)| (v, pos as u32))
            .collect();

        // Compute left-descent sets.
        // s ∈ Ldesc(w)  ⟺  lft(w, s) < w  (by element table invariant).
        let isets: Vec<Vec<Gen>> = vertices
            .iter()
            .map(|&w| {
                (0..rank)
                    .filter(|&s| t.elms.lft(w, s) < w)
                    .map(|s| s as Gen)
                    .collect()
            })
            .collect();

        // Compute lft_cell: for each (pos, s), the position of s·vertex within
        // self.vertices (or u32::MAX if not in the cell).
        let mut lft_cell = vec![u32::MAX; n * rank];
        for (pos, &v) in vertices.iter().enumerate() {
            for s in 0..rank {
                let sv = t.elms.lft(v, s);
                if let Some(&sv_pos) = pos_of.get(&sv) {
                    lft_cell[pos * rank + s] = sv_pos;
                }
            }
        }

        // Build edges.
        // Positions i < j; element indices ei = vertices[i] < vertices[j] = ej.
        // Bruhat-comparable check: bruhat_leq(ei, ej).
        let mut edges: HashMap<(u32, u32), Vec<Laurent>> = HashMap::new();

        for j in 0..n {
            for i in 0..j {
                let ei = vertices[i];
                let ej = vertices[j];
                // ei < ej as canonical indices (sorted ascending).
                if !t.bruhat_leq(ei, ej) {
                    continue; // not Bruhat-comparable
                }

                // Collect μ-values for all generators.
                let mus: Vec<Laurent> = (0..rank).map(|s| t.mu(s, ei, ej)).collect();

                // Only store the edge if some μ is non-zero.
                if mus.iter().any(|m| !m.is_zero()) {
                    edges.insert((i as u32, j as u32), mus);
                }
            }
        }

        WGraph {
            vertices,
            isets,
            edges,
            lft_cell,
            rank,
        }
    }

    /// Decompose the W-graph into strongly-connected components (SCCs) and
    /// return them as sub-WGraphs, sorted canonically (by first vertex).
    ///
    /// ## Arrow conditions (matching `cells.rs::build_arrows`)
    ///
    /// **Type 1 — mu-based:** For each stored edge `(i, j)` and non-zero `μ^s`:
    /// - `s ∈ I(vertices[i])` and `s ∉ I(vertices[j])` → arrow `j → i`.
    /// - `s ∈ I(vertices[j])` and `s ∉ I(vertices[i])` → arrow `i → j`.
    ///
    /// **Type 2 — direct:** For each vertex `u` at position `pos` and each
    /// generator `s` with `s ∉ I(u)`: if `s·u ∈ self.vertices`, add `pos → pos_of(s·u)`.
    ///
    /// ## Canonicalisation
    ///
    /// Each component's vertex list is sorted ascending; the list of
    /// components is sorted lexicographically by first vertex.
    pub fn decompose(&self) -> Vec<WGraph> {
        let n = self.vertices.len();
        if n == 0 {
            return vec![];
        }

        // Build directed adjacency for Tarjan.
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];

        // Helper: add u→v without duplicates.
        let push_unique = |adj: &mut Vec<Vec<u32>>, u: usize, v: usize| {
            if !adj[u].contains(&(v as u32)) {
                adj[u].push(v as u32);
            }
        };

        // Type 1: mu-based arrows.
        //
        // Edge (pi, pj) with pi < pj stores mus for (y=vertices[pi], w=vertices[pj]).
        // build_arrows condition: s ∈ I(y) AND s ∉ I(w) → arrow w→y, i.e., pj→pi.
        // Symmetric:            s ∈ I(w) AND s ∉ I(y) → arrow y→w is NOT in build_arrows.
        //
        // Wait: build_arrows also emits (wu, sw) for sw > wu (weight>0 case).
        // That corresponds to type-2 arrows.  So type-1 is strictly:
        //   s ∈ I(y=smaller) AND s ∉ I(w=larger) → arrow w→y (pj→pi).
        //
        // Note: in PyCox wgraph constructor, mmat[(y, x)] is set when
        //   s ∈ Isets[x] AND s ∉ Isets[y]  (x < y as positions).
        // Then decompose does pp0[y].append(x) = arrow y→x.
        // In PyCox, x = position of Bruhat-smaller (smaller index in xset['elms']).
        // So (y, x) means "y→x" where y=larger-index (Bruhat-larger = w),
        // x=smaller-index (Bruhat-smaller = el).
        // Condition: s ∈ Isets[x=smaller] AND s ∉ Isets[y=larger].
        // Arrow: y→x = w→y_small (in our notation, pj→pi when pi<pj).
        for (&(pi, pj), mus) in &self.edges {
            let pi = pi as usize;
            let pj = pj as usize;
            for (s, mu) in mus.iter().enumerate() {
                if mu.is_zero() {
                    continue;
                }
                let s_gen = s as Gen;
                // i_has_s: s ∈ I(vertices[pi]) = I(Bruhat-smaller y)
                let i_has_s = self.isets[pi].contains(&s_gen);
                // j_has_s: s ∈ I(vertices[pj]) = I(Bruhat-larger w)
                let j_has_s = self.isets[pj].contains(&s_gen);
                // build_arrows condition: s ∈ I(y=pi) AND s ∉ I(w=pj) → arrow w→y = pj→pi.
                if i_has_s && !j_has_s {
                    push_unique(&mut adj, pj, pi);
                }
                // The reverse condition (s ∈ I(w) AND s ∉ I(y)) is covered by
                // the wgraph mmat constructor's y<x range — but here x < y as
                // positions and the mu is defined as mu(s, y_small, w_large),
                // so there is no separate "reverse mu".  This case never adds an
                // arrow from the mu-based edges alone.
            }
        }

        // Type 2: direct s·u arrows (matches build_arrows' (wu, sw) emission).
        //
        // For each vertex u at position pos and generator s with s ∉ I(u):
        // lft(u, s) > u (u is longer after left-mult by s).  If s·u ∈ vertices,
        // add arrow pos → pos_of(s·u).
        for pos in 0..n {
            for s in 0..self.rank {
                let s_gen = s as Gen;
                if self.isets[pos].contains(&s_gen) {
                    continue; // s is a descent: lft(v,s) < v (shorter), not sw > w
                }
                let sv_pos = self.lft_cell[pos * self.rank + s];
                if sv_pos != u32::MAX {
                    push_unique(&mut adj, pos, sv_pos as usize);
                }
            }
        }

        // Tarjan SCC.
        let (comp_of, num_comp) = tarjan_scc(&adj, n);

        // Group vertex positions by component.
        let mut comp_members: Vec<Vec<usize>> = vec![Vec::new(); num_comp];
        for v in 0..n {
            comp_members[comp_of[v]].push(v);
        }

        // Build a sub-WGraph for each component.
        let mut components: Vec<WGraph> = comp_members
            .into_iter()
            .map(|members| self.subgraph(&members))
            .collect();

        // Sort components by their first (smallest) vertex element index.
        components.sort_by_key(|g| g.vertices.first().copied().unwrap_or(u32::MAX));

        components
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Build a sub-WGraph from a list of vertex positions (within `self`).
    ///
    /// The returned sub-graph has its vertices sorted ascending and its
    /// edges re-keyed by new positions within the sub-graph.  `lft_cell` is
    /// recomputed for the sub-graph by looking up positions in the new set.
    fn subgraph(&self, positions: &[usize]) -> WGraph {
        // Sort positions so vertices come out ascending.
        let mut sorted_pos = positions.to_vec();
        sorted_pos.sort_unstable();

        let new_n = sorted_pos.len();
        let rank = self.rank;

        // Map old position → new position.
        let mut old_to_new: HashMap<u32, u32> = HashMap::with_capacity(new_n);
        for (new_pos, &old_pos) in sorted_pos.iter().enumerate() {
            old_to_new.insert(old_pos as u32, new_pos as u32);
        }

        let vertices: Vec<ElmIdx> = sorted_pos.iter().map(|&p| self.vertices[p]).collect();
        let isets: Vec<Vec<Gen>> = sorted_pos.iter().map(|&p| self.isets[p].clone()).collect();

        // Collect edges where both endpoints are in this component.
        let mut edges: HashMap<(u32, u32), Vec<Laurent>> = HashMap::new();
        for (&(pi, pj), mus) in &self.edges {
            if let (Some(&ni), Some(&nj)) = (old_to_new.get(&pi), old_to_new.get(&pj)) {
                let (lo, hi) = if ni < nj { (ni, nj) } else { (nj, ni) };
                edges.insert((lo, hi), mus.clone());
            }
        }

        // Recompute lft_cell for the sub-graph.
        // For each new position and generator, look up whether the lft target
        // is in the new sub-graph.
        let mut lft_cell = vec![u32::MAX; new_n * rank];
        for (new_pos, &old_pos) in sorted_pos.iter().enumerate() {
            for s in 0..rank {
                let old_sv_pos = self.lft_cell[old_pos * rank + s];
                if old_sv_pos != u32::MAX {
                    if let Some(&new_sv_pos) = old_to_new.get(&old_sv_pos) {
                        lft_cell[new_pos * rank + s] = new_sv_pos;
                    }
                }
            }
        }

        WGraph {
            vertices,
            isets,
            edges,
            lft_cell,
            rank,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        group::CoxeterGroup,
        kl::{cells::CellData, klpolynomials_seq, KlOpts},
    };

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn build_table(ty: &str, weights: Vec<u32>) -> (CoxeterGroup, KlTable) {
        let group = CoxeterGroup::from_type(ty).unwrap();
        let opts = KlOpts {
            weights,
            threads: None,
            layer_chunk: None,
        };
        let table = klpolynomials_seq(&group, &opts).unwrap();
        (group, table)
    }

    // -----------------------------------------------------------------------
    // Test 1: a3_cell_wgraphs
    // -----------------------------------------------------------------------
    /// A3 equal params: every left cell's W-graph is indecomposable (exactly
    /// one SCC component), and the union of all cell vertex sets covers all
    /// 24 elements.
    #[test]
    fn a3_cell_wgraphs() {
        let (group, t) = build_table("A3", vec![1; 3]);
        let cd = CellData::from_table(&t);

        let mut total_vertices = 0usize;

        for (cell_idx, cell) in cd.lcells.iter().enumerate() {
            let wg = WGraph::of_cell(&t, cell);

            // Vertices must equal the cell.
            assert_eq!(wg.vertices, *cell, "cell {cell_idx}: vertices mismatch");

            // Isets must equal left descent sets from the group.
            for (pos, &elm) in wg.vertices.iter().enumerate() {
                let word = &t.elms.elms[elm as usize];
                let perm = group.word_to_perm(word);
                let expected_iset = group.left_descents(&perm);
                assert_eq!(
                    wg.isets[pos], expected_iset,
                    "cell {cell_idx} vertex {elm}: iset mismatch (got {:?}, expected {:?})",
                    wg.isets[pos], expected_iset
                );
            }

            // decompose() must yield exactly 1 component.
            let comps = wg.decompose();
            assert_eq!(
                comps.len(),
                1,
                "cell {cell_idx} ({cell:?}): expected 1 SCC component, got {}",
                comps.len()
            );

            // The single component must cover all vertices.
            assert_eq!(
                comps[0].vertices, wg.vertices,
                "cell {cell_idx}: component vertices mismatch"
            );

            total_vertices += cell.len();
        }

        assert_eq!(total_vertices, 24, "A3 has 24 elements total");
    }

    // -----------------------------------------------------------------------
    // Test 2: b3_cell_wgraphs_connected
    // -----------------------------------------------------------------------
    /// B3 equal params: every left cell's W-graph has exactly 1 SCC.
    #[test]
    fn b3_cell_wgraphs_connected() {
        let (_group, t) = build_table("B3", vec![1; 3]);
        let cd = CellData::from_table(&t);

        for (cell_idx, cell) in cd.lcells.iter().enumerate() {
            let wg = WGraph::of_cell(&t, cell);
            let comps = wg.decompose();
            assert_eq!(
                comps.len(),
                1,
                "B3 cell {cell_idx} ({cell:?}): expected 1 SCC, got {}",
                comps.len()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: full_group_decompose_reproduces_cells
    // -----------------------------------------------------------------------
    /// A3: build W-graph of all 24 elements, then decompose().  The resulting
    /// component vertex-sets must match the canonical left cells from CellData.
    #[test]
    fn full_group_decompose_reproduces_cells() {
        let (_group, t) = build_table("A3", vec![1; 3]);
        let cd = CellData::from_table(&t);
        let n = t.n();

        let all: Vec<ElmIdx> = (0..n as u32).collect();
        let full_wg = WGraph::of_cell(&t, &all);
        let mut comps = full_wg.decompose();

        // Sort both component sets and lcells canonically.
        comps.sort_by(|a, b| a.vertices.cmp(&b.vertices));
        let mut lcells = cd.lcells.clone();
        lcells.sort();

        assert_eq!(comps.len(), lcells.len(), "number of components");

        for (c, (comp, cell)) in comps.iter().zip(lcells.iter()).enumerate() {
            assert_eq!(
                comp.vertices, *cell,
                "component {c}: vertices {:?} != lcell {:?}",
                comp.vertices, cell
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: b2_w2_1_edge_values
    // -----------------------------------------------------------------------
    /// B2 with weights [2,1] (Stored mode): build the W-graph of cell [1,4]
    /// (canonical indices from the golden file lcells = [[0],[1,4],...]).
    ///
    /// The edge between positions (0,1) must exist and its mu vector must
    /// match `table.mu(s, y=1, w=4)` for s in 0..2.
    ///
    /// Cross-check against golden `kl_B2_w2_1.json`:
    /// - `mumat[4][1] = [0, -1]` where index 0 in `mues[0]` is `v^{-1}+v`
    ///   and -1 means NO_MU.
    /// - So `mu(s=0, y=1, w=4) = v^{-1}+v` and `mu(s=1, y=1, w=4) = 0`.
    #[test]
    fn b2_w2_1_edge_values() {
        let (_group, t) = build_table("B2", vec![2, 1]);
        let cd = CellData::from_table(&t);

        // Verify the cell [1,4] is in the lcells (canonical from golden).
        let cell14: Vec<ElmIdx> = vec![1, 4];
        assert!(
            cd.lcells.contains(&cell14),
            "B2[2,1] lcells should contain [1,4]: got {:?}",
            cd.lcells
        );

        let wg = WGraph::of_cell(&t, &cell14);

        // Vertices should be [1, 4].
        assert_eq!(wg.vertices, vec![1u32, 4u32]);

        // Edge (0,1) must exist.
        let mus = wg
            .edges
            .get(&(0, 1))
            .expect("B2[2,1]: edge (0,1) between elms 1 and 4 must be present");

        assert_eq!(mus.len(), 2, "rank=2: mu vector length should be 2");

        // Cross-check: mu(s=0, y=1, w=4) == table.mu(0, 1, 4).
        // From golden: mues[0][0] = v^{-1}+v (Laurent: v=-1, coeffs=[1,0,1]).
        let expected_mu0 = t.mu(0, 1, 4);
        let expected_mu1 = t.mu(1, 1, 4);

        assert_eq!(
            mus[0], expected_mu0,
            "mu[s=0] mismatch: got {:?}, expected {:?}",
            mus[0], expected_mu0
        );
        assert_eq!(
            mus[1], expected_mu1,
            "mu[s=1] mismatch: got {:?}, expected {:?}",
            mus[1], expected_mu1
        );

        // Concrete cross-check against golden kl_B2_w2_1.json:
        // mues[0][0] = {"c":[1,0,1],"v":-1} = v^{-1} + 0·v^0 + v^1.
        // mu(s=0, y=1, w=4) should be non-zero.
        assert!(
            !expected_mu0.is_zero(),
            "golden: mu(s=0, y=1, w=4) should be non-zero (= v^{{-1}}+v)"
        );
        // mu(s=1, y=1, w=4) should be zero (NO_MU in golden).
        assert!(
            expected_mu1.is_zero(),
            "golden: mu(s=1, y=1, w=4) should be zero"
        );
    }
}
