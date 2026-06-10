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
//! each generator `s`: if `weights[s] == 0` (unconditionally, even when
//! `s ∈ I(u)`) OR (`s ∉ I(u)` ascent and `weights[s] > 0`), and
//! `s·u ∈ self.vertices`, add arrow `pos → pos_of(s·u)`.  This mirrors
//! `build_arrows`'s condition `weight == 0 || (sw > wu && weight > 0)` exactly.

use std::collections::{HashMap, HashSet};

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
    /// Generator weights `L(s)` copied from the KL table.  Used by `decompose`
    /// to determine whether a weight-0 type-2 arrow must be emitted.
    weights: Vec<u32>,
}

impl WGraph {
    /// Build the W-graph of `cell` (element indices) from the KL table.
    ///
    /// Complexity: O(n²·rank) for the edge scan, where n = `cell.len()` and
    /// rank = number of generators.  Suitable for single cells; for the full
    /// group (n = |W|) the quadratic scan dominates — use only when necessary.
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

        let weights = t.weights.clone();

        WGraph {
            vertices,
            isets,
            edges,
            lft_cell,
            rank,
            weights,
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
    /// generator `s`: if `weights[s] == 0` (unconditionally) OR (`s ∉ I(u)` and
    /// `weights[s] > 0`), and `s·u ∈ self.vertices`, add `pos → pos_of(s·u)`.
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
        // Use per-row HashSets for O(1) dedup during construction, then discard
        // them before calling tarjan_scc (which only needs Vec<Vec<u32>>).
        let mut adj_sets: Vec<HashSet<u32>> = vec![HashSet::new(); n];

        // Type 1: mu-based arrows.
        //
        // Edge (pi, pj) with pi < pj stores mus for (y=vertices[pi], w=vertices[pj]).
        // build_arrows condition: s ∈ I(y=pi) AND s ∉ I(w=pj) → arrow w→y = pj→pi.
        //
        // The reverse (s ∈ I(w) AND s ∉ I(y)) is intentionally absent here.
        // PyCox `klpolynomials` arrows block (pycox-ref/pycox_ref.py ≈10387–10394)
        // only appends (w, y) when `s ∈ Isets[y=smaller]` AND `s ∉ Isets[w=larger]`,
        // i.e. strictly asymmetric: only the Bruhat-smaller endpoint owning the
        // descent generates a type-1 arrow (pointing toward it from the larger end).
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
                    adj_sets[pj].insert(pi as u32);
                }
            }
        }

        // Type 2: direct s·u arrows (matches build_arrows' (wu, sw) emission).
        //
        // build_arrows emits (wu, sw) when:
        //   weight == 0   — unconditionally, even if s ∈ I(u) (descent), OR
        //   sw > wu (ascent, s ∉ I(u)) AND weight > 0.
        //
        // Mirror that condition here.
        for (pos, adj_row) in adj_sets.iter_mut().enumerate() {
            for s in 0..self.rank {
                let s_gen = s as Gen;
                let weight = self.weights[s];
                let is_descent = self.isets[pos].contains(&s_gen);
                // Emit arrow when weight == 0 (unconditional) OR weight > 0 AND ascent.
                if weight == 0 || !is_descent {
                    let sv_pos = self.lft_cell[pos * self.rank + s];
                    if sv_pos != u32::MAX {
                        adj_row.insert(sv_pos);
                    }
                }
            }
        }

        // Convert to Vec<Vec<u32>> and drop the HashSets before calling tarjan_scc.
        let adj: Vec<Vec<u32>> = adj_sets
            .into_iter()
            .map(|s| s.into_iter().collect())
            .collect();

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
                // pi < pj (invariant of self.edges), and sorted_pos is sorted
                // ascending, so the position-sorted order is preserved: ni < nj.
                debug_assert!(
                    ni < nj,
                    "edge ({pi},{pj}) mapped to ({ni},{nj}): ni must be < nj"
                );
                edges.insert((ni, nj), mus.clone());
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
            weights: self.weights.clone(),
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
            assert_eq!(
                comps[0].vertices, wg.vertices,
                "B3 cell {cell_idx}: single component must cover all vertices"
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

    // -----------------------------------------------------------------------
    // Test 5: b2_w0_1_wgraph_decompose
    // -----------------------------------------------------------------------
    /// B2 with weights [0, 1]: build the full-group W-graph (all 8 elements)
    /// and verify that decompose() produces component vertex-sets equal to
    /// CellData::from_table lcells (both canonically sorted).
    ///
    /// With weight-0 generators the type-2 arrow condition must emit arrows
    /// even for descents (weight == 0 case in build_arrows), which is the
    /// bug fixed in this patch.
    #[test]
    fn b2_w0_1_wgraph_decompose() {
        let (_, t) = build_table("B2", vec![0, 1]);
        let cd = CellData::from_table(&t);
        let n = t.n();

        let all: Vec<ElmIdx> = (0..n as u32).collect();
        let full_wg = WGraph::of_cell(&t, &all);
        let mut comps = full_wg.decompose();

        // Canonical sort for comparison.
        comps.sort_by(|a, b| a.vertices.cmp(&b.vertices));
        let mut lcells = cd.lcells.clone();
        lcells.sort();

        assert_eq!(
            comps.len(),
            lcells.len(),
            "B2[0,1]: number of decompose() components ({}) != number of lcells ({})",
            comps.len(),
            lcells.len()
        );

        for (c, (comp, cell)) in comps.iter().zip(lcells.iter()).enumerate() {
            assert_eq!(
                comp.vertices, *cell,
                "B2[0,1] component {c}: vertices {:?} != lcell {:?}",
                comp.vertices, cell
            );
        }
    }
}
