//! Kazhdan–Lusztig star operations, star orbits, and Vogan's generalised
//! τ-invariant.
//!
//! This module ports the following PyCox functions (all in `pycox_ref.py`):
//!
//! | PyCox function | Rust function | Lines |
//! |---|---|---|
//! | `klstaroperation` | [`star_op_right`] | 11359–11397 |
//! | `klstarorbitperm` | [`star_orbit_right`] | 11439–11459 |
//! | `leftklstar` | [`left_star`] | 11482–11501 |
//! | `leftklstarorbitelm` | [`left_star_orbit_elm`] | 11504–11525 |
//! | `generalisedtau` | [`generalised_tau`] | 11670–11689 |
//!
//! ## Conventions
//!
//! Throughout this module, **right-multiplication** of permutation `pw` by
//! generator `s` is `pw.then(&g.permgens[s])`, matching PyCox's
//! `tuple([W.permgens[s][r] for r in pw])`.
//!
//! **Left-multiplication** is `g.permgens[s].then(&pw)`, matching PyCox's
//! `tuple([pw[i] for i in W.permgens[s]])`.
//!
//! **Right descent test**: `s` is a right descent of `p` iff
//! `p.inverse()[s] >= N`.  For irreducible groups (where `simple_root[s] == s`)
//! this is the direct test used by PyCox.  We use `g.right_descents(p)` or
//! the inlined check `inv.0[g.simple_root[s]] as usize >= n_pos` to stay
//! correct for reducible groups too.
//!
//! **CoxElm deduplication**: orbit cells are tracked by the `CoxElm` of their
//! first element (images of simple roots under `simple_root`).

use std::collections::HashSet;

use crate::{
    element::{CoxElm, Gen, Perm},
    group::CoxeterGroup,
};

// ---------------------------------------------------------------------------
// Helper: is generator `s` a right descent of perm `p`?
// ---------------------------------------------------------------------------

/// Returns `true` iff `s` is a right descent of `p`.
///
/// Equivalent to PyCox `perminverse(p)[s] >= W.N` (for simple-root index s).
/// We use `simple_root[s]` to handle reducible groups correctly.
#[inline]
fn is_right_descent(g: &CoxeterGroup, p: &Perm, s: usize) -> bool {
    // s is a right descent iff simple_root[s] is sent to a negative root by
    // p, i.e. p.inverse()[simple_root[s]] >= n_pos.
    // Equivalently: the preimage of simple_root[s] under p lies in [N, 2N).
    let r = g.simple_root[s];
    let n = g.n_pos as usize;
    // Find j in [N, 2N) such that p.0[j] == r.
    (n..2 * n).any(|j| p.0[j] as usize == r)
}

// ---------------------------------------------------------------------------
// Right star operation
// ---------------------------------------------------------------------------

/// Right KL star operation w.r.t. generators `(s, t)` with `m_st = 3`,
/// applied to a *cell* (a slice of permutations all sharing one right descent
/// set).
///
/// Returns `None` iff the operation is undefined for this cell — i.e. iff
/// both `s` and `t` are right descents of `cell[0]`, or neither is.  The XOR
/// gate is tested on the first element only (all elements share the same right
/// descent set by assumption).
///
/// ## Mapping rule (per element)
///
/// Given element `pw`, compute `ws = pw · s` (right-multiply by `s`).
/// Check whether exactly one of `{s, t}` is a right descent of `ws`:
/// - If yes → keep `ws`.
/// - If no (neither or both) → take `wt = pw · t` instead.
///
/// ## PyCox correspondence
///
/// Ports `klstaroperation(W, s, t, pcell)` (lines 11359–11397).
/// Right-multiplication: `pw.then(&g.permgens[s])`.
pub fn star_op_right(g: &CoxeterGroup, s: Gen, t: Gen, cell: &[Perm]) -> Option<Vec<Perm>> {
    if cell.is_empty() {
        return Some(Vec::new());
    }

    // Gate: check the first element's right descent set.
    let pw0 = &cell[0];
    let s_in = is_right_descent(g, pw0, s as usize);
    let t_in = is_right_descent(g, pw0, t as usize);

    // Undefined iff both in or both out.
    if s_in == t_in {
        return None;
    }

    let mut result = Vec::with_capacity(cell.len());
    for pw in cell {
        // Right-multiply by s.
        let ws = pw.then(&g.permgens[s as usize]);
        // Check if exactly one of {s,t} is a right descent of ws.
        let ws_s = is_right_descent(g, &ws, s as usize);
        let ws_t = is_right_descent(g, &ws, t as usize);
        if ws_s != ws_t {
            // Exactly one → keep ws.
            result.push(ws);
        } else {
            // Neither or both → take wt = pw · t.
            result.push(pw.then(&g.permgens[t as usize]));
        }
    }
    Some(result)
}

// ---------------------------------------------------------------------------
// Right star orbit of a cell (BFS)
// ---------------------------------------------------------------------------

/// Right star orbit of a cell under all `(s, t)` pairs with `m_st = 3`.
///
/// Returns the list of orbit cells (the input cell first), where each cell is
/// a list of full permutations.
///
/// ## Deduplication
///
/// A candidate new cell `nc` is included iff its first element's `CoxElm` is
/// not already present in the union of `CoxElm` sets of all known orbit
/// members.  More precisely, for each existing orbit cell `orb[k]`,
/// `orb1[k]` is the `HashSet<CoxElm>` of all elements of `orb[k]`; the new
/// cell is admitted iff `nc[0].coxelm_sr(simple_root)` is not in any
/// `orb1[k]`.
///
/// ## PyCox correspondence
///
/// Ports `klstarorbitperm(W, l)` (lines 11439–11459).
pub fn star_orbit_right(g: &CoxeterGroup, cell: &[Perm]) -> Vec<Vec<Perm>> {
    // orb: the orbit cells (full perms)
    let mut orb: Vec<Vec<Perm>> = vec![cell.to_vec()];
    // orb1: CoxElm set of each orbit member
    let first_set: HashSet<CoxElm> = cell.iter().map(|p| p.coxelm_sr(&g.simple_root)).collect();
    let mut orb1: Vec<HashSet<CoxElm>> = vec![first_set];

    // BFS: iterate over orb (orb may grow during iteration).
    let mut i = 0;
    while i < orb.len() {
        let rank = g.rank;
        // Iterate pairs (s, t) with s > t and m_st == 3 (matching PyCox's
        // `for s in range(len(gens)): for t in range(s): ...`).
        for s in 0..rank {
            for t in 0..s {
                if g.coxmat[s][t] != 3 {
                    continue;
                }
                // PyCox calls klstaroperation(W, gens[s], gens[t], cell) once
                // per s>t pair; klstaroperation is not symmetric in (s,t).
                // Replicating the exact PyCox loop: one call per s>t pair.
                // The BFS will expand each found cell the same way, so all
                // reachable orbit members are found.
                let cell_i = orb[i].clone();
                if let Some(nc) = star_op_right(g, s as Gen, t as Gen, &cell_i) {
                    let nc_first_ce = nc[0].coxelm_sr(&g.simple_root);
                    // Admit iff nc's first-element CoxElm is absent from all
                    // known orbit members.
                    if !orb1.iter().any(|o| o.contains(&nc_first_ce)) {
                        let nc_set: HashSet<CoxElm> =
                            nc.iter().map(|p| p.coxelm_sr(&g.simple_root)).collect();
                        orb.push(nc);
                        orb1.push(nc_set);
                    }
                }
            }
        }
        i += 1;
    }

    orb
}

// ---------------------------------------------------------------------------
// Left star operation (single element)
// ---------------------------------------------------------------------------

/// Left KL star operation w.r.t. generators `(s, t)` with `m_st = 3`,
/// applied to a single element `p`.
///
/// It is the **caller's responsibility** to verify that the operation applies
/// (i.e. exactly one of `{s, t}` is a left descent of `p`).
///
/// ## Mapping rule
///
/// - If `s` is a left descent and `t` is not:
///   - compute `sw = s · p` (left-multiply by s); check if `t` is a left
///     descent of `sw`.  If yes → return `sw`; otherwise return `t · p`.
/// - Else (`t` is a left descent and `s` is not):
///   - compute `tw = t · p`; check if `s` is a left descent of `tw`.  If
///     yes → return `tw`; otherwise return `s · p`.
///
/// ## PyCox correspondence
///
/// Ports `leftklstar(W, pw, s, t)` (lines 11482–11501).
/// Left-multiplication: `g.permgens[s].then(&pw)` matches
/// `tuple([pw[i] for i in W.permgens[s]])`.
/// Left-descent test: `pw[simple_root[s]] >= n_pos`.
pub fn left_star(g: &CoxeterGroup, p: &Perm, s: Gen, t: Gen) -> Perm {
    let n = g.n_pos as usize;
    let sr_s = g.simple_root[s as usize];
    let sr_t = g.simple_root[t as usize];

    // Left-descent test: p[simple_root[s]] >= n_pos.
    let s_is_ldesc = p.0[sr_s] as usize >= n;
    // t must be the non-descent (caller ensures XOR).

    if s_is_ldesc {
        // sw = s · p
        let sw = g.permgens[s as usize].then(p);
        // Check if t is a left descent of sw.
        if sw.0[sr_t] as usize >= n {
            sw
        } else {
            g.permgens[t as usize].then(p)
        }
    } else {
        // t is left descent; tw = t · p
        let tw = g.permgens[t as usize].then(p);
        // Check if s is a left descent of tw.
        if tw.0[sr_s] as usize >= n {
            tw
        } else {
            g.permgens[s as usize].then(p)
        }
    }
}

// ---------------------------------------------------------------------------
// Left star orbit of a single element (BFS)
// ---------------------------------------------------------------------------

/// Left star orbit of a single element `p` under all `(s, t)` pairs with
/// `m_st = 3`.
///
/// Returns the orbit as a list of full permutations (the input element first).
/// Deduplication is by `CoxElm` (images of simple roots).
///
/// ## PyCox correspondence
///
/// Ports `leftklstarorbitelm(W, pw)` (lines 11504–11525).
pub fn left_star_orbit_elm(g: &CoxeterGroup, p: &Perm) -> Vec<Perm> {
    let mut orb: Vec<Perm> = vec![p.clone()];
    let mut orb1: HashSet<CoxElm> = {
        let mut s = HashSet::new();
        s.insert(p.coxelm_sr(&g.simple_root));
        s
    };
    let n = g.n_pos as usize;

    // BFS: iterate over orb (orb may grow during iteration).
    let mut o = 0;
    while o < orb.len() {
        let d = orb[o].clone();
        let rank = g.rank;
        for i in 0..rank {
            for j in 0..i {
                let s = i as Gen;
                let t = j as Gen;
                if g.coxmat[s as usize][t as usize] != 3 {
                    continue;
                }
                let sr_s = g.simple_root[s as usize];
                let sr_t = g.simple_root[t as usize];
                // Gate: exactly one of {s,t} is a left descent of d.
                let s_in = d.0[sr_s] as usize >= n;
                let t_in = d.0[sr_t] as usize >= n;
                if s_in == t_in {
                    continue; // both in or both out → undefined
                }
                let d1 = left_star(g, &d, s, t);
                let ce = d1.coxelm_sr(&g.simple_root);
                if !orb1.contains(&ce) {
                    orb1.insert(ce);
                    orb.push(d1);
                }
            }
        }
        o += 1;
    }

    orb
}

// ---------------------------------------------------------------------------
// Vogan's generalised τ-invariant
// ---------------------------------------------------------------------------

/// Vogan's generalised τ-invariant of element `p`.
///
/// BFS orbit of the single element `p` under right star operations, capped at
/// `maxd` orbit members.  Returns the sequence of right-descent sets (as
/// sorted `Vec<Gen>`) of all orbit members, in BFS order.
///
/// Two elements in the same left cell (equal-parameter case) have identical
/// generalised τ-invariants.
///
/// ## Key implementation detail
///
/// PyCox does **not** deduplicate when appending to the orbit.  Each
/// applicable `star_op_right` result for `[orb[o]]` is appended
/// unconditionally, possibly producing duplicates.  We replicate this exactly.
///
/// The cap is `while o < len(orb) <= maxd` — the loop exits as soon as either
/// `o == len(orb)` (orbit saturated) or `len(orb) > maxd` (cap exceeded).
///
/// ## PyCox correspondence
///
/// Ports `generalisedtau(W, pw, maxd)` (lines 11670–11689).
pub fn generalised_tau(g: &CoxeterGroup, p: &Perm, maxd: usize) -> Vec<Vec<Gen>> {
    let mut orb: Vec<Perm> = vec![p.clone()];
    let mut o = 0;
    while o < orb.len() && orb.len() <= maxd {
        let rank = g.rank;
        for s in 0..rank {
            for t in 0..s {
                if g.coxmat[s][t] != 3 {
                    continue;
                }
                // Apply star_op to the singleton [orb[o]].
                let d = orb[o].clone();
                if let Some(k) = star_op_right(g, s as Gen, t as Gen, &[d]) {
                    // Append unconditionally (no dedup — matches PyCox exactly).
                    orb.push(k.into_iter().next().unwrap());
                }
            }
        }
        o += 1;
    }

    // Return the sequence of right-descent sets.
    orb.iter().map(|q| g.right_descents(q)).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kl::{klpolynomials, CellData, KlOpts};
    use std::collections::HashSet;

    // -----------------------------------------------------------------------
    // Helper: build the KL table and CellData for a given group type.
    // -----------------------------------------------------------------------
    fn kl_cells(type_str: &str) -> (CoxeterGroup, Vec<Vec<Perm>>) {
        let g = CoxeterGroup::from_type(type_str).unwrap();
        let opts = KlOpts::equal(g.rank);
        let table = klpolynomials(&g, &opts).unwrap();
        let cd = CellData::from_table(&table);

        // Convert left-cell element-index lists to Perm lists.
        // table.elms.elms contains the canonical words in canonical order.
        let cells_as_perms: Vec<Vec<Perm>> = cd
            .lcells
            .iter()
            .map(|cell| {
                cell.iter()
                    .map(|&idx| g.word_to_perm(&table.elms.elms[idx as usize]))
                    .collect()
            })
            .collect();

        (g, cells_as_perms)
    }

    // -----------------------------------------------------------------------
    // Test 1: star_orbit_partitions_cells
    //
    // For A3 and B3: take the full-table left cells (as permutation lists).
    // Assert:
    //   (a) Every orbit member equals (as a CoxElm set) some full-table cell.
    //   (b) The union of orbits over all cells covers all cells.
    //   (c) Orbit of any orbit member = same orbit set.
    // -----------------------------------------------------------------------
    #[test]
    fn star_orbit_partitions_cells_a3() {
        let (g, cells) = kl_cells("A3");
        check_orbit_partitions_cells(&g, &cells);
    }

    #[test]
    fn star_orbit_partitions_cells_b3() {
        let (g, cells) = kl_cells("B3");
        check_orbit_partitions_cells(&g, &cells);
    }

    fn check_orbit_partitions_cells(g: &CoxeterGroup, cells: &[Vec<Perm>]) {
        // Build a lookup: CoxElm-set (of a cell) → cell index.
        // We represent a CoxElm-set by a sorted Vec<CoxElm>.
        let cell_keys: Vec<Vec<CoxElm>> = cells
            .iter()
            .map(|c| {
                let mut v: Vec<CoxElm> = c.iter().map(|p| p.coxelm_sr(&g.simple_root)).collect();
                v.sort_by_key(|ce| ce.0.to_vec());
                v
            })
            .collect();

        let cell_key_set: HashSet<Vec<Vec<u32>>> = cell_keys
            .iter()
            .map(|v| v.iter().map(|ce| ce.0.to_vec()).collect())
            .collect();

        // Track which cells have been seen as orbit members.
        let mut covered: HashSet<usize> = HashSet::new();

        for (ci, cell) in cells.iter().enumerate() {
            let orbit = star_orbit_right(g, cell);

            // (a) Every orbit member must equal some full-table cell.
            for oc in &orbit {
                let mut oc_key: Vec<Vec<u32>> = oc
                    .iter()
                    .map(|p| p.coxelm_sr(&g.simple_root).0.to_vec())
                    .collect();
                oc_key.sort();
                assert!(
                    cell_key_set.contains(&oc_key),
                    "Orbit member of cell {ci} is not a full-table cell"
                );
            }

            // Mark orbit members as covered.
            for oc in &orbit {
                let mut oc_key: Vec<Vec<u32>> = oc
                    .iter()
                    .map(|p| p.coxelm_sr(&g.simple_root).0.to_vec())
                    .collect();
                oc_key.sort();
                let idx = cell_keys
                    .iter()
                    .position(|ck| ck.iter().map(|c| c.0.to_vec()).collect::<Vec<_>>() == oc_key)
                    .unwrap();
                covered.insert(idx);
            }
        }

        // (b) All cells are covered.
        assert_eq!(
            covered.len(),
            cells.len(),
            "Not all cells covered: {}/{} covered",
            covered.len(),
            cells.len()
        );

        // (c) Orbit closure: orbit of any orbit member = same orbit.
        for cell in cells.iter().take(cells.len().min(5)) {
            let orbit1 = star_orbit_right(g, cell);
            for oc in &orbit1 {
                let orbit2 = star_orbit_right(g, oc);
                // The orbit sets (by first-element CoxElm) should be equal.
                let key1: HashSet<Vec<u32>> = orbit1
                    .iter()
                    .map(|c| c[0].coxelm_sr(&g.simple_root).0.to_vec())
                    .collect();
                let key2: HashSet<Vec<u32>> = orbit2
                    .iter()
                    .map(|c| c[0].coxelm_sr(&g.simple_root).0.to_vec())
                    .collect();
                assert_eq!(key1, key2, "Orbit closure violated");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: star_op_gate
    //
    // B2 has coxmat = [[1,4],[4,1]] — NO m=3 pairs → star_orbit of any cell
    // is just {cell}.
    //
    // A2: m=3 (the only pair). Hand-check a concrete star op result.
    // -----------------------------------------------------------------------
    #[test]
    fn star_op_gate_b2_no_m3() {
        let g = CoxeterGroup::from_type("B2").unwrap();
        let opts = KlOpts::equal(g.rank);
        let table = klpolynomials(&g, &opts).unwrap();
        let cd = CellData::from_table(&table);

        // Every orbit must be a singleton (just the input cell).
        for cell_idx in &cd.lcells {
            let cell_perms: Vec<Perm> = cell_idx
                .iter()
                .map(|&i| g.word_to_perm(&table.elms.elms[i as usize]))
                .collect();
            let orbit = star_orbit_right(&g, &cell_perms);
            assert_eq!(
                orbit.len(),
                1,
                "B2 orbit should be trivial (no m=3 pairs), got orbit of size {}",
                orbit.len()
            );
        }
    }

    #[test]
    fn star_op_gate_a2_explicit() {
        // A2: generators 0, 1; coxmat[0][1] = 3.
        // Hand-check: cell [s0] has right descent {0}.
        // star w.r.t. (s=1, t=0) (since s > t in the loop):
        //   pw = s0 = word [0], right descents = {0}.
        //   Gate: s=1 not in descent, t=0 in descent → XOR passes.
        //   ws = s0 · s1, right descents of (s0·s1): check.
        //   s0·s1 in A2: length 2, right descents = {1}. Exactly one of {1,0} → take ws.
        //   Result: [s0·s1] = [word [0,1]].
        let g = CoxeterGroup::from_type("A2").unwrap();
        let s0 = g.word_to_perm(&[0]);
        // cell = [s0], apply star (s=1, t=0).
        let result = star_op_right(&g, 1, 0, &[s0]);
        assert!(result.is_some(), "Star op should be defined for s0 in A2");
        let mapped = result.unwrap();
        assert_eq!(mapped.len(), 1);
        let expected = g.word_to_perm(&[0, 1]);
        assert_eq!(
            mapped[0], expected,
            "star_op(A2, s=1, t=0, [s0]) should give s0·s1"
        );
    }

    #[test]
    fn star_op_gate_a2_undefined() {
        // Cell [e] has right descent {} — both 0 and 1 absent → None.
        let g = CoxeterGroup::from_type("A2").unwrap();
        let id = g.id_perm();
        let result = star_op_right(&g, 1, 0, &[id]);
        assert!(
            result.is_none(),
            "Star op should be undefined for identity in A2"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: generalised_tau_cell_invariant
    //
    // For A4: for every left cell, all members have THE SAME generalised_tau
    // (with maxd = 3 * rank).
    // -----------------------------------------------------------------------
    #[test]
    fn generalised_tau_cell_invariant_a4() {
        let (g, cells) = kl_cells("A4");
        let maxd = 3 * g.rank;
        for (ci, cell) in cells.iter().enumerate() {
            if cell.is_empty() {
                continue;
            }
            let tau0 = generalised_tau(&g, &cell[0], maxd);
            for (ei, p) in cell.iter().enumerate().skip(1) {
                let tau_i = generalised_tau(&g, p, maxd);
                assert_eq!(
                    tau0, tau_i,
                    "A4 cell {ci}: element {ei} has different generalised_tau"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: left_right_mirror
    //
    // For B3: left_star_orbit_elm(p) == {inverse(q) : q in right-star orbit
    // of inverse(p)}.
    //
    // Also check the single-step version: for an applicable (s,t),
    //   left_star(p, s, t) == inverse(right_star(inverse(p), s, t)[0])
    // -----------------------------------------------------------------------
    #[test]
    fn left_right_mirror_b3() {
        let g = CoxeterGroup::from_type("B3").unwrap();

        // Test on a sample of words.
        let test_words: Vec<&[Gen]> = vec![
            &[0],
            &[1],
            &[2],
            &[0, 1],
            &[1, 2],
            &[0, 1, 2],
            &[0, 1, 2, 0],
            &[1, 2, 1],
            &[0, 2],
        ];

        for word in test_words {
            let p = g.word_to_perm(word);

            // Left star orbit of p.
            let left_orbit_p: HashSet<Vec<u32>> = left_star_orbit_elm(&g, &p)
                .iter()
                .map(|q| q.coxelm_sr(&g.simple_root).0.to_vec())
                .collect();

            // Right star orbit of p.inverse(), then invert each member.
            let p_inv = p.inverse();
            let right_orbit_pinv = star_orbit_right(&g, &[p_inv]);
            // Take the union of all elements across all orbit cells, invert them.
            let right_orbit_pinv_inverted: HashSet<Vec<u32>> = right_orbit_pinv
                .iter()
                .flat_map(|cell| cell.iter())
                .map(|q| q.inverse().coxelm_sr(&g.simple_root).0.to_vec())
                .collect();

            assert_eq!(
                left_orbit_p, right_orbit_pinv_inverted,
                "left/right mirror mismatch for word {:?}",
                word
            );
        }
    }

    #[test]
    fn left_right_single_step_b3() {
        // For B3, for applicable (s,t), check:
        //   left_star(p, s, t) == inverse(star_op_right(inverse(p), s, t)[0])
        let g = CoxeterGroup::from_type("B3").unwrap();
        let rank = g.rank;

        let words: Vec<Vec<Gen>> = vec![
            vec![0, 1, 2],
            vec![1, 2, 1],
            vec![0, 1, 2, 0, 1],
            vec![2, 1, 0, 2],
        ];

        for word in &words {
            let p = g.word_to_perm(word);
            let p_inv = p.inverse();
            let n = g.n_pos as usize;

            for s in 0..rank {
                for t in 0..s {
                    if g.coxmat[s][t] != 3 {
                        continue;
                    }
                    let sr_s = g.simple_root[s];
                    let sr_t = g.simple_root[t];

                    // Check if left star is applicable to p.
                    let s_ldesc = p.0[sr_s] as usize >= n;
                    let t_ldesc = p.0[sr_t] as usize >= n;
                    if s_ldesc == t_ldesc {
                        continue; // not applicable
                    }

                    let l = left_star(&g, &p, s as Gen, t as Gen);

                    // Right star on p.inverse(), then invert.
                    let r_opt = star_op_right(&g, s as Gen, t as Gen, std::slice::from_ref(&p_inv));
                    // If right star is applicable to p.inverse()...
                    if let Some(r) = r_opt {
                        let r_inv = r[0].inverse();
                        assert_eq!(
                            l.coxelm_sr(&g.simple_root),
                            r_inv.coxelm_sr(&g.simple_root),
                            "left/right single-step mismatch for word {:?} s={} t={}",
                            word,
                            s,
                            t
                        );
                    }
                    // (If right star is not applicable on p.inverse() but left star
                    // is applicable on p, that is a discrepancy worth checking too —
                    // but PyCox's klstarorbitelm just uses left star on inverse, so
                    // applicability should match.)
                }
            }
        }
    }
}
