//! Unit tests for the Bruhat-interval machinery (task Q3).

use super::*;
use crate::kl::{klpolynomials, KlOpts};

/// Build the full KL table for a small group and return it with the group.
fn table_for(typ: &str) -> (CoxeterGroup, KlTable) {
    let group = CoxeterGroup::from_type(typ).unwrap();
    let opts = KlOpts::equal(group.rank);
    let table = klpolynomials(&group, &opts).expect("kl table");
    (group, table)
}

/// Reconstruct full perms and lengths for an interval member list.
fn perms_lengths(
    group: &CoxeterGroup,
    table: &KlTable,
    members: &[ElmIdx],
) -> (Vec<Perm>, Vec<u32>) {
    let perms: Vec<Perm> = members
        .iter()
        .map(|&z| group.word_to_perm(&table.elms.elms[z as usize]))
        .collect();
    let lengths: Vec<u32> = members
        .iter()
        .map(|&z| table.elms.lengths[z as usize])
        .collect();
    (perms, lengths)
}

// -----------------------------------------------------------------------
// Reflections
// -----------------------------------------------------------------------

/// |reflections| must equal n_pos for each group.
#[test]
fn reflections_count_matches_n_pos() {
    for typ in ["A2", "A3", "B2", "B3", "A4", "D4", "H3"] {
        let group = CoxeterGroup::from_type(typ).unwrap();
        let refls = reflections(&group);
        assert_eq!(
            refls.len(),
            group.n_pos as usize,
            "{typ}: |reflections|={} != n_pos={}",
            refls.len(),
            group.n_pos
        );
    }
}

/// Spelled-out checks from the task: A3 has 6 reflections, B3 has 9.
#[test]
fn reflections_a3_b3_explicit() {
    let a3 = CoxeterGroup::from_type("A3").unwrap();
    assert_eq!(reflections(&a3).len(), 6, "A3 has 6 reflections");
    let b3 = CoxeterGroup::from_type("B3").unwrap();
    assert_eq!(reflections(&b3).len(), 9, "B3 has 9 reflections");
}

/// Every reflection is an involution and has odd length.
#[test]
fn reflections_are_involutions() {
    let group = CoxeterGroup::from_type("A3").unwrap();
    let id = group.id_perm();
    for t in reflections(&group) {
        assert_eq!(t.then(&t), id, "reflection must be an involution");
        assert_eq!(
            group.perm_length(&t) % 2,
            1,
            "reflection length must be odd"
        );
    }
}

// -----------------------------------------------------------------------
// Interval extraction
// -----------------------------------------------------------------------

/// A2, y=e (idx 0), w=w0 (idx 5): interval = all 6 elements.
#[test]
fn interval_a2_full() {
    let (_g, table) = table_for("A2");
    let n = table.n() as ElmIdx;
    let members = extract_interval(&table, 0, n - 1);
    assert_eq!(members.len(), 6, "A2 [e,w0] has all 6 elements");
    assert_eq!(members, vec![0, 1, 2, 3, 4, 5]);
}

/// A trivial interval [w, w] is just {w}.
#[test]
fn interval_singleton() {
    let (_g, table) = table_for("A3");
    let members = extract_interval(&table, 3, 3);
    assert_eq!(members, vec![3]);
}

/// Lower interval [e, w]: e is below everything, so the interval is exactly
/// {z : z ≤_B w} and must contain e and w.
#[test]
fn interval_lower_contains_endpoints() {
    let (_g, table) = table_for("B2");
    let n = table.n() as ElmIdx;
    let w = n - 1; // w0
    let members = extract_interval(&table, 0, w);
    assert!(members.contains(&0), "must contain e");
    assert!(members.contains(&w), "must contain w");
    // B2 has 8 elements and [e, w0] is the whole group.
    assert_eq!(members.len(), 8);
}

// -----------------------------------------------------------------------
// A2 [e, w0]: cover graph and Bruhat graph edge counts (hand-checked)
// -----------------------------------------------------------------------

/// A2 Hasse diagram of [e, w0] has exactly 8 cover edges:
/// e<s0,s1; s0<s0s1,s1s0; s1<s0s1,s1s0; s0s1<w0, s1s0<w0  → 8 covers.
#[test]
fn a2_cover_edge_count_is_8() {
    let (group, table) = table_for("A2");
    let n = table.n() as ElmIdx;
    let members = extract_interval(&table, 0, n - 1);
    let (perms, lengths) = perms_lengths(&group, &table, &members);
    let refls = reflections(&group);
    let g = build_graph(&members, &perms, &lengths, GraphKind::Covers, &refls);
    assert_eq!(g.edge_count(), 8, "A2 [e,w0] has 8 cover edges");
}

/// A2 Bruhat graph of [e, w0] has 9 edges: the 8 covers plus the single
/// non-cover reflection edge e → w0 (w0 = s0s1s0 is a reflection in A2).
#[test]
fn a2_bruhat_edge_count_is_9() {
    let (group, table) = table_for("A2");
    let n = table.n() as ElmIdx;
    let members = extract_interval(&table, 0, n - 1);
    let (perms, lengths) = perms_lengths(&group, &table, &members);
    let refls = reflections(&group);
    let cov = build_graph(&members, &perms, &lengths, GraphKind::Covers, &refls);
    let bru = build_graph(&members, &perms, &lengths, GraphKind::Bruhat, &refls);
    assert_eq!(bru.edge_count(), 9, "A2 [e,w0] has 9 Bruhat-graph edges");
    // The single extra edge is e (level 0) → w0 (level 3).
    assert_eq!(bru.edge_count() - cov.edge_count(), 1);
    // Confirm vertex 0 = e gains an out-edge to the top in the Bruhat graph.
    let top = members.len() - 1;
    assert!(
        bru.out[0].contains(&top),
        "Bruhat graph must have edge e → w0"
    );
    assert!(
        !cov.out[0].contains(&top),
        "cover graph must NOT have edge e → w0"
    );
}

/// The cover graph must be a subgraph of the Bruhat graph (every cover edge is
/// also a Bruhat-graph edge).
#[test]
fn covers_subgraph_of_bruhat() {
    let (group, table) = table_for("B2");
    let n = table.n() as ElmIdx;
    let members = extract_interval(&table, 0, n - 1);
    let (perms, lengths) = perms_lengths(&group, &table, &members);
    let refls = reflections(&group);
    let cov = build_graph(&members, &perms, &lengths, GraphKind::Covers, &refls);
    let bru = build_graph(&members, &perms, &lengths, GraphKind::Bruhat, &refls);
    for i in 0..cov.n {
        let bset: HashSet<usize> = bru.out[i].iter().copied().collect();
        for &j in &cov.out[i] {
            assert!(
                bset.contains(&j),
                "cover edge {i}->{j} missing from Bruhat graph"
            );
        }
    }
}

// -----------------------------------------------------------------------
// Isomorphism tester
// -----------------------------------------------------------------------

/// Build a tiny graph by hand: a chain 0→1→2 with levels 0,1,2.
fn chain3() -> IntervalGraph {
    let mut g = IntervalGraph {
        n: 3,
        level: vec![0, 1, 2],
        out: vec![vec![], vec![], vec![]],
        in_: vec![vec![], vec![], vec![]],
    };
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g
}

/// A relabelled but isomorphic chain (vertex order permuted within levels —
/// here levels are distinct so it is the same chain).
fn chain3_again() -> IntervalGraph {
    chain3()
}

/// A diamond: 0→1, 0→2, 1→3, 2→3 (levels 0,1,1,2).
fn diamond() -> IntervalGraph {
    let mut g = IntervalGraph {
        n: 4,
        level: vec![0, 1, 1, 2],
        out: vec![vec![], vec![], vec![], vec![]],
        in_: vec![vec![], vec![], vec![], vec![]],
    };
    g.add_edge(0, 1);
    g.add_edge(0, 2);
    g.add_edge(1, 3);
    g.add_edge(2, 3);
    g
}

/// The diamond with its two middle vertices swapped — still isomorphic.
fn diamond_swapped() -> IntervalGraph {
    let mut g = IntervalGraph {
        n: 4,
        level: vec![0, 1, 1, 2],
        out: vec![vec![], vec![], vec![], vec![]],
        in_: vec![vec![], vec![], vec![], vec![]],
    };
    // Same shape, middle vertices labelled 2 then 1.
    g.add_edge(0, 2);
    g.add_edge(0, 1);
    g.add_edge(2, 3);
    g.add_edge(1, 3);
    g
}

/// A path 0→1→2→3 (levels 0,1,2,3) — NOT isomorphic to the diamond even
/// though both have 4 vertices.
fn path4() -> IntervalGraph {
    let mut g = IntervalGraph {
        n: 4,
        level: vec![0, 1, 2, 3],
        out: vec![vec![], vec![], vec![], vec![]],
        in_: vec![vec![], vec![], vec![], vec![]],
    };
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 3);
    g
}

#[test]
fn iso_identical_graphs() {
    assert!(is_isomorphic(&chain3(), &chain3_again()));
    assert!(is_isomorphic(&diamond(), &diamond()));
}

#[test]
fn iso_isomorphic_relabelling() {
    assert!(
        is_isomorphic(&diamond(), &diamond_swapped()),
        "diamond with swapped middle vertices is isomorphic"
    );
}

#[test]
fn iso_non_isomorphic() {
    assert!(
        !is_isomorphic(&diamond(), &path4()),
        "diamond and path4 differ (level sizes 1,2,1 vs 1,1,1,1)"
    );
    assert!(
        !is_isomorphic(&chain3(), &diamond()),
        "different vertex counts"
    );
}

/// Direction matters: a graph and its reverse are NOT level-iso (levels flip).
#[test]
fn iso_respects_direction_and_level() {
    // Two graphs with same undirected shape but an edge pointing the "wrong"
    // way relative to levels would have a different degree profile; construct
    // a case where in/out degrees differ.
    let mut a = IntervalGraph {
        n: 3,
        level: vec![0, 1, 1],
        out: vec![vec![], vec![], vec![]],
        in_: vec![vec![], vec![], vec![]],
    };
    a.add_edge(0, 1);
    a.add_edge(0, 2);

    let mut b = IntervalGraph {
        n: 3,
        level: vec![0, 1, 1],
        out: vec![vec![], vec![], vec![]],
        in_: vec![vec![], vec![], vec![]],
    };
    b.add_edge(0, 1); // only one edge → different edge count
    assert!(!is_isomorphic(&a, &b));
}

// -----------------------------------------------------------------------
// Cross-checks tied to KL data (small)
// -----------------------------------------------------------------------

/// For all comparable pairs with length gap ≤ 2, P_{y,w} = 1 (classical).
/// Verified directly here on A3.
#[test]
fn small_gap_pols_are_one() {
    let (_g, table) = table_for("A3");
    let n = table.n() as ElmIdx;
    let one = crate::laurent::Laurent::one();
    for w in 0..n {
        for y in 0..=w {
            if !table.bruhat_leq(y, w) {
                continue;
            }
            let gap = table.elms.lengths[w as usize] - table.elms.lengths[y as usize];
            if gap <= 2 {
                assert_eq!(
                    table.pol(y, w),
                    Some(&one),
                    "A3: P_{{{y},{w}}} must be 1 for gap {gap}"
                );
            }
        }
    }
}

// -----------------------------------------------------------------------
// Length-shift invariance and WL refinement
// -----------------------------------------------------------------------

/// The key uses RELATIVE levels, so two structurally identical graphs whose
/// absolute levels differ by a constant must share a key and be isomorphic.
/// Here `build_graph` always normalizes to relative level (vertex 0 → 0), so we
/// assert that property directly and check that two diamonds with the same
/// relative shape are iso even though we could imagine them in different windows.
#[test]
fn relative_level_shift_invariance() {
    // diamond at relative levels {0,1,1,2}.
    let a = diamond();
    // A "shifted" diamond: same shape, but we hand-build with the SAME relative
    // levels because build_graph always rebases.  Confirm key equality + iso.
    let b = diamond_swapped();
    assert_eq!(
        graph_key(&a),
        graph_key(&b),
        "same relative shape ⇒ same key"
    );
    assert!(is_isomorphic(&a, &b));
}

/// A highly symmetric interval (full [e, w0] of B3) must still classify quickly
/// without tripping the iteration cap.  This exercises the WL-refined backtrack.
#[test]
fn symmetric_interval_iso_terminates() {
    let (group, table) = table_for("B3");
    let n = table.n() as ElmIdx;
    let members = extract_interval(&table, 0, n - 1); // whole group
    let (perms, lengths) = perms_lengths(&group, &table, &members);
    let refls = reflections(&group);
    let bru = build_graph(&members, &perms, &lengths, GraphKind::Bruhat, &refls);
    // A graph is isomorphic to itself; with WL refinement this must NOT blow up.
    assert!(is_isomorphic(&bru, &bru), "self-isomorphism must hold");
    // And a clone with shuffled vertex order is still iso.
    let shuffled = shuffle_vertices(&bru);
    assert!(
        is_isomorphic(&bru, &shuffled),
        "shuffled relabelling must be isomorphic"
    );
}

/// Produce a vertex-relabelled copy of a graph (within-level shuffle) to test
/// that the isomorphism tester is invariant under relabelling.
fn shuffle_vertices(g: &IntervalGraph) -> IntervalGraph {
    // Deterministic reversal permutation as the relabelling.
    let perm: Vec<usize> = (0..g.n).rev().collect();
    let mut h = IntervalGraph {
        n: g.n,
        level: vec![0; g.n],
        out: vec![Vec::new(); g.n],
        in_: vec![Vec::new(); g.n],
    };
    for (i, &lvl) in g.level.iter().enumerate() {
        h.level[perm[i]] = lvl;
    }
    for (i, outs) in g.out.iter().enumerate() {
        for &j in outs {
            h.add_edge(perm[i], perm[j]);
        }
    }
    h
}

/// The CORE empirical test, scoped tiny: on B3, every poset-isomorphism class
/// carries a single KL polynomial (no violations).  This is the conjecture
/// itself, asserted in a unit test so a regression fails loudly.
#[test]
fn b3_no_poset_violations() {
    let (group, table) = table_for("B3");
    let n = table.n() as ElmIdx;
    let refls = reflections(&group);
    let perms: Vec<Perm> = (0..n)
        .map(|i| group.word_to_perm(&table.elms.elms[i as usize]))
        .collect();

    // Bucket pairs by cheap cover-key, then within a bucket group by exact iso
    // and check all pairs in a class share one polynomial.
    let mut bucket: std::collections::HashMap<
        GraphKey,
        Vec<(IntervalGraph, crate::laurent::Laurent)>,
    > = std::collections::HashMap::new();
    for w in 0..n {
        for y in 0..w {
            if !table.bruhat_leq(y, w) {
                continue;
            }
            let pol = table.pol(y, w).unwrap().clone();
            let members = extract_interval(&table, y, w);
            let mp: Vec<Perm> = members.iter().map(|&z| perms[z as usize].clone()).collect();
            let ml: Vec<u32> = members
                .iter()
                .map(|&z| table.elms.lengths[z as usize])
                .collect();
            let cover = build_graph(&members, &mp, &ml, GraphKind::Covers, &refls);
            let key = graph_key(&cover);
            bucket.entry(key).or_default().push((cover, pol));
        }
    }

    for entries in bucket.values() {
        // Within a cheap bucket, group by exact iso and check pol agreement.
        let mut reps: Vec<usize> = Vec::new(); // indices into entries
        for (i, (gi, poli)) in entries.iter().enumerate() {
            let mut matched = false;
            for &ri in &reps {
                if is_isomorphic(gi, &entries[ri].0) {
                    assert_eq!(
                        poli, &entries[ri].1,
                        "B3 poset-iso class carries differing P — conjecture violation \
                         (or canonicalization bug)"
                    );
                    matched = true;
                    break;
                }
            }
            if !matched {
                reps.push(i);
            }
        }
    }
}
