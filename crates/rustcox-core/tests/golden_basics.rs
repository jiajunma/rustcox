//! Integration tests: verify Cartan/Coxeter/degree data against golden files.
//!
//! Each golden file encodes the authoritative `coxetermat`, `degrees`, and
//! `order` for a finite Coxeter group.  This test suite builds the data with
//! `cartan_mat` + `coxeter_mat_from_cartan` + `degrees_of` + `order_from_degrees`
//! and asserts exact agreement.

mod common;

use rustcox_core::cartan::{cartan_mat, coxeter_mat_from_cartan, degrees_of, order_from_degrees};
use rustcox_core::roots;

/// Names of all golden basics files that can be tested without CycInt (Task 18).
const BASICS_NAMES: &[&str] = &[
    "basics_A1",
    "basics_A2",
    "basics_A3",
    "basics_A4",
    "basics_A5",
    "basics_B2",
    "basics_B3",
    "basics_B4",
    "basics_C3",
    "basics_D4",
    "basics_D5",
    "basics_E6",
    "basics_F4",
    "basics_G2",
    "basics_H3",
    "basics_H4",
    "basics_I5",
];

#[test]
fn cartan_data() {
    for name in BASICS_NAMES {
        let g = common::golden(name);
        let components = common::components_of(&g);

        // ---- Coxeter matrix ------------------------------------------------
        // All committed basics are single-component, so the Coxeter matrix
        // of the group equals the Coxeter matrix of that component directly.
        assert_eq!(
            components.len(),
            1,
            "{name}: expected single component, got {}",
            components.len()
        );
        let (series, rank) = components[0];
        let cartan = cartan_mat(series, rank)
            .unwrap_or_else(|e| panic!("{name}: cartan_mat({series:?}, {rank}) failed: {e}"));
        let cox = coxeter_mat_from_cartan(&cartan);

        let golden_cox: Vec<Vec<u32>> = serde_json::from_value(g["coxetermat"].clone())
            .unwrap_or_else(|e| panic!("{name}: failed to parse golden coxetermat: {e}"));

        assert_eq!(
            cox, golden_cox,
            "{name}: coxeter matrix mismatch\n  got:      {cox:?}\n  expected: {golden_cox:?}"
        );

        // ---- Degrees -------------------------------------------------------
        let mut degrees = degrees_of(series, rank)
            .unwrap_or_else(|e| panic!("{name}: degrees_of({series:?}, {rank}) failed: {e}"));
        degrees.sort_unstable();

        let mut golden_degrees: Vec<u32> = serde_json::from_value(g["degrees"].clone())
            .unwrap_or_else(|e| panic!("{name}: failed to parse golden degrees: {e}"));
        golden_degrees.sort_unstable();

        assert_eq!(
            degrees, golden_degrees,
            "{name}: degrees mismatch\n  got:      {degrees:?}\n  expected: {golden_degrees:?}"
        );

        // ---- Order ---------------------------------------------------------
        let order = order_from_degrees(&degrees);
        let golden_order = g["order"]
            .as_u64()
            .unwrap_or_else(|| panic!("{name}: golden \"order\" is not a u64"))
            as u128;

        assert_eq!(
            order, golden_order,
            "{name}: order mismatch: got {order}, expected {golden_order}"
        );
    }
}

/// Verify root systems against golden files.
///
/// For every basics golden file, builds the root system and checks:
/// - `n_pos == golden["N"]`
/// - if golden has a `"roots"` key: the full 2N coordinate list matches exactly.
///
/// Types H3, H4, I5 exercise `GoldenInt` arithmetic and have no `"roots"` key
/// (golden stores integer-only roots).
#[test]
fn root_systems() {
    for name in BASICS_NAMES {
        let g = common::golden(name);
        let components = common::components_of(&g);

        assert_eq!(
            components.len(),
            1,
            "{name}: expected single component, got {}",
            components.len()
        );
        let (series, rank) = components[0];
        let cartan = cartan_mat(series, rank)
            .unwrap_or_else(|e| panic!("{name}: cartan_mat({series:?}, {rank}) failed: {e}"));
        let rs = roots::build(&cartan);

        // --- Check n_pos ---
        let golden_n = g["N"]
            .as_u64()
            .unwrap_or_else(|| panic!("{name}: golden \"N\" is not a u64"))
            as u32;
        assert_eq!(
            rs.n_pos, golden_n,
            "{name}: n_pos mismatch: got {}, expected {}",
            rs.n_pos, golden_n
        );

        // --- Check roots coordinate list (Int types only) ---
        if let Some(golden_roots_val) = g.get("roots") {
            let golden_roots: Vec<Vec<i64>> = serde_json::from_value(golden_roots_val.clone())
                .unwrap_or_else(|e| panic!("{name}: failed to parse golden roots: {e}"));
            let roots_int = rs
                .roots_int
                .as_ref()
                .unwrap_or_else(|| panic!("{name}: expected roots_int but got None"));
            assert_eq!(
                roots_int.len(),
                golden_roots.len(),
                "{name}: roots length mismatch: got {}, expected {}",
                roots_int.len(),
                golden_roots.len()
            );
            for (i, (got, expected)) in roots_int.iter().zip(golden_roots.iter()).enumerate() {
                assert_eq!(
                    got, expected,
                    "{name}: roots[{i}] mismatch: got {got:?}, expected {expected:?}"
                );
            }
        }
    }
}

/// Verify that permgens satisfy the Coxeter relations for every type in BASICS_NAMES.
///
/// For each entry we check:
/// 1. Every permgen is an involution: `perm[perm[i]] == i` for all `i < 2N`.
/// 2. For every pair `s < t`, the product permutation `p = permgens[s] ∘ permgens[t]`
///    (i.e. `p[i] = permgens[t][permgens[s][i]]`) has multiplicative order exactly
///    `coxmat[s][t]` (capped at 100 to detect divergence).
#[test]
fn permgen_coxeter_relations() {
    for name in BASICS_NAMES {
        let g = common::golden(name);
        let components = common::components_of(&g);

        assert_eq!(
            components.len(),
            1,
            "{name}: expected single component, got {}",
            components.len()
        );
        let (series, rank) = components[0];
        let cartan = cartan_mat(series, rank)
            .unwrap_or_else(|e| panic!("{name}: cartan_mat({series:?}, {rank}) failed: {e}"));
        let coxmat = coxeter_mat_from_cartan(&cartan);
        let rs = roots::build(&cartan);
        let n2 = rs.permgens[0].0.len(); // 2N

        // 1. Involution check
        for s in 0..rank {
            let perm = &rs.permgens[s].0;
            for i in 0..n2 {
                let j = perm[i] as usize;
                assert_eq!(
                    perm[j] as usize, i,
                    "{name}: permgens[{s}] is not an involution at root index {i}"
                );
            }
        }

        // 2. Coxeter-relation order check
        for s in 0..rank {
            for t in (s + 1)..rank {
                let m = coxmat[s][t] as usize;
                // Compose: p[i] = permgens[t][permgens[s][i]]
                let ps = &rs.permgens[s].0;
                let pt = &rs.permgens[t].0;
                // Start with identity, apply (ps then pt) repeatedly
                let mut current: Vec<usize> = (0..n2).collect();
                let mut order = 0usize;
                loop {
                    // Apply one step of the product: i -> pt[ps[i]]
                    let next: Vec<usize> = current
                        .iter()
                        .map(|&i| pt[ps[i] as usize] as usize)
                        .collect();
                    order += 1;
                    if next.iter().enumerate().all(|(i, &v)| v == i) {
                        // Reached identity
                        break;
                    }
                    current = next;
                    assert!(
                        order < 100,
                        "{name}: product permgens[{s}]∘permgens[{t}] did not return to \
                         identity within 100 steps (coxmat[{s}][{t}]={m})"
                    );
                }
                assert_eq!(
                    order, m,
                    "{name}: product permgens[{s}]∘permgens[{t}] has order {order}, \
                     expected coxmat[{s}][{t}]={m}"
                );
            }
        }
    }
}

/// Placeholder for I7 and I8 — needs CycInt (Task 18).
#[test]
#[ignore = "needs CycInt (Task 18)"]
fn cartan_data_i7_i8() {
    for name in &["basics_I7", "basics_I8"] {
        let g = common::golden(name);
        let components = common::components_of(&g);
        let (series, rank) = components[0];
        let cartan = cartan_mat(series, rank)
            .unwrap_or_else(|e| panic!("{name}: cartan_mat({series:?}, {rank}) failed: {e}"));
        let cox = coxeter_mat_from_cartan(&cartan);
        let golden_cox: Vec<Vec<u32>> = serde_json::from_value(g["coxetermat"].clone())
            .unwrap_or_else(|e| panic!("{name}: failed to parse golden coxetermat: {e}"));
        assert_eq!(cox, golden_cox, "{name}: coxeter matrix mismatch");
        let mut degrees = degrees_of(series, rank)
            .unwrap_or_else(|e| panic!("{name}: degrees_of({series:?}, {rank}) failed: {e}"));
        degrees.sort_unstable();
        let order = order_from_degrees(&degrees);
        let golden_order = g["order"]
            .as_u64()
            .unwrap_or_else(|| panic!("{name}: golden \"order\" is not a u64"))
            as u128;
        assert_eq!(order, golden_order, "{name}: order mismatch");
    }
}
