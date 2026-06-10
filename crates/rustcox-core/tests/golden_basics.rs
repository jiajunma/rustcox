//! Integration tests: verify Cartan/Coxeter/degree data against golden files.
//!
//! Each golden file encodes the authoritative `coxetermat`, `degrees`, and
//! `order` for a finite Coxeter group.  This test suite builds the data with
//! `cartan_mat` + `coxeter_mat_from_cartan` + `degrees_of` + `order_from_degrees`
//! and asserts exact agreement.

mod common;

use rustcox_core::cartan::{cartan_mat, coxeter_mat_from_cartan, degrees_of, order_from_degrees};

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
        let golden_cox: Vec<Vec<u32>> = serde_json::from_value(g["coxetermat"].clone()).unwrap();
        assert_eq!(cox, golden_cox, "{name}: coxeter matrix mismatch");
        let mut degrees = degrees_of(series, rank).unwrap();
        degrees.sort_unstable();
        let order = order_from_degrees(&degrees);
        let golden_order = g["order"].as_u64().unwrap() as u128;
        assert_eq!(order, golden_order, "{name}: order mismatch");
    }
}
