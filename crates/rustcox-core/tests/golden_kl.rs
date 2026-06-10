//! Golden tests: verify sequential KL polynomials against PyCox golden data.
//!
//! Task 9 compares only the `elms`, `pols`, `klmat`, and `mumat` keys (the
//! KL-polynomial stub of `io::table_json`).  Task 11 adds the cell/arrow keys;
//! Task 14 compares the full document.

mod common;

use rustcox_core::group::CoxeterGroup;
use rustcox_core::kl::{klpolynomials_seq, KlOpts};

/// Build the group, run the sequential equal-parameter KL computation, and
/// compare the canonical-JSON stub against the golden file (keys `elms`,
/// `pols`, `klmat`, `mumat`).
fn check_kl_golden(name: &str) {
    let g = common::golden(name);
    let components = common::components_of(&g);
    let group = CoxeterGroup::from_components(&components)
        .unwrap_or_else(|e| panic!("{name}: build group failed: {e:?}"));

    let weights: Vec<u32> = g["weights"]
        .as_array()
        .expect("golden \"weights\" is not an array")
        .iter()
        .map(|w| w.as_u64().expect("weight not an integer") as u32)
        .collect();

    let opts = KlOpts {
        weights,
        threads: None,
        layer_chunk: None,
    };
    let table = klpolynomials_seq(&group, &opts)
        .unwrap_or_else(|e| panic!("{name}: klpolynomials_seq failed: {e:?}"));

    let ours = rustcox_core::io::table_json(&table);
    for key in ["elms", "pols", "klmat", "mumat"] {
        assert_eq!(ours[key], g[key], "{name}:{key} mismatch");
    }
}

#[test]
fn kl_a1() {
    check_kl_golden("kl_A1_w1");
}

#[test]
fn kl_a2() {
    check_kl_golden("kl_A2_w1");
}

#[test]
fn kl_a3() {
    check_kl_golden("kl_A3_w1");
}

#[test]
fn kl_a4() {
    check_kl_golden("kl_A4_w1");
}

#[test]
fn kl_b2() {
    check_kl_golden("kl_B2_w1");
}

#[test]
fn kl_b3() {
    check_kl_golden("kl_B3_w1");
}

#[test]
fn kl_b4() {
    check_kl_golden("kl_B4_w1");
}

#[test]
fn kl_c3() {
    check_kl_golden("kl_C3_w1");
}

#[test]
fn kl_d4() {
    check_kl_golden("kl_D4_w1");
}

#[test]
fn kl_g2() {
    check_kl_golden("kl_G2_w1");
}

#[test]
fn kl_h3() {
    check_kl_golden("kl_H3_w1");
}

#[test]
fn kl_i5() {
    check_kl_golden("kl_I5_w1");
}
