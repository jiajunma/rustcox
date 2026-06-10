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
        let got = &ours[key];
        let want = &g[key];
        // For array-valued keys compare element-by-element so failures name the row.
        if let (Some(got_rows), Some(want_rows)) = (got.as_array(), want.as_array()) {
            assert_eq!(
                got_rows.len(),
                want_rows.len(),
                "{name}:{key} length mismatch"
            );
            for (i, (g_row, w_row)) in got_rows.iter().zip(want_rows.iter()).enumerate() {
                assert_eq!(g_row, w_row, "{name}:{key}[{i}] mismatch");
            }
        } else {
            assert_eq!(got, want, "{name}:{key} mismatch");
        }
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

// ---------------------------------------------------------------------------
// Unequal-parameter golden suites (Task 10)
// ---------------------------------------------------------------------------

#[test]
fn kl_b2_w2_1() {
    check_kl_golden("kl_B2_w2_1");
}

#[test]
fn kl_b2_w1_2() {
    check_kl_golden("kl_B2_w1_2");
}

#[test]
fn kl_b3_w2_1_1() {
    check_kl_golden("kl_B3_w2_1_1");
}

#[test]
fn kl_g2_w1_3() {
    check_kl_golden("kl_G2_w1_3");
}

#[test]
fn kl_g2_w3_1() {
    check_kl_golden("kl_G2_w3_1");
}

/// Weight-0 generator coverage: B2 with weights [0, 1].  PyCox accepts
/// weight 0 (verified during Task 10); the pol pool may then contain the
/// zero polynomial, and weight-0 generators carry no mu slots.
#[test]
fn kl_b2_w0_1() {
    check_kl_golden("kl_B2_w0_1");
}
