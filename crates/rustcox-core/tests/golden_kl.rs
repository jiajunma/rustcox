//! Golden tests: verify full canonical KL documents against PyCox golden data.
//!
//! Task 14 upgrades this from partial key comparison (Task 9/11) to a
//! full-document comparison: we call [`rustcox_core::io::to_canonical_json`]
//! and compare the entire JSON value against the golden file, key-by-key for
//! useful diagnostics plus a final whole-document assertion.
//!
//! All 18 regular suites × both drivers (seq + par) must pass.
//! Two `#[ignore = "slow"]` tests cover the big gz goldens (A5, F4).

mod common;

use rustcox_core::io;
use rustcox_core::kl::{klpolynomials, klpolynomials_seq, CellData, KlOpts};

/// Build the group, run a KL computation, and compare the full canonical
/// JSON document against the golden file key-by-key (plus whole-doc).
///
/// Runs against **both** drivers (sequential and parallel at `threads = Some(4)`).
fn check_kl_golden(name: &str) {
    let g = common::golden(name);

    let type_val = &g["type"];
    let group = io::group_from_type_json(type_val)
        .unwrap_or_else(|e| panic!("{name}: build group failed: {e:?}"));

    let weights = io::weights_from_json(&g["weights"], group.rank)
        .unwrap_or_else(|e| panic!("{name}: parse weights failed: {e}"));

    // Sequential reference driver.
    let seq_opts = KlOpts {
        weights: weights.clone(),
        threads: None,
        layer_chunk: None,
    };
    let seq_table = klpolynomials_seq(&group, &seq_opts)
        .unwrap_or_else(|e| panic!("{name}: klpolynomials_seq failed: {e:?}"));
    let seq_cells = CellData::from_table(&seq_table);
    let seq_doc = io::to_canonical_json(&seq_table, &seq_cells, &group);
    compare_full_document(name, "seq", &seq_doc, &g);

    // Parallel driver at 4 threads — must match golden byte-for-byte too.
    let par_opts = KlOpts {
        weights,
        threads: Some(4),
        layer_chunk: None,
    };
    let par_table = klpolynomials(&group, &par_opts)
        .unwrap_or_else(|e| panic!("{name}: klpolynomials (t=4) failed: {e:?}"));
    let par_cells = CellData::from_table(&par_table);
    let par_doc = io::to_canonical_json(&par_table, &par_cells, &group);
    compare_full_document(name, "par(t=4)", &par_doc, &g);
}

/// Full-document comparison: assert every key present in the golden matches the
/// computed document, then assert the whole documents are equal for a clean
/// final check.
fn compare_full_document(
    name: &str,
    driver: &str,
    ours: &serde_json::Value,
    golden: &serde_json::Value,
) {
    let golden_obj = golden.as_object().expect("golden is a JSON object");

    // Per-key diagnostics: array keys are compared element-by-element for
    // better failure messages.
    for (key, want) in golden_obj {
        let got = &ours[key];
        if let (Some(got_rows), Some(want_rows)) = (got.as_array(), want.as_array()) {
            assert_eq!(
                got_rows.len(),
                want_rows.len(),
                "{name}[{driver}]:{key} length mismatch"
            );
            for (i, (g_row, w_row)) in got_rows.iter().zip(want_rows.iter()).enumerate() {
                assert_eq!(g_row, w_row, "{name}[{driver}]:{key}[{i}] mismatch");
            }
        } else {
            assert_eq!(got, want, "{name}[{driver}]:{key} mismatch");
        }
    }

    // Whole-document equality catches any extra keys emitted by our code.
    assert_eq!(ours, golden, "{name}[{driver}] whole-document mismatch");
}

// ---------------------------------------------------------------------------
// Regular test suite (18 golden files, both drivers each)
// ---------------------------------------------------------------------------

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

// Cyclotomic dihedral types (Task 18: CycInt). Both drivers must match PyCox
// byte-for-byte, exercising the full CycInt → root-system → KL pipeline.
#[test]
fn kl_i7() {
    check_kl_golden("kl_I7_w1");
}

#[test]
fn kl_i8_w1_2() {
    check_kl_golden("kl_I8_w1_2");
}

// Even-m dihedral I₂(10): exercises the asymmetric cyclotomic Cartan
// [[2,-1],[-2-ir(5),2]] (m/2 = 5) through the full KL pipeline.
#[test]
fn kl_i10() {
    check_kl_golden("kl_I10_w1");
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

/// Weight-0 generator coverage: B2 with weights [0, 1].
#[test]
fn kl_b2_w0_1() {
    check_kl_golden("kl_B2_w0_1");
}

// ---------------------------------------------------------------------------
// Big gz goldens (A5, F4) — ignored in default profile, included in release CI
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow"]
fn kl_a5() {
    check_kl_golden("kl_A5_w1");
}

#[test]
#[ignore = "slow"]
fn kl_f4() {
    check_kl_golden("kl_F4_w1");
}
