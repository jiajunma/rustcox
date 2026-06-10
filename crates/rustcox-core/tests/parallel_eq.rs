//! Determinism tests for the parallel KL driver (Task 12).
//!
//! The parallel driver [`klpolynomials`] must produce a `KlTable` that is
//! **byte-identical** (full `PartialEq`, including pool order) to the
//! sequential reference [`klpolynomials_seq`], across thread counts, layer
//! chunkings, and both the equal- and unequal-parameter paths.

mod common;

use rustcox_core::group::CoxeterGroup;
use rustcox_core::kl::{klpolynomials, klpolynomials_seq, KlOpts};

/// Equal parameters: the parallel result must equal the sequential one for a
/// spread of groups and thread counts.
#[test]
fn parallel_equals_sequential_equal_params() {
    for spec in ["B3", "D4", "H3", "A4"] {
        let g = CoxeterGroup::from_type(spec).unwrap();
        let seq = klpolynomials_seq(&g, &KlOpts::equal(g.rank)).unwrap();
        for t in [2, 4, 8] {
            let opts = KlOpts {
                weights: vec![1; g.rank],
                threads: Some(t),
                layer_chunk: None,
            };
            assert_eq!(klpolynomials(&g, &opts).unwrap(), seq, "{spec} t={t}");
        }
    }
}

/// Unequal parameters: the Stored-mode path must also be byte-identical.
#[test]
fn parallel_uneq_equals_sequential() {
    let cases: &[(&str, &[u32])] = &[("B3", &[2, 1, 1]), ("B2", &[0, 1]), ("G2", &[1, 3])];
    for &(spec, weights) in cases {
        let g = CoxeterGroup::from_type(spec).unwrap();
        let seq_opts = KlOpts {
            weights: weights.to_vec(),
            threads: None,
            layer_chunk: None,
        };
        let seq = klpolynomials_seq(&g, &seq_opts).unwrap();
        for t in [2, 4] {
            let opts = KlOpts {
                weights: weights.to_vec(),
                threads: Some(t),
                layer_chunk: None,
            };
            assert_eq!(
                klpolynomials(&g, &opts).unwrap(),
                seq,
                "{spec} weights={weights:?} t={t}"
            );
        }
    }
}

/// Layer chunking must not change the result: chunks follow the unit order, so
/// the intern order is unchanged regardless of chunk size.
#[test]
fn layer_chunking_is_deterministic() {
    let g = CoxeterGroup::from_type("H3").unwrap();
    let seq = klpolynomials_seq(&g, &KlOpts::equal(g.rank)).unwrap();
    for chunk in [Some(7), Some(1)] {
        let opts = KlOpts {
            weights: vec![1; g.rank],
            threads: Some(4),
            layer_chunk: chunk,
        };
        assert_eq!(
            klpolynomials(&g, &opts).unwrap(),
            seq,
            "H3 layer_chunk={chunk:?}"
        );
    }
}

/// `threads = Some(1)` falls back to the sequential driver and is identical.
#[test]
fn threads_one_falls_back() {
    let g = CoxeterGroup::from_type("B2").unwrap();
    let seq = klpolynomials_seq(&g, &KlOpts::equal(g.rank)).unwrap();
    let opts = KlOpts {
        weights: vec![1; g.rank],
        threads: Some(1),
        layer_chunk: None,
    };
    assert_eq!(klpolynomials(&g, &opts).unwrap(), seq, "B2 t=1");
}
