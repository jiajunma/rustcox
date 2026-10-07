//! The unchanged PyCox-backed engine is the oracle for the compact engine.
use rustcox_core::{
    compact::{compute, CompactOpts, IntervalLimits},
    group::CoxeterGroup,
    kl::{klpolynomials_seq, KlOpts},
};

fn compare(spec: &str, intervals: bool) {
    let g = CoxeterGroup::from_type(spec).unwrap();
    let reference = klpolynomials_seq(&g, &KlOpts::equal(g.rank)).unwrap();
    let opts = CompactOpts {
        intervals: intervals.then(IntervalLimits::default),
        ..CompactOpts::default()
    };
    let compact = compute(&g, &opts).unwrap();
    assert_eq!(compact.elms, reference.elms, "element order: {spec}");
    assert_eq!(compact.pols, reference.pols, "polynomial pool order: {spec}");
    let n = reference.n() as u32;
    for w in 0..n {
        for y in 0..=w {
            assert_eq!(compact.pol(y, w), reference.pol(y, w), "{spec}: ({y},{w})");
            assert_eq!(compact.leq(y, w), reference.bruhat_leq(y, w));
            for s in 0..g.rank {
                let expected = reference.mu(s, y, w).coeff(0);
                assert_eq!(compact.mu_for(s, y, w), expected, "{spec}: mu {s},{y},{w}");
            }
        }
    }
    let report = compact.storage();
    assert!(report.bruhat_words_bytes > 0);
    assert_eq!(report.mu_flag_bytes, 0);
    if intervals {
        assert!(compact.stats.interval_cache_bytes <= opts.intervals.unwrap().max_bytes);
    }
}

#[test]
fn compact_matches_reference_small() {
    for spec in ["A1", "A2", "A3", "A4", "B2", "B3", "B4", "D4", "H3", "I5", "A2xA1"] {
        compare(spec, false);
    }
}

#[test]
fn interval_cache_matches_reference_small() {
    for spec in ["A3", "A4", "B3", "B4", "D4", "H3"] {
        compare(spec, true);
    }
}

#[test]
fn exhausted_cache_budget_is_a_safe_miss() {
    let g = CoxeterGroup::from_type("B4").unwrap();
    let baseline = compute(&g, &CompactOpts::default()).unwrap();
    let opts = CompactOpts {
        intervals: Some(IntervalLimits {
            max_vertices: 4,
            max_bytes: 0,
            max_search_nodes: 0,
            ..IntervalLimits::default()
        }),
        ..CompactOpts::default()
    };
    let limited = compute(&g, &opts).unwrap();
    assert_eq!(baseline.pols, limited.pols);
    for w in 0..baseline.elms.len() as u32 {
        for y in 0..=w {
            assert_eq!(baseline.pol_id(y, w), limited.pol_id(y, w));
        }
    }
    assert_eq!(limited.stats.interval_cache_bytes, 0);
}

#[test]
#[ignore = "release golden-size check; CI includes ignored tests"]
fn compact_and_interval_cache_match_f4() {
    compare("F4", false);
    compare("F4", true);
}
