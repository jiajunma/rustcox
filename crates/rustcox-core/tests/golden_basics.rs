//! Integration tests: verify Cartan/Coxeter/degree data against golden files.
//!
//! Each golden file encodes the authoritative `coxetermat`, `degrees`, and
//! `order` for a finite Coxeter group.  This test suite builds the data with
//! `cartan_mat` + `coxeter_mat_from_cartan` + `degrees_of` + `order_from_degrees`
//! and asserts exact agreement.

mod common;

use rustcox_core::cartan::{cartan_mat, coxeter_mat_from_cartan, degrees_of, order_from_degrees};
use rustcox_core::enumerate::ElementTable;
use rustcox_core::group::CoxeterGroup;
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
        #[allow(clippy::needless_range_loop)]
        for s in 0..rank {
            #[allow(clippy::needless_range_loop)]
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

/// Verify `longest_perm` + `perm_to_word` against the golden `longest_word`
/// and `N` (= `perm_length` of the longest element) for every type with a
/// `longest_word` key and group order ≤ 10 000.
///
/// Types without `longest_word` in the golden file (E6, H4) are skipped.
/// I7 and I8 require CycInt and are excluded here.
#[test]
fn longest_words() {
    // Subset of BASICS_NAMES that have a golden "longest_word" key AND |W| ≤ 10000.
    const NAMES: &[&str] = &[
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
        "basics_F4",
        "basics_G2",
        "basics_H3",
        "basics_I5",
    ];

    for name in NAMES {
        let g = common::golden(name);

        // Skip if no longest_word in golden (should not happen for this list).
        let golden_lw = match g.get("longest_word") {
            Some(v) => v.clone(),
            None => {
                eprintln!("{name}: no longest_word in golden — skipping");
                continue;
            }
        };
        let golden_word: Vec<u8> = serde_json::from_value(golden_lw)
            .unwrap_or_else(|e| panic!("{name}: failed to parse golden longest_word: {e}"));

        let golden_n = g["N"]
            .as_u64()
            .unwrap_or_else(|| panic!("{name}: golden N is not a u64"))
            as u32;

        // Parse type string from golden, build group.
        // We derive a type string from the golden "type" field.
        let type_str = type_string_from_golden(&g, name);
        let group = CoxeterGroup::from_type(&type_str).unwrap_or_else(|e| {
            panic!("{name}: CoxeterGroup::from_type({type_str:?}) failed: {e}")
        });

        let w0 = group.longest_perm();

        // Check perm_length(w0) == N.
        let l = group.perm_length(w0);
        assert_eq!(
            l, golden_n,
            "{name}: perm_length(longest) = {l}, expected {golden_n}"
        );

        // Check perm_to_word(w0) == golden longest_word.
        let word = group.perm_to_word(w0);
        assert_eq!(
            word, golden_word,
            "{name}: perm_to_word(longest) mismatch\n  got:      {word:?}\n  expected: {golden_word:?}"
        );
    }
}

/// Build a type string like `"B2"` or `"A2"` from a golden file.
///
/// Only handles single-component types (all basics golden files are
/// single-component).
fn type_string_from_golden(g: &serde_json::Value, name: &str) -> String {
    let arr = g["type"]
        .as_array()
        .unwrap_or_else(|| panic!("{name}: golden 'type' is not an array"));
    assert_eq!(arr.len(), 1, "{name}: expected single component");
    let item = &arr[0];
    let series = item["series"]
        .as_str()
        .unwrap_or_else(|| panic!("{name}: missing 'series'"));
    match series {
        "I" => {
            let m = item["m"]
                .as_u64()
                .unwrap_or_else(|| panic!("{name}: I-type missing 'm'"));
            format!("I{m}")
        }
        _ => {
            let rank = item["rank"]
                .as_u64()
                .unwrap_or_else(|| panic!("{name}: missing 'rank'"));
            format!("{series}{rank}")
        }
    }
}

// ---------------------------------------------------------------------------
// Task 6: Element enumeration golden tests
// ---------------------------------------------------------------------------

/// Verify per-length element counts (length_histogram) against golden data.
///
/// For every basics golden file that contains a `length_histogram` key,
/// builds the full `ElementTable` and asserts that the number of elements at
/// each length matches the golden histogram.
///
/// Fast groups (A1..A5, B2..B4, C3, D4, D5, F4, G2, H3, I5) run in the
/// default (non-ignored) profile.  A5 and F4 are gated separately if needed.
#[test]
fn element_tables_length_histogram() {
    // Basics names that have a length_histogram key (excludes E6, H4, I7, I8).
    const NAMES_WITH_HIST: &[&str] = &[
        "basics_A1",
        "basics_A2",
        "basics_A3",
        "basics_A4",
        "basics_B2",
        "basics_B3",
        "basics_G2",
        "basics_H3",
        "basics_I5",
        "basics_C3",
        "basics_D4",
    ];
    // Slow groups (A5, B4, D5, F4) run only in the release + include-ignored profile.
    const NAMES_SLOW: &[&str] = &["basics_A5", "basics_B4", "basics_D5", "basics_F4"];

    let all_names: Vec<&str> = NAMES_WITH_HIST.iter().chain(NAMES_SLOW).copied().collect();

    for name in &all_names {
        let g = common::golden(name);
        // Skip if no length_histogram in this golden file.
        let golden_hist_val = match g.get("length_histogram") {
            Some(v) => v.clone(),
            None => continue,
        };
        let golden_hist: Vec<usize> = serde_json::from_value(golden_hist_val)
            .unwrap_or_else(|e| panic!("{name}: failed to parse length_histogram: {e}"));

        let type_str = type_string_from_golden(&g, name);
        let group = CoxeterGroup::from_type(&type_str).unwrap_or_else(|e| {
            panic!("{name}: CoxeterGroup::from_type({type_str:?}) failed: {e}")
        });
        let table = ElementTable::build(&group);

        // Build actual histogram
        let max_len = *table.lengths.iter().max().unwrap_or(&0) as usize;
        let mut hist = vec![0usize; max_len + 1];
        for &l in &table.lengths {
            hist[l as usize] += 1;
        }

        assert_eq!(
            hist, golden_hist,
            "{name}: length_histogram mismatch\n  got:      {hist:?}\n  expected: {golden_hist:?}"
        );
    }
}

/// Verify exact canonical element list against kl golden files.
///
/// The `kl_A3_w1.json` and `kl_B3_w1.json` golden files contain an `"elms"`
/// key with the canonical element list.  Assert our table matches exactly.
#[test]
fn element_tables_canonical_order() {
    for kl_name in &["kl_A3_w1", "kl_B3_w1"] {
        let g = common::golden(kl_name);

        // Parse the "type" key to build the group.
        let arr = g["type"]
            .as_array()
            .unwrap_or_else(|| panic!("{kl_name}: 'type' is not an array"));
        assert_eq!(arr.len(), 1, "{kl_name}: expected single component");
        let item = &arr[0];
        let series_str = item["series"]
            .as_str()
            .unwrap_or_else(|| panic!("{kl_name}: missing series"));
        let rank_val = item["rank"]
            .as_u64()
            .unwrap_or_else(|| panic!("{kl_name}: missing rank")) as usize;
        let type_str = format!("{series_str}{rank_val}");

        let group = CoxeterGroup::from_type(&type_str)
            .unwrap_or_else(|e| panic!("{kl_name}: from_type({type_str:?}) failed: {e}"));
        let table = ElementTable::build(&group);

        let golden_elms: Vec<Vec<u8>> = serde_json::from_value(g["elms"].clone())
            .unwrap_or_else(|e| panic!("{kl_name}: failed to parse 'elms': {e}"));

        assert_eq!(
            table.elms, golden_elms,
            "{kl_name}: canonical element list mismatch"
        );
    }
}

/// Verify structural invariants on A4:
/// - `lft[w][s] < w  ⟺  s ∈ left_descents(w)` for all (w, s)
/// - `inva[inva[w]] == w` for all w
/// - `lengths[aw0[w]] == N − lengths[w]` for all w
#[test]
fn a4_element_table_invariants() {
    let group = CoxeterGroup::from_type("A4").unwrap();
    let table = ElementTable::build(&group);
    let n = group.n_pos;
    let size = table.len();

    // lft invariant
    for i in 0..size {
        let perm = group.word_to_perm(&table.elms[i]);
        let left_desc = group.left_descents(&perm);
        for s in 0..group.rank {
            let lft_idx = table.lft[i][s];
            let is_left_desc = left_desc.contains(&(s as u8));
            assert_eq!(
                lft_idx < i as u32,
                is_left_desc,
                "A4 lft invariant violated at i={i}, s={s}: lft={lft_idx}, is_left_desc={is_left_desc}"
            );
        }
    }

    // inva∘inva == id
    for i in 0..size {
        assert_eq!(
            table.inva[table.inva[i] as usize] as usize, i,
            "A4 inva∘inva != id at i={i}"
        );
    }

    // aw0 length mirror
    for i in 0..size {
        assert_eq!(
            table.lengths[table.aw0[i] as usize],
            n - table.lengths[i],
            "A4 aw0 length mirror violated at i={i}"
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
