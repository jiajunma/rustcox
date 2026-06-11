//! Experimental test of the **combinatorial invariance conjecture**
//! (Lusztig/Dyer) on small finite Coxeter groups, plus a storage-compression
//! analysis.
//!
//! The conjecture: the Kazhdan–Lusztig polynomial `P_{y,w}` depends only on the
//! isomorphism type of the Bruhat interval `[y, w]`.  The *poset* form (covers /
//! Hasse diagram) is the standard one; a *graph* form uses the full Bruhat
//! graph.  This driver tests both.
//!
//! For a group named on the command line (`bruhat_invariance B3`):
//! 1. Build the full KL table.
//! 2. For every comparable pair `y <_B w` (y ≠ w), extract the interval
//!    `I = {z : y ≤ z ≤ w}` and build the poset-of-covers and the Bruhat graph
//!    with **relative-length** vertex levels (so the test is length-shift
//!    invariant).
//! 3. Classify intervals into isomorphism classes (cheap key + exact test).
//! 4. For each class, collect the set of distinct `P_{y,w}`.  A class with more
//!    than one distinct polynomial is a CONJECTURE VIOLATION.
//! 5. Print a per-group summary, any violations, and a storage-cost table.
//!
//! Conventions: Bruhat-graph edges use the LEFT convention `z2 = t · z1`
//! (`z2 · z1⁻¹` a reflection).  Scope is limited to small groups
//! (A2..B4, H3); never larger.
//!
//! Usage: `cargo run --release --example bruhat_invariance -- B3`

use std::collections::HashMap;

use rustcox_core::{
    element::{ElmIdx, Perm},
    group::CoxeterGroup,
    interval::{
        build_graph, extract_interval, graph_key, is_isomorphic, reflections, GraphKey, GraphKind,
        IntervalGraph,
    },
    kl::{klpolynomials, table::KlTable, KlOpts},
    laurent::Laurent,
};

/// Groups this experiment is permitted to run on.  HARD SCOPE LIMIT.
const ALLOWED: &[&str] = &["A2", "A3", "B2", "B3", "A4", "D4", "H3", "B4"];

fn main() {
    let typ = std::env::args()
        .nth(1)
        .unwrap_or_else(|| panic!("usage: bruhat_invariance <{}>", ALLOWED.join("|")));
    if !ALLOWED.contains(&typ.as_str()) {
        panic!(
            "group {typ:?} is out of scope; allowed: {}",
            ALLOWED.join(", ")
        );
    }

    let report = run_experiment(&typ);
    print!("{}", report.render());
}

// ---------------------------------------------------------------------------
// Per-pair record
// ---------------------------------------------------------------------------

/// One comparable pair `y <_B w` and its interval data.
struct PairRecord {
    y: ElmIdx,
    w: ElmIdx,
    pol: Laurent,
    cover: IntervalGraph,
    bruhat: IntervalGraph,
    cover_key: GraphKey,
    bruhat_key: GraphKey,
}

// ---------------------------------------------------------------------------
// Isomorphism-class bucketing
// ---------------------------------------------------------------------------

/// For one graph flavour: assign each pair to an isomorphism class.
///
/// Two-tier: first bucket by the cheap [`GraphKey`]; within each bucket run the
/// exact level-respecting digraph isomorphism test against existing class
/// representatives.  Returns `class_of[i]` = class index for pair `i`, plus the
/// number of classes.
fn classify(
    records: &[PairRecord],
    key_of: impl Fn(&PairRecord) -> &GraphKey,
    graph_of: impl Fn(&PairRecord) -> &IntervalGraph,
) -> (Vec<usize>, usize) {
    // Bucket pair-indices by cheap key.
    let mut buckets: HashMap<&GraphKey, Vec<usize>> = HashMap::new();
    for (i, rec) in records.iter().enumerate() {
        buckets.entry(key_of(rec)).or_default().push(i);
    }

    let mut class_of = vec![usize::MAX; records.len()];
    let mut next_class = 0usize;

    for (_key, idxs) in buckets {
        // Within a bucket, representatives of distinct classes.
        let mut reps: Vec<usize> = Vec::new(); // pair-index reps
        for &i in &idxs {
            let gi = graph_of(&records[i]);
            let mut found = None;
            for &r in &reps {
                if is_isomorphic(gi, graph_of(&records[r])) {
                    found = Some(class_of[r]);
                    break;
                }
            }
            match found {
                Some(c) => class_of[i] = c,
                None => {
                    class_of[i] = next_class;
                    next_class += 1;
                    reps.push(i);
                }
            }
        }
    }

    (class_of, next_class)
}

/// For a class assignment, collect the set of distinct polynomials per class
/// and return the classes that contain more than one distinct polynomial
/// (conjecture violations) along with one example pair per distinct polynomial.
fn violations(
    records: &[PairRecord],
    class_of: &[usize],
    n_classes: usize,
) -> Vec<(usize, Vec<(usize, Laurent)>)> {
    // class -> map from pol -> example pair-index
    let mut per_class: Vec<HashMap<Laurent, usize>> = vec![HashMap::new(); n_classes];
    for (i, rec) in records.iter().enumerate() {
        per_class[class_of[i]].entry(rec.pol.clone()).or_insert(i);
    }
    let mut out = Vec::new();
    for (c, m) in per_class.into_iter().enumerate() {
        if m.len() > 1 {
            let mut examples: Vec<(usize, Laurent)> = m.into_iter().map(|(p, i)| (i, p)).collect();
            examples.sort_by_key(|(i, _)| *i);
            out.push((c, examples));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

struct Report {
    typ: String,
    order: u128,
    n_pos: u32,
    n_refl: usize,
    n_pairs: usize,
    n_distinct_pols: usize,
    n_poset_classes: usize,
    n_graph_classes: usize,
    poset_violations: Vec<ViolationDetail>,
    graph_violations: Vec<ViolationDetail>,
    /// (relative-gap≤2 pairs checked, all were P=1)
    gap_le2_checked: usize,
}

struct ViolationDetail {
    class: usize,
    interval_size: usize,
    /// (word_y, word_w, P) for each distinct polynomial in the class.
    pairs: Vec<(String, String, Laurent)>,
}

impl Report {
    fn render(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("## {} (order {})\n\n", self.typ, self.order));
        s.push_str(&format!(
            "- positive roots / reflections: {} / {}\n",
            self.n_pos, self.n_refl
        ));
        s.push_str(&format!("- comparable pairs (y < w): {}\n", self.n_pairs));
        s.push_str(&format!(
            "- distinct KL polynomials: {}\n",
            self.n_distinct_pols
        ));
        s.push_str(&format!(
            "- poset (Hasse) iso-classes: {}\n",
            self.n_poset_classes
        ));
        s.push_str(&format!(
            "- Bruhat-graph iso-classes: {}\n",
            self.n_graph_classes
        ));
        s.push_str(&format!(
            "- poset classes with >1 distinct P (VIOLATIONS): {}\n",
            self.poset_violations.len()
        ));
        s.push_str(&format!(
            "- Bruhat-graph classes with >1 distinct P (VIOLATIONS): {}\n",
            self.graph_violations.len()
        ));
        s.push_str(&format!(
            "- cross-check: {} pairs with relative gap ≤ 2 all had P = 1\n",
            self.gap_le2_checked
        ));
        s.push('\n');

        render_violations(&mut s, "Poset (Hasse) violations", &self.poset_violations);
        render_violations(&mut s, "Bruhat-graph violations", &self.graph_violations);

        // Storage analysis.
        s.push_str("### Storage analysis\n\n");
        s.push_str(&self.storage_table());
        s.push('\n');
        s
    }

    fn storage_table(&self) -> String {
        // Scheme A: pair -> pol-index, 4 bytes/pair.
        let bytes_pol_index = self.n_pairs * 4;
        // Pol table itself (count of distinct pols * approx bytes each).
        // We do not serialize pols here; report index cost + a notional table.
        // Scheme B: pair -> class-index (4 B/pair) + class -> pol-index table.
        let bytes_class_index = self.n_pairs * 4 + self.n_poset_classes * 4;
        // Scheme C: nothing per pair; recompute interval on demand → 0 B/pair.
        let bytes_on_demand = 0usize;

        let mut s = String::new();
        s.push_str("| scheme | bytes/pair | total stored | note |\n");
        s.push_str("|--------|-----------|--------------|------|\n");
        s.push_str(&format!(
            "| pair → pol-index | 4 | {} | current scheme |\n",
            bytes_pol_index
        ));
        s.push_str(&format!(
            "| pair → class-index + class→pol | 4 (+4·#classes) | {} | classes ({}) ≥ pols ({}) ⇒ no win |\n",
            bytes_class_index, self.n_poset_classes, self.n_distinct_pols
        ));
        s.push_str(&format!(
            "| nothing per pair, recompute | 0 | {} | trades storage for CPU |\n",
            bytes_on_demand
        ));
        s
    }
}

fn render_violations(s: &mut String, title: &str, vs: &[ViolationDetail]) {
    if vs.is_empty() {
        s.push_str(&format!("_{title}: none._\n\n"));
        return;
    }
    s.push_str(&format!("### {title}\n\n"));
    for v in vs {
        s.push_str(&format!(
            "- class {} (interval size {}):\n",
            v.class, v.interval_size
        ));
        for (wy, ww, p) in &v.pairs {
            s.push_str(&format!("    - y={wy} w={ww} : P = {}\n", fmt_pol(p)));
        }
    }
    s.push('\n');
}

/// Render a Laurent polynomial in `v` compactly.
fn fmt_pol(p: &Laurent) -> String {
    if p.is_zero() {
        return "0".to_string();
    }
    let val = p.val();
    let mut terms = Vec::new();
    // Iterate exponents from val upward; use coeff().
    let deg = p.degree().unwrap_or(val);
    for e in val..=deg {
        let c = p.coeff(e);
        if c == 0 {
            continue;
        }
        let term = match e {
            0 => format!("{c}"),
            1 => {
                if c == 1 {
                    "v".to_string()
                } else {
                    format!("{c}*v")
                }
            }
            _ => {
                if c == 1 {
                    format!("v^{e}")
                } else {
                    format!("{c}*v^{e}")
                }
            }
        };
        terms.push(term);
    }
    if terms.is_empty() {
        "0".to_string()
    } else {
        terms.join(" + ")
    }
}

/// Render a canonical word as a generator string like `[0,1,0]`.
fn word_str(w: &[u8]) -> String {
    let inner: Vec<String> = w.iter().map(|g| g.to_string()).collect();
    format!("[{}]", inner.join(","))
}

// ---------------------------------------------------------------------------
// Experiment driver
// ---------------------------------------------------------------------------

fn run_experiment(typ: &str) -> Report {
    let group = CoxeterGroup::from_type(typ).expect("group");
    let opts = KlOpts::equal(group.rank);
    let table = klpolynomials(&group, &opts).expect("kl table");
    let refls = reflections(&group);
    assert_eq!(
        refls.len(),
        group.n_pos as usize,
        "reflection-set size sanity check failed for {typ}"
    );

    let n = table.n() as ElmIdx;

    // Precompute all perms and lengths once.
    let perms: Vec<Perm> = (0..n)
        .map(|i| group.word_to_perm(&table.elms.elms[i as usize]))
        .collect();
    let lengths: Vec<u32> = table.elms.lengths.clone();

    // Build the per-pair records.
    let mut records: Vec<PairRecord> = Vec::new();
    let mut distinct_pols: std::collections::HashSet<Laurent> = std::collections::HashSet::new();
    let mut gap_le2_checked = 0usize;
    let one = Laurent::one();

    for w in 0..n {
        for y in 0..w {
            if !table.bruhat_leq(y, w) {
                continue;
            }
            // Cross-check: relative gap ≤ 2 ⇒ P = 1 (classical).
            let gap = lengths[w as usize] - lengths[y as usize];
            let pol = table.pol(y, w).expect("comparable ⇒ pol present").clone();
            if gap <= 2 {
                assert_eq!(
                    pol, one,
                    "{typ}: P_{{{y},{w}}} must be 1 for length gap {gap}"
                );
                gap_le2_checked += 1;
            }
            distinct_pols.insert(pol.clone());

            let members = extract_interval(&table, y, w);
            let mp: Vec<Perm> = members.iter().map(|&z| perms[z as usize].clone()).collect();
            let ml: Vec<u32> = members.iter().map(|&z| lengths[z as usize]).collect();

            let cover = build_graph(&members, &mp, &ml, GraphKind::Covers, &refls);
            let bruhat = build_graph(&members, &mp, &ml, GraphKind::Bruhat, &refls);
            // Sanity: covers ⊆ bruhat (edge count).
            debug_assert!(cover.edge_count() <= bruhat.edge_count());
            let cover_key = graph_key(&cover);
            let bruhat_key = graph_key(&bruhat);

            records.push(PairRecord {
                y,
                w,
                pol,
                cover,
                bruhat,
                cover_key,
                bruhat_key,
            });
        }
    }

    // Classify (poset / cover) and (Bruhat graph).
    let (poset_class, n_poset_classes) = classify(&records, |r| &r.cover_key, |r| &r.cover);
    let (graph_class, n_graph_classes) = classify(&records, |r| &r.bruhat_key, |r| &r.bruhat);

    // Cross-check (proven case): lower intervals [e, w] of the SAME poset type
    // MUST have equal P.  e is canonical index 0.  Assert here.
    assert_lower_interval_invariance(&records, &poset_class);

    // Cross-check: identical-poset-class pairs that live in different length
    // windows still share a class — guaranteed by relative-level keys; spot
    // check that at least the construction used relative levels (level[0]==0).
    for r in &records {
        debug_assert_eq!(r.cover.level[0], 0, "vertex 0 must be relative level 0");
    }

    let poset_violations =
        build_violation_details(&group, &table, &records, &poset_class, n_poset_classes);
    let graph_violations =
        build_violation_details(&group, &table, &records, &graph_class, n_graph_classes);

    Report {
        typ: typ.to_string(),
        order: group.order,
        n_pos: group.n_pos,
        n_refl: refls.len(),
        n_pairs: records.len(),
        n_distinct_pols: distinct_pols.len(),
        n_poset_classes,
        n_graph_classes,
        poset_violations,
        graph_violations,
        gap_le2_checked,
    }
}

/// Proven cross-check: for lower intervals `[e, w]` (y == index 0), any two that
/// are poset-isomorphic must carry equal `P_{e,w}`.  This is a theorem; a
/// failure means the canonicalization (not mathematics) is wrong.
fn assert_lower_interval_invariance(records: &[PairRecord], poset_class: &[usize]) {
    // class -> (pol, example pair index) for lower intervals only.
    let mut seen: HashMap<usize, (Laurent, usize)> = HashMap::new();
    for (i, r) in records.iter().enumerate() {
        if r.y != 0 {
            continue; // only lower intervals [e, w]
        }
        let c = poset_class[i];
        match seen.get(&c) {
            None => {
                seen.insert(c, (r.pol.clone(), i));
            }
            Some((p0, j)) => {
                assert_eq!(
                    &r.pol, p0,
                    "PROVEN lower-interval invariance violated: pairs {j} and {i} are \
                     poset-isomorphic lower intervals [e,w] but P differs — \
                     canonicalization bug"
                );
            }
        }
    }
}

fn build_violation_details(
    group: &CoxeterGroup,
    table: &KlTable,
    records: &[PairRecord],
    class_of: &[usize],
    n_classes: usize,
) -> Vec<ViolationDetail> {
    let vs = violations(records, class_of, n_classes);
    vs.into_iter()
        .map(|(class, examples)| {
            let interval_size = records[examples[0].0].cover.n;
            let pairs = examples
                .into_iter()
                .map(|(i, p)| {
                    let r = &records[i];
                    let wy = word_str(&table.elms.elms[r.y as usize]);
                    let ww = word_str(&table.elms.elms[r.w as usize]);
                    let _ = group; // group available if richer rendering needed
                    (wy, ww, p)
                })
                .collect();
            ViolationDetail {
                class,
                interval_size,
                pairs,
            }
        })
        .collect()
}
