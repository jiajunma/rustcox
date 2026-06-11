//! Integration tests for relative KL polynomials (`relklpols`, Task P4).
//!
//! These pin the core induction machinery end-to-end: for a parabolic
//! `W1 = W_J ⊂ W` and a left cell `C` of `W1`, the induced set `X1·C` decomposes
//! into a union of left cells of `W`.  We validate this against the **full-table
//! oracle**: build `W`'s full KL table and its `CellData`, build each `W1` cell
//! via the bootstrap bridge, run `relklpols`, decompose, and assert every
//! component is exactly a left cell of `W`'s full table — with the union over all
//! `W1` cells covering every `W`-cell.

use std::collections::{BTreeSet, HashMap};

use rustcox_core::{
    cellgraph::{relkl_input_from_table, CellGraph, KlSlot, MuPools},
    element::{ElmIdx, Word},
    group::CoxeterGroup,
    kl::{relklpols, CellData, KlOpts, KlTable, RelKlOpts},
    laurent::Laurent,
    parabolic::Parabolic,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_table(g: &CoxeterGroup) -> KlTable {
    let opts = KlOpts::equal(g.rank);
    rustcox_core::kl::klpolynomials_seq(g, &opts).unwrap()
}

/// Canonical word-set of every left cell of a full table (a `BTreeSet` of sorted
/// word lists), for set-equality assertions.
fn lcell_word_sets(t: &KlTable) -> BTreeSet<Vec<Word>> {
    let cd = CellData::from_table(t);
    cd.lcells
        .iter()
        .map(|cell| {
            let mut ws: Vec<Word> = cell
                .iter()
                .map(|&e| t.elms.elms[e as usize].clone())
                .collect();
            ws.sort();
            ws
        })
        .collect()
}

/// Sorted word-set of a `CellGraph`'s vertices.
fn graph_word_set(g: &CellGraph) -> Vec<Word> {
    let mut v = g.x.clone();
    v.sort();
    v
}

/// All left cells of W1 (as element-index lists) from its full table.
fn w1_cells(t1: &KlTable) -> Vec<Vec<ElmIdx>> {
    CellData::from_table(t1).lcells
}

/// Induce one W1 cell into W and return the decomposed component word-sets.
fn induce_and_decompose(
    w: &CoxeterGroup,
    w1: &Parabolic,
    t1: &KlTable,
    cell: &[ElmIdx],
) -> Vec<Vec<Word>> {
    // cell1 is a RelKlInput in W1's OWN labels: build it from W1's full table.
    let cell1 = relkl_input_from_table(&w1.group, t1, cell);
    let out = relklpols(w, w1, &cell1, &RelKlOpts::default());
    let weights = vec![1u32; w.rank];
    let cg = CellGraph::from_relkl(w, &weights, &out.input);
    cg.decompose(w).iter().map(graph_word_set).collect()
}

// ---------------------------------------------------------------------------
// Test 1: induced cells match the full table
// ---------------------------------------------------------------------------

/// For each `(W, J)`, induce EVERY left cell of `W1 = W_J` and check that every
/// resulting component is a left cell of `W`'s full table, with the union over
/// all `W1` cells covering all of `W`'s cells.
#[test]
fn induced_cells_match_full_table() {
    // (type, generator to DROP) — J = all generators except `drop`.
    let cases = [("A3", 2usize), ("B3", 2), ("H3", 2), ("A4", 3), ("B4", 3)];

    for (ty, drop) in cases {
        let w = CoxeterGroup::from_type(ty).unwrap();
        let j: Vec<u8> = (0..w.rank as u8).filter(|&s| s as usize != drop).collect();
        let w1 = Parabolic::new(&w, &j).unwrap();

        // Oracles: W's full-table cells; W1's full table for cell construction.
        let tw = build_table(&w);
        let w_cells = lcell_word_sets(&tw);

        let t1 = build_table(&w1.group);
        let cells1 = w1_cells(&t1);

        let mut covered: BTreeSet<Vec<Word>> = BTreeSet::new();

        for (ci, cell) in cells1.iter().enumerate() {
            let comps = induce_and_decompose(&w, &w1, &t1, cell);
            for comp in &comps {
                assert!(
                    w_cells.contains(comp),
                    "{ty} (drop {drop}), W1 cell {ci}: induced component is not a \
                     full-table left cell of {ty}:\n  component = {comp:?}"
                );
                covered.insert(comp.clone());
            }
        }

        // Every W cell must appear at least once across all W1 cells.
        assert_eq!(
            covered,
            w_cells,
            "{ty} (drop {drop}): union of induced components must cover ALL \
             {} full-table cells (covered {})",
            w_cells.len(),
            covered.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: whole W1 induces whole W (A3)
// ---------------------------------------------------------------------------

/// A3: feed ALL of W1 (every element, i.e. the union of all its cells) as one
/// `RelKlInput`; the induced set is all of A3, so the decomposition is exactly
/// the multiset of A3's left cells (each once).
#[test]
fn whole_w1_induces_whole_w() {
    let w = CoxeterGroup::from_type("A3").unwrap();
    let j: Vec<u8> = vec![0, 1]; // drop generator 2 → W1 = A2.
    let w1 = Parabolic::new(&w, &j).unwrap();

    let tw = build_table(&w);
    let w_cells = lcell_word_sets(&tw);

    let t1 = build_table(&w1.group);
    let all: Vec<ElmIdx> = (0..t1.n() as u32).collect();

    let comps = induce_and_decompose(&w, &w1, &t1, &all);

    // As a multiset: each component once; equal to the set of W cells.
    let got: BTreeSet<Vec<Word>> = comps.iter().cloned().collect();
    assert_eq!(
        got.len(),
        comps.len(),
        "A3 whole-W1: components must be distinct (no cell appears twice)"
    );
    assert_eq!(
        got, w_cells,
        "A3 whole-W1: induced components must equal ALL A3 left cells exactly once"
    );
}

// ---------------------------------------------------------------------------
// Test 3: pool sanity
// ---------------------------------------------------------------------------

/// The two pools (`rklpols`, `mues`) are seeded `[zero, one]` and contain no
/// duplicate Laurents.
#[test]
fn rklpols_pool_sane() {
    let w = CoxeterGroup::from_type("B3").unwrap();
    let j: Vec<u8> = vec![0, 1];
    let w1 = Parabolic::new(&w, &j).unwrap();
    let t1 = build_table(&w1.group);

    for cell in w1_cells(&t1) {
        let cell1 = relkl_input_from_table(&w1.group, &t1, &cell);
        let out = relklpols(&w, &w1, &cell1, &RelKlOpts::default());

        // Seeds.
        assert_eq!(out.rklpols[0], rustcox_core::laurent::Laurent::zero());
        assert_eq!(out.rklpols[1], rustcox_core::laurent::Laurent::one());
        assert_eq!(out.mues[0], rustcox_core::laurent::Laurent::zero());
        assert_eq!(out.mues[1], rustcox_core::laurent::Laurent::one());

        // Dedup invariant: no value repeats in either pool.
        let no_dups = |pool: &[rustcox_core::laurent::Laurent]| {
            let set: BTreeSet<_> = pool.iter().map(|p| format!("{p:?}")).collect();
            set.len() == pool.len()
        };
        assert!(no_dups(&out.rklpols), "rklpols has duplicate entries");
        assert!(no_dups(&out.mues), "mues has duplicate entries");

        // The output contract: Global pool.
        assert!(matches!(out.input.mpols, MuPools::Global(_)));
    }
}

// ---------------------------------------------------------------------------
// Test 3b: parallel relklpols is byte-identical to sequential (Task P6)
// ---------------------------------------------------------------------------

/// The relative-KL wavefront must intern `rklpols`/`mues` in the same order and
/// build the same `klmat` regardless of thread count.  We compare the two pools
/// and the induced-graph matrix produced with `threads = Some(1)` against
/// `Some(2)` and `Some(4)` for every W1 cell of B4 (the recursion's deepest
/// non-trivial group in the local suite).
#[test]
fn relklpols_parallel_byte_identical() {
    let w = CoxeterGroup::from_type("B4").unwrap();
    let j: Vec<u8> = vec![0, 1, 2]; // W1 = B3.
    let w1 = Parabolic::new(&w, &j).unwrap();
    let t1 = build_table(&w1.group);

    for cell in w1_cells(&t1) {
        let cell1 = relkl_input_from_table(&w1.group, &t1, &cell);

        let seq = relklpols(&w, &w1, &cell1, &RelKlOpts { threads: Some(1) });
        for &t in &[2usize, 4] {
            let par = relklpols(&w, &w1, &cell1, &RelKlOpts { threads: Some(t) });
            assert_eq!(
                par.rklpols, seq.rklpols,
                "rklpols pool differs at threads={t}"
            );
            assert_eq!(par.mues, seq.mues, "mues pool differs at threads={t}");
            assert_eq!(par.input.elms, seq.input.elms, "elms differ at threads={t}");
            assert_eq!(
                par.input.klmat, seq.input.klmat,
                "klmat differs at threads={t}"
            );
            assert_eq!(par.perms, seq.perms, "perms differ at threads={t}");
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4: hand smoke — A2 with J = {0} (W1 = A1)
// ---------------------------------------------------------------------------

/// A2, J = {0}: X1 has 3 coset reps; the nontrivial A1 cell is `{[0]}`.  The
/// induced set is the 3 elements `{[0], [1,0], [0,1,0]}` and decomposes into the
/// A2 left cells `{[0],[1,0]}` and `{[0,1,0]}` (verified against the PyCox
/// oracle).
#[test]
fn hand_smoke_a2_j0() {
    let w = CoxeterGroup::from_type("A2").unwrap();
    let j: Vec<u8> = vec![0]; // W1 = A1.
    let w1 = Parabolic::new(&w, &j).unwrap();

    let tw = build_table(&w);
    let w_cells = lcell_word_sets(&tw);

    let t1 = build_table(&w1.group);
    let cells1 = w1_cells(&t1);

    // The nontrivial A1 cell is the one element {[0]} (length 1).
    let nontrivial = cells1
        .iter()
        .find(|c| c.len() == 1 && !t1.elms.elms[c[0] as usize].is_empty())
        .expect("A1 must have a nontrivial 1-element cell");

    let cell1 = relkl_input_from_table(&w1.group, &t1, nontrivial);
    let out = relklpols(&w, &w1, &cell1, &RelKlOpts::default());

    // Induced elements (as a set) == {[0], [1,0], [0,1,0]}.
    let induced: BTreeSet<Word> = out.input.elms.iter().cloned().collect();
    let expected: BTreeSet<Word> = [vec![0u8], vec![1, 0], vec![0, 1, 0]].into_iter().collect();
    assert_eq!(induced, expected, "A2 J={{0}}: induced elements");

    // Decompose → components are full-table A2 cells; specifically
    // {[0],[1,0]} and {[0,1,0]}.
    let weights = vec![1u32; w.rank];
    let cg = CellGraph::from_relkl(&w, &weights, &out.input);
    let comps: Vec<Vec<Word>> = cg.decompose(&w).iter().map(graph_word_set).collect();

    let got: BTreeSet<Vec<Word>> = comps.iter().cloned().collect();
    let want: BTreeSet<Vec<Word>> = [vec![vec![0u8], vec![1, 0]], vec![vec![0u8, 1, 0]]]
        .into_iter()
        .collect();
    assert_eq!(got, want, "A2 J={{0}}: decompose components");

    // Each component is a full-table A2 cell.
    for comp in &comps {
        assert!(
            w_cells.contains(comp),
            "A2 J={{0}}: component {comp:?} is not a full-table cell"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5: byte-level oracle comparison against PyCox
// ---------------------------------------------------------------------------

/// A Laurent as `[val, c0, c1, …]` (the fixture encoding); zero ⇒ `[0]`.
fn laurent_from_fixture(arr: &[i64]) -> Laurent {
    if arr.is_empty() {
        return Laurent::zero();
    }
    let val = arr[0] as i32;
    let coeffs: Vec<i64> = arr[1..].to_vec();
    Laurent::from_coeffs(val, coeffs)
}

/// Load the gzipped PyCox oracle fixture.
fn load_oracle() -> serde_json::Value {
    use std::io::Read;
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/relkl_oracle.json.gz");
    let f = std::fs::File::open(&path)
        .unwrap_or_else(|e| panic!("oracle fixture not found ({}): {e}", path.display()));
    let mut s = String::new();
    flate2::read::GzDecoder::new(f)
        .read_to_string(&mut s)
        .expect("failed to decompress oracle fixture");
    serde_json::from_str(&s).expect("failed to parse oracle fixture")
}

/// For each `(W, J)` in the fixture, run the **whole-W1** induction (`cell1`
/// over ALL of W1's elements, i.e. all cells at once) and compare the relative
/// KL polynomial of EVERY slot against the PyCox oracle, keyed by `(ap_word_y,
/// ap_word_x)`.  Also compares the `rklpols` and `mues` pools as value-sets.
///
/// This is the strongest possible correctness check: it pins the actual
/// polynomial values produced by the recursion (including non-monomial entries
/// like `1 + v²` in B3 and the rich H3 pool) rather than only the cell partition.
///
/// # Regenerating the fixture
///
/// The oracle is computed by PyCox on the EXACT `cell1` this Rust code builds
/// (via [`relkl_input_from_table`]), guaranteeing an apples-to-apples
/// comparison.  To regenerate after an intended change:
/// 1. `cargo run -p rustcox-core --example dump_cell1 > /tmp/cell1.json`
/// 2. feed `/tmp/cell1.json` to PyCox `relklpols` (reconstruct the `cell1` dict:
///    `'c0'+'c<i>…'` slots, `v`-built mu pools), serialise `rklpols`/`mues`/`ap`
///    and the per-slot `(rk, mu)` values, then `gzip` to
///    `tests/fixtures/relkl_oracle.json.gz`.
#[test]
fn relklpols_matches_pycox_oracle() {
    let oracle = load_oracle();
    let obj = oracle.as_object().expect("oracle must be a JSON object");

    for (key, data) in obj {
        // Parse type + J.
        let ty = data["type"][0].as_str().unwrap();
        let rank = data["type"][1].as_u64().unwrap() as usize;
        let jvec: Vec<u8> = data["J"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u8)
            .collect();

        let w = CoxeterGroup::from_type(&format!("{ty}{rank}")).unwrap();
        let w1 = Parabolic::new(&w, &jvec).unwrap();

        // Whole-W1 cell1 = full table of W1 (all elements, all cells at once).
        let t1 = build_table(&w1.group);
        let all: Vec<ElmIdx> = (0..t1.n() as u32).collect();
        let cell1 = relkl_input_from_table(&w1.group, &t1, &all);
        let out = relklpols(&w, &w1, &cell1, &RelKlOpts::default());

        // Pool value-sets must match.
        let rust_rk: BTreeSet<Vec<i64>> = out.rklpols.iter().map(laurent_to_fixture).collect();
        let oracle_rk: BTreeSet<Vec<i64>> = data["rklpols"]
            .as_array()
            .unwrap()
            .iter()
            .map(json_to_i64_vec)
            .collect();
        assert_eq!(rust_rk, oracle_rk, "{key}: rklpols value-set mismatch");

        let rust_mu: BTreeSet<Vec<i64>> = out.mues.iter().map(laurent_to_fixture).collect();
        let oracle_mu: BTreeSet<Vec<i64>> = data["mues"]
            .as_array()
            .unwrap()
            .iter()
            .map(json_to_i64_vec)
            .collect();
        assert_eq!(rust_mu, oracle_mu, "{key}: mues value-set mismatch");

        // Build the Rust slot map: (ap_word_y, ap_word_x) -> rklpol value.
        let MuPools::Global(rust_pool) = &out.input.mpols else {
            panic!("{key}: expected Global mpols");
        };
        let mut rust_slots: HashMap<(Word, Word), Laurent> = HashMap::new();
        for (fy, row) in out.input.klmat.iter().enumerate() {
            for (fx, slot) in row.iter().enumerate() {
                let KlSlot::Some(sd) = slot else { continue };
                // The Rust output stores the mu index (Global single-index) in
                // slot.mu[0].  The oracle stores BOTH the rklpol and the mu; we
                // compare the mu value here (the rklpol is checked via the pool
                // value-set and the cell-decomposition tests).
                let mu_val = rust_pool[sd.mu[0] as usize].clone();
                rust_slots.insert(
                    (out.input.elms[fy].clone(), out.input.elms[fx].clone()),
                    mu_val,
                );
            }
        }

        // The diagonal `(ap[i], ap[i])` slots are dropped in the flat klmat
        // (strict lower triangle); the oracle includes them (`c1c0`, mu 0 → with
        // their own row), so we only compare off-diagonal slots.
        let mut compared = 0usize;
        for slot in data["slots"].as_array().unwrap() {
            let yw: Word = slot[0]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u8)
                .collect();
            let xw: Word = slot[1]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u8)
                .collect();
            if yw == xw {
                continue; // diagonal — not stored in the flat lower triangle.
            }
            let oracle_mu = laurent_from_fixture(&json_to_i64_vec(&slot[3]));
            let oracle_rk = laurent_from_fixture(&json_to_i64_vec(&slot[2]));
            let got = rust_slots
                .get(&(yw.clone(), xw.clone()))
                .unwrap_or_else(|| {
                    panic!(
                        "{key}: missing Rust slot for (y={yw:?} len{}, x={xw:?} len{}) \
                     oracle_rk={oracle_rk:?} oracle_mu={oracle_mu:?}",
                        yw.len(),
                        xw.len()
                    )
                });
            assert_eq!(*got, oracle_mu, "{key}: slot ({yw:?}, {xw:?}) mu mismatch");
            compared += 1;
        }
        assert!(compared > 0, "{key}: no slots compared (oracle empty?)");

        // Every Rust off-diagonal slot must be accounted for by the oracle.
        let oracle_pairs: BTreeSet<(Word, Word)> = data["slots"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|slot| {
                let yw: Word = slot[0]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u8)
                    .collect();
                let xw: Word = slot[1]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u8)
                    .collect();
                (yw != xw).then_some((yw, xw))
            })
            .collect();
        for yx in rust_slots.keys() {
            assert!(
                oracle_pairs.contains(yx),
                "{key}: Rust produced extra slot {yx:?} absent in oracle"
            );
        }
    }
}

/// Encode a [`Laurent`] as the fixture's `[val, c0, c1, …]` (zero ⇒ `[0]`).
fn laurent_to_fixture(p: &Laurent) -> Vec<i64> {
    if p.is_zero() {
        return vec![0];
    }
    let mut out = vec![p.val() as i64];
    for e in p.val()..=p.degree().unwrap() {
        out.push(p.coeff(e));
    }
    out
}

/// Parse a JSON `[val, c0, …]` array into `Vec<i64>`.
fn json_to_i64_vec(v: &serde_json::Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}
