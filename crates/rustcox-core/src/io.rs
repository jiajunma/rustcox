//! Canonical JSON export of the KL table (golden format).
//!
//! This is the **Task 9 stub**: it emits only the `elms`, `pols`, `klmat`,
//! and `mumat` keys of the golden document — enough to verify the KL
//! polynomial computation against `golden/kl_*_w1.json`.  Task 14 replaces
//! it with the full `to_canonical_json(&KlTable, &CellData)` document
//! (adding `schema`, `kind`, `type`, `weights`, cells, arrows, …).
//!
//! ## Canonicalisation (must match `pycox-ref/gen_golden.py`)
//!
//! - A `Laurent` serialises as `{"v": val, "c": [coeffs]}`; zero is
//!   `{"v": 0, "c": []}`.
//! - The polynomial pool is deduplicated and sorted by `(val, coeffs)` with
//!   `coeffs` compared lexicographically (empty `[]` sorts first).  `klmat`
//!   stores indices into the sorted pool; `-1` marks a non-Bruhat-comparable
//!   pair.
//! - `mumat[w][y][s]` indexes a per-generator mu pool synthesised from the
//!   *present* slots' values (`table.mu(s, y, w)`), each pool seeded with the
//!   zero polynomial at index 0 and sorted by the same `(val, coeffs)` key;
//!   `-1` marks an absent slot.
//!
//! `klmat[w]` has `w + 1` entries (`y = 0..=w`); the diagonal entry `(w, w)`
//! is the index of the constant polynomial `1`.  `mumat[w][y]` is a
//! rank-length array; the diagonal `(w, w)` is all `-1`.

use serde_json::{json, Value};

use crate::{
    kl::cells::CellData,
    kl::table::{KlTable, MuMode, NOT_LEQ, NO_MU},
    laurent::Laurent,
};

// ---------------------------------------------------------------------------
// Pool canonicalisation key
// ---------------------------------------------------------------------------

/// Sort key for a `Laurent`: `(val, coeffs)`.  The zero polynomial uses
/// `val = 0` and an empty coefficient vector, so it sorts before every
/// non-empty polynomial of `val == 0` (lexicographic, `[]` first).
fn pol_key(p: &Laurent) -> (i32, Vec<i64>) {
    if p.is_zero() {
        (0, Vec::new())
    } else {
        (
            p.val(),
            (p.val()..=p.degree().unwrap())
                .map(|e| p.coeff(e))
                .collect(),
        )
    }
}

// ---------------------------------------------------------------------------
// Polynomial pool: dedup + canonical sort + remap
// ---------------------------------------------------------------------------

/// Deduplicate and canonically sort `pols`, returning `(sorted_json, remap)`
/// where `remap[i]` is the canonical index of the polynomial originally at
/// pool index `i`.
fn canonical_pol_pool(pols: &[Laurent]) -> (Vec<Value>, Vec<usize>) {
    // Compute keys once; reuse for both dedup and remap.
    let keys: Vec<(i32, Vec<i64>)> = pols.iter().map(pol_key).collect();

    let mut uniq = keys.clone();
    uniq.sort();
    uniq.dedup();

    let remap: Vec<usize> = keys
        .iter()
        .map(|k| uniq.binary_search(k).expect("key present"))
        .collect();
    let sorted_json: Vec<Value> = uniq.iter().map(|(v, c)| json!({"v": v, "c": c})).collect();
    (sorted_json, remap)
}

// ---------------------------------------------------------------------------
// Public stub exporter
// ---------------------------------------------------------------------------

/// Emit `{"elms", "pols", "klmat", "mumat"}` of the canonical golden document.
///
/// This is the Task 9 stub used by `tests/golden_kl.rs`; it does **not**
/// emit the cell/arrow keys (Task 11) nor the document envelope (Task 14).
pub fn table_json(t: &KlTable) -> Value {
    let n = t.elms.len();
    let rank = t.elms.rank;

    // -- elms: canonical words --
    let elms: Vec<Value> = t
        .elms
        .elms
        .iter()
        .map(|w| Value::from(w.iter().map(|&s| s as u64).collect::<Vec<_>>()))
        .collect();

    // -- pols: dedup + canonical sort + remap --
    let (pols_json, pol_remap) = canonical_pol_pool(&t.pols);

    // -- klmat: per-row pol indices (or -1) --
    let mut klmat: Vec<Value> = Vec::with_capacity(n);
    for w in 0..n {
        let row = &t.rows[w];
        let entries: Vec<i64> = (0..=w)
            .map(|y| {
                if y == w {
                    // diagonal: P_{w,w} = 1 = pols[0]; remap it canonically
                    pol_remap[0] as i64
                } else {
                    let idx = row.pol[y];
                    if idx == NOT_LEQ {
                        -1
                    } else {
                        pol_remap[idx as usize] as i64
                    }
                }
            })
            .collect();
        klmat.push(Value::from(entries));
    }

    // -- mumat: synthesise per-generator mu pools from present slots --
    let mumat = mumat_json(t, n, rank);

    json!({
        "elms": elms,
        "pols": pols_json,
        "klmat": klmat,
        "mumat": mumat,
    })
}

// ---------------------------------------------------------------------------
// Cell data JSON
// ---------------------------------------------------------------------------

/// Emit `{"arrows", "lcells", "duflo", "lorder", "rcells", "tcells"}` of the
/// canonical golden document, matching `gen_golden.py`'s types exactly.
///
/// - `arrows`: sorted list of 2-element integer arrays `[w, y]`.
/// - `lcells` / `rcells` / `tcells`: lists of ascending integer arrays.
/// - `duflo`: list of `[d, a(d), n_d]` integer triples, aligned with `lcells`.
/// - `lorder`: incidence matrix with `0`/`1` integer entries, aligned with
///   `lcells`.
pub fn cells_json(c: &CellData) -> Value {
    let arrows: Vec<Value> = c
        .arrows
        .iter()
        .map(|&(w, y)| Value::from(vec![w as i64, y as i64]))
        .collect();

    let lcells = cells_to_json(&c.lcells);
    let rcells = cells_to_json(&c.rcells);
    let tcells = cells_to_json(&c.tcells);

    let duflo: Vec<Value> = c
        .duflo
        .iter()
        .map(|&(d, a, n)| Value::from(vec![d as i64, a as i64, n]))
        .collect();

    let lorder: Vec<Value> = c
        .lorder
        .iter()
        .map(|row| Value::from(row.iter().map(|&b| b as i64).collect::<Vec<_>>()))
        .collect();

    json!({
        "arrows": arrows,
        "lcells": lcells,
        "duflo": duflo,
        "lorder": lorder,
        "rcells": rcells,
        "tcells": tcells,
    })
}

/// Convert a list of cells (each a slice of element indices) to JSON.
fn cells_to_json(cells: &[Vec<u32>]) -> Vec<Value> {
    cells
        .iter()
        .map(|cell| Value::from(cell.iter().map(|&w| w as i64).collect::<Vec<_>>()))
        .collect()
}

// ---------------------------------------------------------------------------
// mu pool synthesis + mumat remap
// ---------------------------------------------------------------------------

/// Build `mumat`: for each generator collect the present slots' mu *values*,
/// synthesise a canonical pool seeded with zero, and remap each present slot
/// to its value's pool index (`-1` for absent slots).
///
/// Pool synthesis matches `gen_golden.py`: PyCox seeds `mues[s]` with `0` at
/// index 0, so the synthesised pool is `{zero} ∪ {present values}` sorted by
/// the `(val, coeffs)` key with `[]` first.  In Implicit (equal-parameter)
/// mode a present slot may carry a zero value; it maps to pool index 0.  In
/// Stored (unequal-parameter) mode the present slots and their values come
/// from `rows[w].mu` / `mues[s]` (via [`mu_slot_present`] / [`KlTable::mu`]),
/// so weight-0 generators contribute no slots; the synthesised pool is a
/// canonical re-sort of the values that actually appear and therefore matches
/// PyCox's `mues[s]` after canonicalisation.
fn mumat_json(t: &KlTable, n: usize, rank: usize) -> Vec<Value> {
    // Per-generator sorted unique value keys (zero always included).
    let mut uniq_per_gen: Vec<Vec<(i32, Vec<i64>)>> = vec![vec![pol_key(&Laurent::zero())]; rank];

    // First pass: collect present-slot values.
    for w in 0..n {
        for y in 0..w {
            // Only Bruhat-comparable pairs can carry mu slots.
            if !t.bruhat_leq(y as u32, w as u32) {
                continue;
            }
            for (s, uniq) in uniq_per_gen.iter_mut().enumerate() {
                if mu_slot_present(t, s, y, w) {
                    uniq.push(pol_key(&t.mu(s, y as u32, w as u32)));
                }
            }
        }
    }
    for uniq in &mut uniq_per_gen {
        uniq.sort();
        uniq.dedup();
    }

    // Second pass: emit mumat[w][y] = [pool index per generator, -1 absent].
    let mut mumat: Vec<Value> = Vec::with_capacity(n);
    for w in 0..n {
        let mut wrow: Vec<Value> = Vec::with_capacity(w + 1);
        for y in 0..=w {
            let entry: Vec<i64> = (0..rank)
                .map(|s| {
                    if y == w {
                        return -1; // diagonal: no mu slots
                    }
                    if !t.bruhat_leq(y as u32, w as u32) || !mu_slot_present(t, s, y, w) {
                        -1
                    } else {
                        let key = pol_key(&t.mu(s, y as u32, w as u32));
                        uniq_per_gen[s].binary_search(&key).expect("mu key present") as i64
                    }
                })
                .collect();
            wrow.push(Value::from(entry));
        }
        mumat.push(Value::from(wrow));
    }
    mumat
}

/// Whether the mu slot `(s, y, w)` is *present*.
///
/// - **Implicit (equal-parameter) mode:** the geometric condition
///   `lft(y, s) < y && lft(w, s) > w` (a present slot may still be zero).
/// - **Stored (unequal-parameter) mode:** the stored slot id is not `NO_MU`.
///   Weight-0 generators carry no slot, and PyCox's geometric condition is
///   gated on `poids[s] > 0`, so reading the row directly is authoritative.
fn mu_slot_present(t: &KlTable, s: usize, y: usize, w: usize) -> bool {
    match t.mu_mode {
        MuMode::Implicit => {
            let elms = &t.elms;
            elms.lft(y as u32, s) < y as u32 && elms.lft(w as u32, s) > w as u32
        }
        MuMode::Stored => {
            let rank = t.elms.rank;
            let row = &t.rows[w];
            let mu_vec = row.mu.as_ref().expect("Stored mode: mu vec missing");
            mu_vec[y * rank + s] != NO_MU
        }
    }
}
