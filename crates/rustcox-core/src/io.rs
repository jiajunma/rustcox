//! Canonical JSON export/import of KL and basics data (golden format).
//!
//! ## Canonicalisation (must match `pycox-ref/gen_golden.py`)
//!
//! - A `Laurent` serialises as `{"v": val, "c": [coeffs]}`; zero is
//!   `{"v": 0, "c": []}`.
//! - Polynomial pools ("pols", and "mues" per generator) are deduplicated and
//!   sorted by `(val, coeffs)` with coeffs compared lexicographically; matrices
//!   store indices into the sorted pools; `-1` marks a non-Bruhat-comparable pair.
//! - `mumat[w][y][s]` is the index into `mues[s]` for the mu-value at `(s,y,w)`
//!   if the slot is present, else `-1`; the diagonal is all `-1`.
//! - Cells are sorted ascending; cell lists are sorted lexicographically.
//!   `duflo` and `lorder` are aligned with `lcells`.
//! - The type field serialises each component as `{"series":"B","rank":2}` or
//!   `{"series":"I","rank":2,"m":7}` (for dihedral groups).

use serde_json::{json, Value};
use thiserror::Error;

use crate::{
    cartan::Series,
    enumerate::ElementTable,
    group::{CoxeterGroup, TypeComponent},
    kl::cells::CellData,
    kl::table::{KlTable, MuMode, NOT_LEQ, NO_MU},
    laurent::Laurent,
};

/// Errors that can occur when parsing golden JSON.
#[derive(Debug, Error)]
pub enum IoError {
    #[error("weight at index {index} is not an integer: {value:?}")]
    NonIntegerWeight { index: usize, value: Value },
}

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
// Type field serialisation
// ---------------------------------------------------------------------------

/// Serialise the group's component list into the golden "type" field.
///
/// Each component produces `{"series":"B","rank":2}` for normal types, or
/// `{"series":"I","rank":2,"m":7}` for dihedral I₂(m) types.
fn type_json(components: &[TypeComponent]) -> Value {
    Value::Array(
        components
            .iter()
            .map(|c| {
                let rank = c.indices.len();
                match c.series {
                    Series::I(m) => json!({
                        "series": "I",
                        "rank": rank,
                        "m": m,
                    }),
                    s => {
                        let series_str = format!("{s}");
                        json!({
                            "series": series_str,
                            "rank": rank,
                        })
                    }
                }
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// mu pool synthesis: build per-generator canonical pools
// ---------------------------------------------------------------------------

/// Sorted unique pool keys per generator: `Vec<(val, coeffs)>`, one entry per
/// generator.
type MuPoolKeys = Vec<Vec<(i32, Vec<i64>)>>;

/// Build per-generator mu canonical pools and remaps from the present mu slots.
///
/// Returns `(pools_json, remaps)` where `pools_json[s]` is the canonical sorted
/// JSON pool for generator `s`, and `remaps[s][flat_slot_idx]` is the canonical
/// index in that pool for the value of that slot.
///
/// The pool is seeded with zero at index 0 (PyCox seeding convention), then
/// all present slot values are collected, deduped, and sorted by `(val, coeffs)`.
fn mu_pools(t: &KlTable, n: usize, rank: usize) -> (Vec<Vec<Value>>, MuPoolKeys) {
    // Per-generator sorted unique value keys (zero always included).
    let mut uniq_per_gen: Vec<Vec<(i32, Vec<i64>)>> = vec![vec![pol_key(&Laurent::zero())]; rank];

    // First pass: collect present-slot values.
    for w in 0..n {
        for y in 0..w {
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

    let pools_json: Vec<Vec<Value>> = uniq_per_gen
        .iter()
        .map(|uniq| uniq.iter().map(|(v, c)| json!({"v": v, "c": c})).collect())
        .collect();

    (pools_json, uniq_per_gen)
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
// Full canonical golden document (Task 14)
// ---------------------------------------------------------------------------

/// Complete canonical golden document (schema `rustcox-golden-v1`, kind `"kl"`).
///
/// Emits ALL golden keys:
/// `schema`, `kind`, `type`, `weights`, `rank`, `order`, `N`,
/// `elms`, `pols`, `mues`, `klmat`, `mumat`,
/// `arrows`, `lcells`, `duflo`, `lorder`, `rcells`, `tcells`.
///
/// The `mues` key is the list of per-generator canonical pools (matching
/// PyCox's `canonical_pool(kl['mpols'][s])`).  These pools are the same ones
/// internally used to build `mumat`; here they are also emitted directly.
pub fn to_canonical_json(t: &KlTable, c: &CellData, group: &CoxeterGroup) -> Value {
    let n = t.elms.len();
    let rank = t.elms.rank;

    // -- type --
    let type_val = type_json(&group.components);

    // -- elms: canonical words --
    let elms: Vec<Value> = t
        .elms
        .elms
        .iter()
        .map(|w| Value::from(w.iter().map(|&s| s as u64).collect::<Vec<_>>()))
        .collect();

    // -- pols: dedup + canonical sort + remap --
    let (pols_json, pol_remap) = canonical_pol_pool(&t.pols);

    // -- mues + mumat: compute pools + build mumat using same uniq_per_gen --
    let (mues_json, uniq_per_gen) = mu_pools(t, n, rank);
    let mumat = mumat_json_from_pools(t, n, rank, &uniq_per_gen);

    // -- klmat: per-row pol indices (or -1) --
    let mut klmat: Vec<Value> = Vec::with_capacity(n);
    for w in 0..n {
        let row = &t.rows[w];
        let entries: Vec<i64> = (0..=w)
            .map(|y| {
                if y == w {
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

    // -- cells --
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
        "schema": "rustcox-golden-v1",
        "kind": "kl",
        "type": type_val,
        "weights": t.weights,
        "rank": rank,
        "order": group.order as u64,
        "N": group.n_pos,
        "elms": elms,
        "pols": pols_json,
        "mues": mues_json,
        "klmat": klmat,
        "mumat": mumat,
        "arrows": arrows,
        "lcells": lcells,
        "duflo": duflo,
        "lorder": lorder,
        "rcells": rcells,
        "tcells": tcells,
    })
}

// ---------------------------------------------------------------------------
// Basics document (Task 14)
// ---------------------------------------------------------------------------

/// Emit a complete canonical basics golden document (schema `rustcox-golden-v1`,
/// kind `"basics"`).
///
/// Emits ALL golden keys:
/// `schema`, `kind`, `type`, `rank`, `order`, `N`, `degrees`, `coxetermat`,
/// optionally `roots` (when `group.roots_int` is `Some`),
/// and optionally `length_histogram` + `longest_word` (when `group.order ≤ 10000`).
pub fn basics_json(group: &CoxeterGroup) -> Value {
    let type_val = type_json(&group.components);

    let rank = group.rank;
    let n_pos = group.n_pos;
    let order = group.order;

    // Degrees: sort ascending (gen_golden.py uses sorted(W.degrees)).
    let mut degrees = group.degrees.clone();
    degrees.sort_unstable();

    // Coxeter matrix as nested arrays.
    let coxetermat: Vec<Value> = group
        .coxmat
        .iter()
        .map(|row| Value::from(row.iter().map(|&x| x as i64).collect::<Vec<_>>()))
        .collect();

    let mut doc = json!({
        "schema": "rustcox-golden-v1",
        "kind": "basics",
        "type": type_val,
        "rank": rank,
        "order": order as u64,
        "N": n_pos,
        "degrees": degrees,
        "coxetermat": coxetermat,
    });

    // roots: only when all coordinates are integers (crystallographic types).
    if let Some(roots) = &group.roots_int {
        let roots_val: Vec<Value> = roots.iter().map(|r| Value::from(r.clone())).collect();
        doc["roots"] = Value::Array(roots_val);
    }

    // length_histogram + longest_word: only when |W| ≤ 10000.
    if order <= 10000 {
        let table = ElementTable::build(group);
        let max_len = *table.lengths.iter().max().unwrap_or(&0) as usize;
        let mut hist = vec![0u64; max_len + 1];
        for &l in &table.lengths {
            hist[l as usize] += 1;
        }
        doc["length_histogram"] = Value::Array(hist.iter().map(|&x| json!(x)).collect());

        let w0 = group.longest_perm();
        let word: Vec<u64> = group.perm_to_word(w0).iter().map(|&s| s as u64).collect();
        doc["longest_word"] = json!(word);
    }

    doc
}

// ---------------------------------------------------------------------------
// Cells golden document (Task P6 — klcells parabolic-induction left cells)
// ---------------------------------------------------------------------------

/// Build the canonical `kind: "cells"` golden document for a [`KlCellsResult`].
///
/// Mirrors `gen_golden.py`'s `gen_cells` exactly: keys `schema`, `kind`,
/// `type`, `weights`, `rank`, `order`, `N`, `ncells`, `nstarreps`, `cells`.
/// `cells` is the already-canonicalized partition from [`klcells`] (each word a
/// canonical reduced word, each cell sorted by `(len, lex)`, the cell list
/// sorted lexicographically), so it is emitted verbatim — `verify` can compare
/// the output byte-for-byte against the `cells_*` goldens.
///
/// Equal parameters only: `weights` is all-ones of length `rank`.
///
/// [`klcells`]: fn@crate::kl::klcells
/// [`KlCellsResult`]: crate::kl::KlCellsResult
pub fn cells_json_doc(group: &CoxeterGroup, res: &crate::kl::KlCellsResult) -> Value {
    let type_val = type_json(&group.components);

    // cells: nested arrays of words (each word a list of generator indices).
    let cells: Vec<Value> = res
        .cells
        .iter()
        .map(|cell| {
            Value::Array(
                cell.iter()
                    .map(|w| Value::from(w.iter().map(|&s| s as u64).collect::<Vec<_>>()))
                    .collect(),
            )
        })
        .collect();

    json!({
        "schema": "rustcox-golden-v1",
        "kind": "cells",
        "type": type_val,
        "weights": vec![1u64; group.rank],
        "rank": group.rank,
        "order": group.order as u64,
        "N": group.n_pos,
        "ncells": res.cells.len(),
        "nstarreps": res.n_star_reps,
        "cells": cells,
    })
}

/// Serialize a star-rep [`CellGraph`](crate::cellgraph::CellGraph) W-graph to a
/// self-describing JSON value (Task Q1 `--save-reps`).
///
/// The W-graph is the mathematical payload of a `klcells` run.  This emits every
/// field needed to reconstruct it offline:
/// - `n`: vertex count;
/// - `x`: vertex reduced words (each a list of generator indices);
/// - `xrep`: each vertex's `coxelm` identity (a list of root indices);
/// - `isets`: each vertex's left-descent set;
/// - `weights`: generator weights (all 1 for equal parameters);
/// - `mpols`: per-generator mu pools (`{"v":..,"c":[..]}` Laurents);
/// - `mmat`: W-graph edges as `[[y, x], [pool_idx per generator]]` entries,
///   sorted by `(y, x)` for a deterministic, diff-friendly image.
pub fn cellgraph_json(cg: &crate::cellgraph::CellGraph) -> Value {
    let x: Vec<Value> =
        cg.x.iter()
            .map(|w| Value::from(w.iter().map(|&s| s as u64).collect::<Vec<_>>()))
            .collect();
    let xrep: Vec<Value> = cg
        .xrep
        .iter()
        .map(|ce| Value::from(ce.0.iter().map(|&r| r as u64).collect::<Vec<_>>()))
        .collect();
    let isets: Vec<Value> = cg
        .isets
        .iter()
        .map(|s| Value::from(s.iter().map(|&g| g as u64).collect::<Vec<_>>()))
        .collect();
    let mpols: Vec<Value> = cg
        .mpols
        .iter()
        .map(|pool| {
            Value::Array(
                pool.iter()
                    .map(|l| serde_json::to_value(l).unwrap())
                    .collect(),
            )
        })
        .collect();

    // mmat sorted by (y, x) for determinism.
    let mut keys: Vec<(u32, u32)> = cg.mmat.keys().copied().collect();
    keys.sort_unstable();
    let mmat: Vec<Value> = keys
        .iter()
        .map(|&(y, x)| {
            let row = &cg.mmat[&(y, x)];
            json!([[y, x], row])
        })
        .collect();

    json!({
        "schema": "rustcox-wgraph-v1",
        "n": cg.x.len(),
        "x": x,
        "xrep": xrep,
        "isets": isets,
        "weights": cg.weights,
        "mpols": mpols,
        "mmat": mmat,
    })
}

// ---------------------------------------------------------------------------
// Import: parse "type" and "weights" from golden JSON
// ---------------------------------------------------------------------------

/// Parse the golden `"type"` field into `(Series, rank)` pairs suitable for
/// [`CoxeterGroup::from_components`].
pub fn components_from_type_json(v: &Value) -> Result<Vec<(Series, usize)>, crate::cartan::Error> {
    let arr = v.as_array().ok_or_else(|| {
        crate::cartan::Error::ParseError(String::new(), "\"type\" must be an array".to_string())
    })?;
    arr.iter()
        .map(|item| {
            let series_str = item["series"].as_str().ok_or_else(|| {
                crate::cartan::Error::ParseError(
                    String::new(),
                    "component missing \"series\"".to_string(),
                )
            })?;
            let rank = item["rank"].as_u64().ok_or_else(|| {
                crate::cartan::Error::ParseError(
                    String::new(),
                    "component missing \"rank\"".to_string(),
                )
            })? as usize;
            let series = match series_str {
                "A" => Series::A,
                "B" => Series::B,
                "C" => Series::C,
                "D" => Series::D,
                "E" => Series::E,
                "F" => Series::F,
                "G" => Series::G,
                "H" => Series::H,
                "I" => {
                    let m = item["m"].as_u64().ok_or_else(|| {
                        crate::cartan::Error::ParseError(
                            String::new(),
                            "I-type component missing \"m\"".to_string(),
                        )
                    })? as u32;
                    Series::I(m)
                }
                other => {
                    return Err(crate::cartan::Error::UnknownSeries(other.to_string()));
                }
            };
            Ok((series, rank))
        })
        .collect()
}

/// Build a [`CoxeterGroup`] from the golden `"type"` JSON value.
pub fn group_from_type_json(v: &Value) -> Result<CoxeterGroup, crate::cartan::Error> {
    let components = components_from_type_json(v)?;
    CoxeterGroup::from_components(&components)
}

/// Parse the golden `"weights"` JSON value into a `Vec<u32>`.
///
/// If `v` is `null` or absent (i.e. the key was not present) this returns
/// an all-ones vector of length `rank`.
///
/// Returns [`IoError::NonIntegerWeight`] if any element is not a non-negative
/// integer.
pub fn weights_from_json(v: &Value, rank: usize) -> Result<Vec<u32>, IoError> {
    match v.as_array() {
        Some(arr) => arr
            .iter()
            .enumerate()
            .map(|(index, w)| {
                w.as_u64()
                    .map(|n| n as u32)
                    .ok_or_else(|| IoError::NonIntegerWeight {
                        index,
                        value: w.clone(),
                    })
            })
            .collect(),
        None => Ok(vec![1u32; rank]),
    }
}

// ---------------------------------------------------------------------------
// mu pool synthesis + mumat remap (internal helpers)
// ---------------------------------------------------------------------------

/// Build `mumat` JSON using pre-computed `uniq_per_gen` pools.
fn mumat_json_from_pools(
    t: &KlTable,
    n: usize,
    rank: usize,
    uniq_per_gen: &MuPoolKeys,
) -> Vec<Value> {
    let mut mumat: Vec<Value> = Vec::with_capacity(n);
    for w in 0..n {
        let mut wrow: Vec<Value> = Vec::with_capacity(w + 1);
        for y in 0..=w {
            let entry: Vec<i64> = (0..rank)
                .map(|s| {
                    if y == w {
                        return -1;
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
