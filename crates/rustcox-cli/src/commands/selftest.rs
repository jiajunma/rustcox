//! `rustcox selftest [--golden-dir DIR]` — run golden-file self-tests.

use std::path::{Path, PathBuf};

use anyhow::Context;
use rustcox_core::{
    io::{basics_json, group_from_type_json, to_canonical_json, weights_from_json},
    kl::{cells::CellData, klpolynomials, KlOpts},
};
use serde_json::Value;

use super::kl::read_json_doc;

pub fn run(golden_dir: &Path) -> anyhow::Result<bool> {
    let entries = collect_golden_files(golden_dir)?;

    if entries.is_empty() {
        anyhow::bail!("no golden files found in '{}'", golden_dir.display());
    }

    let mut all_pass = true;

    for path in &entries {
        let fname = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_string();

        let result = run_one(path);
        match result {
            OneResult::Pass => {
                println!("PASS  {fname}");
            }
            OneResult::Fail(msg) => {
                println!("FAIL  {fname}: {msg}");
                all_pass = false;
            }
            OneResult::Skip(reason) => {
                println!("SKIP  {fname}: {reason}");
            }
        }
    }

    if all_pass {
        println!("\nAll tests PASSED.");
    } else {
        println!("\nSome tests FAILED.");
    }

    Ok(all_pass)
}

enum OneResult {
    Pass,
    Fail(String),
    Skip(String),
}

fn run_one(path: &Path) -> OneResult {
    let path_str = path.to_str().unwrap_or("?");
    let doc = match read_json_doc(path_str) {
        Ok(v) => v,
        Err(e) => return OneResult::Fail(format!("read error: {e}")),
    };
    run_one_from_value(&doc)
}

fn run_kl_one(doc: &Value) -> OneResult {
    let type_val = match doc.get("type") {
        Some(v) => v,
        None => return OneResult::Fail("missing 'type' field".to_string()),
    };

    // Check for CycInt (I with m not in {3,4,5,6})
    if needs_cyc_int(type_val) {
        return OneResult::Skip("needs CycInt".to_string());
    }

    let group = match group_from_type_json(type_val) {
        Ok(g) => g,
        Err(e) => return OneResult::Fail(format!("group construction: {e}")),
    };

    let weights = match doc.get("weights") {
        Some(w) => match weights_from_json(w, group.rank) {
            Ok(wv) => wv,
            Err(e) => return OneResult::Fail(format!("weights: {e}")),
        },
        None => vec![1u32; group.rank],
    };

    let opts = KlOpts {
        weights,
        threads: None,
        layer_chunk: None,
    };

    match opts.validate(&group) {
        Ok(()) => {}
        Err(e) => return OneResult::Fail(format!("opts validate: {e}")),
    }

    let table = match klpolynomials(&group, &opts) {
        Ok(t) => t,
        Err(e) => return OneResult::Fail(format!("klpolynomials: {e}")),
    };

    let cells = CellData::from_table(&table);
    let computed = to_canonical_json(&table, &cells, &group);

    if computed == *doc {
        OneResult::Pass
    } else {
        let diff = find_first_kl_diff(&computed, doc);
        OneResult::Fail(format!("mismatch: {diff}"))
    }
}

fn run_basics_one(doc: &Value) -> OneResult {
    let type_val = match doc.get("type") {
        Some(v) => v,
        None => return OneResult::Fail("missing 'type' field".to_string()),
    };

    // Basics files don't have I2(m) with exotic m, but guard anyway
    if needs_cyc_int(type_val) {
        return OneResult::Skip("needs CycInt".to_string());
    }

    let group = match group_from_type_json(type_val) {
        Ok(g) => g,
        Err(e) => return OneResult::Fail(format!("group construction: {e}")),
    };

    let computed = basics_json(&group);

    if computed == *doc {
        OneResult::Pass
    } else {
        let diff = find_first_kl_diff(&computed, doc);
        OneResult::Fail(format!("mismatch: {diff}"))
    }
}

/// Returns `true` if the type JSON contains an I-series component with `m`
/// not in `{3, 4, 5, 6}` (those require CyclotomicInteger support).
fn needs_cyc_int(type_val: &Value) -> bool {
    let arr = match type_val.as_array() {
        Some(a) => a,
        None => return false,
    };
    for comp in arr {
        let series = comp.get("series").and_then(|v| v.as_str()).unwrap_or("");
        if series == "I" {
            let m = comp.get("m").and_then(|v| v.as_u64()).unwrap_or(0);
            if !matches!(m, 3..=6) {
                return true;
            }
        }
    }
    false
}

/// Collect all `kl_*.json`, `kl_*.json.gz`, `basics_*.json`, `basics_*.json.gz`
/// files in the given directory, sorted by filename.
fn collect_golden_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read golden directory '{}'", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            (name.starts_with("kl_") || name.starts_with("basics_"))
                && (name.ends_with(".json") || name.ends_with(".json.gz"))
        })
        .collect();
    entries.sort();
    Ok(entries)
}

/// Internal helper used by tests: run a single golden doc given as a `Value`.
fn run_one_from_value(doc: &Value) -> OneResult {
    let kind = match doc.get("kind").and_then(|v| v.as_str()) {
        Some(k) => k.to_string(),
        None => return OneResult::Fail("missing 'kind' field".to_string()),
    };

    match kind.as_str() {
        "kl" => run_kl_one(doc),
        "basics" => run_basics_one(doc),
        other => OneResult::Fail(format!("unknown kind '{other}' in golden file")),
    }
}

/// Find the first difference between two JSON objects (for error reporting).
fn find_first_kl_diff(a: &Value, b: &Value) -> String {
    match (a, b) {
        (Value::Object(ao), Value::Object(bo)) => {
            for (k, va) in ao {
                match bo.get(k) {
                    None => return format!("key '{k}' present in computed but missing in golden"),
                    Some(vb) if va != vb => {
                        if let (Value::Array(aa), Value::Array(ba)) = (va, vb) {
                            if aa.len() != ba.len() {
                                return format!("key '{k}': length {} != {}", aa.len(), ba.len());
                            }
                            for (i, (x, y)) in aa.iter().zip(ba.iter()).enumerate() {
                                if x != y {
                                    return format!("key '{k}': first diff at index {i}");
                                }
                            }
                        }
                        return format!("key '{k}' differs");
                    }
                    _ => {}
                }
            }
            "documents differ (no key identified)".to_string()
        }
        _ => "value type mismatch".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unknown_kind_is_fail_not_skip() {
        // A golden doc with an unrecognised "kind" must produce Fail, not Skip.
        let doc = json!({"kind": "future_thing", "type": [{"series": "A", "rank": 2}]});
        match run_one_from_value(&doc) {
            OneResult::Fail(msg) => {
                assert!(
                    msg.contains("future_thing"),
                    "Fail message should name the kind, got: {msg}"
                );
            }
            OneResult::Skip(_) => panic!("unknown kind should be Fail, not Skip"),
            OneResult::Pass => panic!("unknown kind should be Fail, not Pass"),
        }
    }

    #[test]
    fn cyc_int_i2m_is_skip() {
        // I2(7) needs CycInt and must produce Skip.
        let doc = json!({"kind": "kl", "type": [{"series": "I", "m": 7}]});
        match run_one_from_value(&doc) {
            OneResult::Skip(reason) => {
                assert!(
                    reason.contains("CycInt"),
                    "Skip reason should mention CycInt, got: {reason}"
                );
            }
            other => panic!(
                "I2(7) should be Skip, got {:?}",
                matches!(other, OneResult::Pass)
            ),
        }
    }
}
