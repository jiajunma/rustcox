//! `rustcox verify <FILE> --against <GOLDEN>` — compare two JSON documents.

use anyhow::Context;
use serde_json::Value;

use super::kl::read_json_doc;

/// Compare `file` against `golden`. Returns `Ok(true)` if they match,
/// `Ok(false)` on mismatch (with a human-readable description printed to
/// stdout), or `Err` on I/O or parse failure.
pub fn run(file: &str, against: &str) -> anyhow::Result<bool> {
    let a = read_json_doc(file).with_context(|| format!("reading '{file}'"))?;
    let b = read_json_doc(against).with_context(|| format!("reading '{against}'"))?;

    if a == b {
        println!("match: {file} == {against}");
        return Ok(true);
    }

    // Find the first differing top-level key (or array index).
    let diff_desc = find_first_diff(&a, &b);
    println!("mismatch: {diff_desc}");
    Ok(false)
}

/// Produce a human-readable description of the first difference between two
/// `Value`s at the top level (object keys) or, for arrays, at the first
/// differing index.
fn find_first_diff(a: &Value, b: &Value) -> String {
    match (a, b) {
        (Value::Object(ao), Value::Object(bo)) => {
            // Check for keys present in one but not the other
            let mut keys_a: Vec<&str> = ao.keys().map(|k| k.as_str()).collect();
            keys_a.sort_unstable();
            let mut keys_b: Vec<&str> = bo.keys().map(|k| k.as_str()).collect();
            keys_b.sort_unstable();

            // Keys in a but not b
            for k in &keys_a {
                if !bo.contains_key(*k) {
                    return format!("key '{k}' present in file but missing in golden");
                }
            }
            // Keys in b but not a
            for k in &keys_b {
                if !ao.contains_key(*k) {
                    return format!("key '{k}' missing in file but present in golden");
                }
            }
            // Values differing
            for k in &keys_a {
                let va = &ao[*k];
                let vb = &bo[*k];
                if va != vb {
                    // Check if it's an array and give index info
                    if let (Value::Array(aa), Value::Array(ba)) = (va, vb) {
                        if aa.len() != ba.len() {
                            return format!("key '{k}': array length {} != {}", aa.len(), ba.len());
                        }
                        for (i, (x, y)) in aa.iter().zip(ba.iter()).enumerate() {
                            if x != y {
                                return format!("key '{k}': first difference at index {i}");
                            }
                        }
                    }
                    return format!("key '{k}' differs");
                }
            }
            "documents differ (no specific key identified)".to_string()
        }
        (Value::Array(aa), Value::Array(ba)) => {
            if aa.len() != ba.len() {
                return format!("array length {} != {}", aa.len(), ba.len());
            }
            for (i, (x, y)) in aa.iter().zip(ba.iter()).enumerate() {
                if x != y {
                    return format!("first difference at index {i}");
                }
            }
            "arrays differ (no specific index identified)".to_string()
        }
        _ => format!("type mismatch: {:?} vs {:?}", a, b),
    }
}
