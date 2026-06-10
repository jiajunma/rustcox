//! Shared test helpers for integration tests.
//!
//! Each integration test binary compiles this module independently, so a
//! helper unused by a given binary (e.g. `parallel_eq.rs` does not read golden
//! files) would otherwise warn as dead code.  These helpers are part of the
//! shared test surface; allow the per-binary dead-code warning.
#![allow(dead_code)]

use std::path::PathBuf;

use rustcox_core::cartan::Series;

/// Load a golden JSON file by name (without `.json` extension).
///
/// Looks for `<repo-root>/golden/<name>.json` first; falls back to
/// `<name>.json.gz` for large golden files.
pub fn golden(name: &str) -> serde_json::Value {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden");
    let plain = root.join(format!("{name}.json"));
    if plain.exists() {
        let text = std::fs::read_to_string(&plain)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", plain.display()));
        serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("failed to parse {}: {e}", plain.display()))
    } else {
        let gz = root.join(format!("{name}.json.gz"));
        let f = std::fs::File::open(&gz)
            .unwrap_or_else(|e| panic!("golden file not found ({}): {e}", gz.display()));
        let mut s = String::new();
        use std::io::Read;
        flate2::read::GzDecoder::new(f)
            .read_to_string(&mut s)
            .unwrap_or_else(|e| panic!("failed to decompress {}: {e}", gz.display()));
        serde_json::from_str(&s).unwrap_or_else(|e| panic!("failed to parse {}: {e}", gz.display()))
    }
}

/// Parse the `"type"` field of a golden file into `(Series, rank)` pairs.
///
/// Delegates to `io::components_from_type_json` — single source of truth.
pub fn components_of(g: &serde_json::Value) -> Vec<(Series, usize)> {
    rustcox_core::io::components_from_type_json(&g["type"])
        .unwrap_or_else(|e| panic!("failed to parse golden \"type\": {e}"))
}
