//! Shared test helpers for integration tests.

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
/// Each element of the JSON array is an object with fields `"series"`, `"rank"`,
/// and optionally `"m"` (for I-type dihedral groups).
pub fn components_of(g: &serde_json::Value) -> Vec<(Series, usize)> {
    let arr = g["type"]
        .as_array()
        .expect("golden \"type\" field is not an array");
    arr.iter()
        .map(|item| {
            let series_str = item["series"]
                .as_str()
                .expect("golden component missing \"series\"");
            let rank = item["rank"]
                .as_u64()
                .expect("golden component missing \"rank\"") as usize;
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
                    let m = item["m"].as_u64().expect("I-type component missing \"m\"") as u32;
                    Series::I(m)
                }
                other => panic!("unknown series in golden: {other}"),
            };
            (series, rank)
        })
        .collect()
}
