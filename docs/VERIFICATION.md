# rustcox — Verification Pipeline

## Overview

Every observable result is compared against canonical JSON ("golden files")
produced by running the vendored PyCox oracle. The Rust test-suite fails if any
value diverges from the oracle.

## Oracle pipeline

```
pycox-ref/pycox_ref.py  (vendored, two patches applied)
        |
        v
pycox-ref/gen_golden.py (canonicalises + serialises)
        |
        v
golden/*.json[.gz]      (committed artifacts, never hand-edited)
        |
        v
cargo test --workspace  (golden_basics, golden_kl, parallel_eq, CLI selftest)
```

**PyCox patches** (applied once at vendoring):
1. Line ≈79: `mybytes=UInt[8]` → `mybytes=bytes`
2. Added `from functools import reduce`

These are the only modifications; the rest of `pycox_ref.py` is the verbatim
GPL-3 source.

## Generating golden files

```bash
cd pycox-ref

# All small/medium groups (< 30 s total):
python3 gen_golden.py suite

# + A5 and F4 as .json.gz (slow, large files):
python3 gen_golden.py suite-big

# Individual files:
python3 gen_golden.py kl B3:2,1,1   # B3, weights [2,1,1]
python3 gen_golden.py kl H3:1       # H3, equal weights
python3 gen_golden.py basics E6     # basics only (no KL)
```

## Canonicalisation spec

The normative spec is the docstring of `pycox-ref/gen_golden.py`. Summary:

- **Elements:** identified by canonical reduced word (smallest-left-descent
  strip). Sorted by `(length, word lex)`. All indices refer to this order.
- **Laurent polynomial:** `{"v": val, "c": [c0, c1, ...]}` where `c0` is
  the coefficient of `v^val`. Zero = `{"v": 0, "c": []}`.
- **Polynomial pools** (`pols`, `mues[s]`): deduplicated, sorted by
  `(val, coeffs)` with coeffs compared lexicographically. Matrices store
  indices into the sorted pools.
- **Sentinels:** `-1` for "not Bruhat-comparable" in klmat; `-1` for "no mu
  slot for this generator" in mumat.
- **Cells:** each cell is a sorted list of element indices; the list of cells
  is sorted lexicographically. `duflo` rows `[d, a, n]` follow the same cell
  order; `lorder` is permuted consistently.

The Rust exporter (`crates/rustcox-core/src/io.rs`, `to_canonical_json`) must
reproduce these bytes (compared as parsed `serde_json::Value`s, not raw bytes).

## Golden file inventory (44 files)

**Basics** (Cartan/Coxeter matrix, degrees, order, N, root coordinates,
length histogram, longest element when |W| ≤ 10 000):

A1–A5, B2–B4, C3, D4, D5, E6, F4, G2, H3, H4, I5, I7, I8, I10, I12

**KL equal parameters** (full pol/mu table, arrows, cells, Duflo, lorder):

A1–A4, B2–B4, C3, D4, G2, H3, I5, I7, I10

**KL unequal parameters:**

B2:[2,1], B2:[1,2], B2:[0,1], B3:[2,1,1], G2:[1,3], G2:[3,1], I8:[1,2]

**Big (gzipped, slow tests):**

A5 (kl+basics), F4 (kl+basics)

## How tests consume golden files

| Test binary | What it checks |
|-------------|---------------|
| `tests/golden_basics.rs` | Coxeter matrix, degrees, order, N, root coordinates, element enumeration, longest element |
| `tests/golden_kl.rs` | Element order, klmat (Bruhat + polynomials), mumat, arrows, lcells, rcells, tcells, duflo, lorder |
| `tests/parallel_eq.rs` | Parallel driver produces byte-identical KlTable to sequential; layer chunking is stable; threads=1 falls back |
| CLI selftest (`rustcox selftest`) | Reads all golden/*.json[.gz], rebuilds each group and computation, compares via io::from_canonical_json |

All golden tests run with `cargo test --workspace`. The slow big-file tests
(A5, F4) are `#[ignore]`d by default; run them with:

```bash
cargo test --workspace --release -- --include-ignored
```

## Adding a new golden case

1. Generate the file:
   ```bash
   cd pycox-ref
   python3 gen_golden.py kl D5:1    # or: basics D5
   ```
2. Verify the output is well-formed JSON with the expected `schema` field.
3. The `golden_kl.rs` / `golden_basics.rs` drivers discover files by glob;
   no Rust code change is needed for standard group types.
4. For a group type that requires new Rust support (e.g. a new series), add
   support first, then generate and commit the golden file.
5. Run `cargo test --workspace` to confirm the new case passes.
6. Commit both the golden file and any Rust changes together.

## The "never hand-edit" rule

Golden files are generated artifacts. Hand-editing them defeats the purpose of
oracle-based verification. If a value looks wrong, debug the Rust or Python
side; never patch the golden file directly. If the canonical format changes,
update `gen_golden.py` and `io.rs` together, regenerate everything, and bump
the `schema` field.
