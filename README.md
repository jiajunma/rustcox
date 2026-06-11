# rustcox

A Rust rewrite of [PyCox](https://github.com/geckmf/PyCox) (M. Geck's Python
port of GAP-CHEVIE) focused on **Kazhdan–Lusztig polynomials, mu-coefficients,
left/right/two-sided cells and Duflo involutions** for all finite Coxeter groups,
with equal *and* unequal parameters — designed for multi-threaded computation
on HPC machines.

**Status: Phase 1 (tasks 0–18) + Phase 2 (tasks P1–P8) complete.**

## Quick start

```bash
export PATH="$HOME/.cargo/bin:$PATH"   # macOS: rustup toolchain in user-space
cargo build --release

# Group info
rustcox info F4

# KL table summary
rustcox kl B4 --summary

# Write compressed golden JSON
rustcox kl F4 -o f4.json.gz --threads 8

# Cells by parabolic induction (Phase 2)
rustcox cells E7 --threads 64 -o e7.json.gz

# Full golden self-test against golden/ directory
rustcox selftest
```

## Feature matrix

| Feature | Status | Notes |
|---------|--------|-------|
| KL polynomials, equal parameters | done | all finite types incl. I₂(m) for all m |
| KL polynomials, unequal parameters | done | negative coefficients supported |
| mu-coefficients | done | Implicit (equal) and Stored (unequal) modes |
| Left/right/two-sided cells | done | Tarjan SCC — same output as PyCox |
| Duflo involutions | done | a-values, n-values, sign checks |
| Cell preorder (lorder) | done | condensation DAG reachability |
| Minimal W-graphs | done | per-cell and decompose |
| All finite types (A–H, I₂(m)) | done | I₂(m) uses CycInt for m ∉ {3,4,5,6} |
| Canonical JSON I/O (gz-transparent) | done | schema `rustcox-golden-v1` |
| Deterministic parallel driver | done | byte-identical to sequential |
| CLI: info, kl, verify, selftest, bench-kl, cells | done | |
| relklpols (relative KL polynomials) | done | equal parameters; parabolic induction |
| Cells by parabolic induction | done | E7-validated; 235× faster than PyCox |
| Star operations / star orbits | done | `star.rs`; generalised tau |
| Hecke characters / chartable | not ported | future work |
| E8 cells | structural limit | relkl matrix for large E7 cells is TB-scale; see docs/HPC.md |

## Cells at scale

Cells by parabolic induction (`rustcox cells`) on the XMU cluster (64-core Intel
node, 256 GB, 2026-06-11):

| Group | \|W\| | Compute (t=64) | Wall | Peak RSS | Left cells | Star-class reps | vs PyCox |
|-------|--------|---------------|------|----------|------------|-----------------|----------|
| H4    | 14 400 | 1.26 s        | —    | 74 MB    | 206        | 90              | validated |
| D6    | 23 040 | 0.25 s        | —    | 33 MB    | 578        | —               | validated |
| B6    | 46 080 | 0.88 s        | —    | 74 MB    | 752        | —               | validated |
| E6    | 51 840 | 0.62 s        | —    | 75 MB    | 652        | 21              | validated |
| **E7** | **2 903 040** | **61.1 s** | **71 s** | **6.5 GB** | **6364** | **56** | **~235× faster than PyCox** |

All left-cell counts and star-class representative counts match the published
PyCox / Geck values exactly. The H4 partition is byte-identical to the Phase-1
full-table archive. E7 output is archived on-cluster at `results/cells_E7.json.gz`
(10 MB). E8 is a known structural limit (see [docs/HPC.md](docs/HPC.md)).

## Performance

Sequential driver on Apple Silicon (M-series), compared to PyCox single-threaded
CPython 3 on similar hardware:

| Group | \|W\| | rustcox seq | PyCox | Speedup |
|-------|--------|-------------|-------|---------|
| B4    |    384 | 6.35 ms     | 1.0 s | ~157×   |
| A5    |    720 | 14.9 ms     | 2.5 s | ~168×   |
| F4    |  1 152 | 66 ms       | 28.9 s | ~438×  |

Parallel driver on F4 (baseline = sequential 66 ms):

| Threads | Median | Speedup |
|---------|--------|---------|
| 2       | 50.1 ms | 1.32× |
| 4       | 45.5 ms | 1.46× |
| 8       | 44.7 ms | 1.48× |

Parallel scaling is modest for F4 (25 length layers, max width 94); it
improves significantly for larger groups with wider layers. See
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full numbers.

## Verification model

Every observable result (element tables, Bruhat order, every KL polynomial,
every mu slot, cells, Duflo involutions) is compared against canonical JSON
produced by running the vendored PyCox under CPython:

```bash
cd pycox-ref
python3 gen_golden.py suite        # regenerate small/medium golden files
python3 gen_golden.py suite-big    # + A5, F4 (gzipped)
```

The Rust test-suite (`cargo test --workspace`) consumes these files.  The
parallel KL driver is deterministic: its output is bit-identical to the
sequential reference implementation across all thread counts.  The cells driver
is verified by a PyCox golden suite (7 `cells_*.json` files) and a byte-level
relkl oracle fixture.  See [docs/VERIFICATION.md](docs/VERIFICATION.md) for the
full pipeline.

## Repository layout

| Path | Content |
|------|---------|
| `crates/rustcox-core/` | library: Laurent polynomials, root systems, Coxeter group calculus, KL engine, cells by induction |
| `crates/rustcox-cli/` | `rustcox` binary (info, kl, verify, selftest, bench-kl, cells) |
| `pycox-ref/` | vendored PyCox (the oracle) + golden-data generator `gen_golden.py` |
| `golden/` | canonical JSON golden files — **never edit by hand** |
| `hpc/` | versioned SLURM scripts (`cells_e7.sbatch`, `cells_e8_experiment.sbatch`, …) |
| `docs/BENCHMARKS.md` | measured timings and acceptance gates |
| `docs/DESIGN.md` | as-built architecture and key design decisions |
| `docs/VERIFICATION.md` | oracle pipeline and how to add golden cases |
| `docs/HPC.md` | build, threading, memory, and SLURM notes |

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md) — module map, element representations, KL recursion overview,
  deterministic parallel design, CycInt, known deviations from PyCox
- [docs/VERIFICATION.md](docs/VERIFICATION.md) — golden pipeline, canonicalisation spec, how to add cases
- [docs/HPC.md](docs/HPC.md) — build instructions, thread control, memory table, SLURM sample

## License

GPL-3.0-or-later. rustcox is a derived work of PyCox (GPL-3),
© 2011–2014 Meinolf Geck; see `pycox-ref/README.md` for provenance.
