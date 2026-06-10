# rustcox — HPC Guide

## Build

```bash
# macOS / Linux: rustup toolchain required (system rustc may be too old)
export PATH="$HOME/.cargo/bin:$PATH"   # required if rustup was installed user-space
rustup update stable

cargo build --release
# Binary: target/release/rustcox
```

The project targets Rust 2021 (MSRV 1.75). The macOS system rustc at
`/opt/local/bin/rustc` (MacPorts) is ancient; always use the rustup binary.

## Thread control

The parallel driver (`kl/parallel.rs`) accepts two mechanisms:

**CLI:**
```bash
rustcox kl F4 --threads 8 -o f4.json.gz
```

**Library `KlOpts::threads`:**
```rust
KlOpts { threads: Some(8), .. }
```

When `threads = None`, the global Rayon thread pool is used (defaults to the
number of logical CPUs). When `threads = Some(t)`, a private Rayon pool of `t`
threads is built for that call. `threads = Some(0)` or `Some(1)` falls back to
the sequential driver.

**Environment variable:** `RAYON_NUM_THREADS=N` sets the global Rayon pool
size and is honoured when `threads = None` (i.e. when the CLI omits
`--threads`). Both mechanisms are valid for HPC use.

## Determinism guarantee and its cost

The parallel output is **byte-identical** to the sequential output: same pool
insertion order, same polynomial ids, same mu ids. This is guaranteed by the
two-phase design:

1. **Phase 1 (parallel):** rows computed with inline `Laurent` values; no shared
   pool writes; reads only frozen lower layers.
2. **Phase 2 (sequential):** inline values interned into shared pools in
   deterministic order (`w` ascending, `y` descending per row).

**Cost:** one sequential barrier per length layer. For F4 this is 25 barriers;
for H4 it is the longest element's length. The Amdahl fraction of the intern
phase shrinks as layers get wider (more rows per layer), so larger groups
benefit more from parallelism.

## Memory expectations

Memory scales as |W|² for the pol-id matrix (u32 per pair):

| Group | \|W\| | pol-id matrix | Notes |
|-------|--------|---------------|-------|
| B4    |    384 | ~0.6 MB       | trivial |
| F4    |  1 152 | ~2.7 MB       | trivial |
| H4    | 14 400 | ~415 MB       | fine on a laptop; minutes parallel |
| D6    | 23 040 | ~1.1 GB       | HPC node |
| E6    | 51 840 | ~5.4 GB       | fat node, experimental |
| E7/E8 | —      | out of scope  | needs relklpols/cells induction |

Equal-parameter mode (`MuMode::Implicit`) does **not** allocate a mu array;
only per-row slot-presence bitmaps. Unequal-parameter mode (`MuMode::Stored`)
adds per-generator mu pools; the cost is proportional to the number of non-zero
mu slots.

Transient memory during phase 1 (inline Laurent values) is bounded by the
widest layer times the average row size; for F4 this is negligible. For very
large groups, `--layer-chunk K` limits peak in-flight units:

```bash
rustcox kl H4 --threads 16 --layer-chunk 64 -o h4.json.gz
```

## Sample SLURM script

```bash
#!/bin/bash
#SBATCH --job-name=rustcox-h4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --time=2:00:00

export PATH="$HOME/.cargo/bin:$PATH"

rustcox kl H4 --threads $SLURM_CPUS_PER_TASK -o h4.json.gz
```

Adjust `--mem` and `--time` for your cluster. H4 is the natural HPC target:
|W|=14 400, pol-id matrix ≈415 MB, run time estimated at a few minutes with
sufficient threads.

## Parallel scaling caveat

The F4 benchmark shows 1.46–1.48× speedup at t=4–8 on 66 ms total work. Three
causes:

1. Layer barriers: 25 sequential boundaries for F4.
2. The intern phase (phase 2) is sequential per layer (Amdahl limit).
3. Per-unit work is small at this scale.

For rank ≥ 5 groups with wider layers and longer rows, the Amdahl fraction
shrinks and speedup improves. See [BENCHMARKS.md](BENCHMARKS.md) for measured
data.

## What NOT to attempt

- **E7 / E8 full tables:** |W| = 2 903 040 / 696 729 600 — full pol-id
  matrices are impractical. The plan (Part 5) describes relklpols and cell
  induction for handling these groups, but that is not yet implemented.
- **Unequal parameters on H4:** not blocked by code, but mu storage cost has
  not been profiled. Use caution and monitor memory.
