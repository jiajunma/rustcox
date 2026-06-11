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

### `rustcox kl` (full KL table)

Memory scales as |W|² for the pol-id matrix (u32 per pair):

| Group | \|W\| | pol-id matrix | Notes |
|-------|--------|---------------|-------|
| B4    |    384 | ~0.6 MB       | trivial |
| F4    |  1 152 | ~2.7 MB       | trivial |
| H4    | 14 400 | ~415 MB       | fine on a laptop; minutes parallel |
| D6    | 23 040 | ~1.1 GB       | HPC node |
| E6    | 51 840 | ~5.4 GB       | fat node, experimental |
| E7/E8 | —      | out of scope  | full pol-id matrix is TB-scale; use `rustcox cells` |

### `rustcox cells` (parabolic induction — Phase 2)

The cells driver avoids the |W|² full-table matrix; memory is dominated by
the relative-KL polynomial pool per star-class representative:

| Group | \|W\| | Peak RSS (t=64) | Notes |
|-------|--------|----------------|-------|
| H4    | 14 400 | 74 MB          | XMU cluster, 2026-06-11 |
| D6    | 23 040 | 33 MB          | XMU cluster, 2026-06-11 |
| B6    | 46 080 | 74 MB          | XMU cluster, 2026-06-11 |
| E6    | 51 840 | 75 MB          | XMU cluster, 2026-06-11 |
| E7    | 2 903 040 | 6.5 GB      | XMU cluster, 2026-06-11; output `results/cells_E7.json.gz` |
| E8    | 696 729 600 | streaming long run | `cells E8 --stream` + checkpoint; see below |

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
sufficient threads. The **concrete, versioned** scripts actually used at XMU are
`hpc/h4_determinism.sbatch` and `hpc/big_groups.sbatch` (account/partition/qos
already filled in); the block above is a bare template.

## Cells ladder (Phase 2 results, XMU cluster 2026-06-11)

All runs: `rustcox cells <Group> --threads 64 -o <output>`, equal parameters,
on one `cpu`-partition node (2× Xeon Gold 6338, 64 cores, 256 GB).

| Group | \|W\| | Compute (t=64) | Wall | Peak RSS | Left cells | Star reps | Validated against |
|-------|--------|---------------|------|----------|------------|-----------|-------------------|
| H4    | 14 400 | 1.26 s        | —    | 74 MB    | 206        | 90        | Phase-1 archive (byte-identical) |
| D6    | 23 040 | 0.25 s        | —    | 33 MB    | 578        | —         | PyCox full-table |
| B6    | 46 080 | 0.88 s        | —    | 74 MB    | 752        | —         | PyCox full-table |
| E6    | 51 840 | 0.62 s        | —    | 75 MB    | 652        | 21        | PyCox full-table |
| **E7** | **2 903 040** | **61.1 s** | **71 s** | **6.5 GB** | **6364** | **56** | PyCox / Geck literature (~235× speedup) |

E7 output is archived on-cluster at `results/cells_E7.json.gz` (10 MB).

## E7 recipe

```bash
# On the XMU login node (after rsync + cargo build --release):
sbatch hpc/cells_e7.sbatch
# Monitor:
squeue -u majj
# Pull results:
rsync -az majj@10.26.14.64:/public/home/majj/rustcox/results/ ./results/
```

The `hpc/cells_e7.sbatch` script requests 64 CPUs, 200 GB memory, and a 12-hour
wall-clock limit (generous headroom; actual peak RSS ~6.5 GB and wall time ~71 s).
It writes `results/cells_E7.json.gz`.

## E8 cells — streaming long run (Task Q1)

E8 left cells (`|W| = 696,729,600`, expected 101796 cells / 106 star-reps) were
historically out of reach: holding all cells in RAM and the relative-KL matrices
for the large E7-induced cells together blow past memory.  The Task-Q1 streaming
driver removes the RAM wall — each cell is streamed to `results/e8_cells.jsonl.gz`
as it is found, and the compact persistent loop state (the involution skip-set
`celms`, `tot`, the rep index, the star-rep registry, and the stream record
count) is checkpointed after every W1-rep.

Run it with `hpc/cells_e8_long.sbatch` (`--time=7-00:00:00`, `--mem=250G`):

```bash
sbatch hpc/cells_e8_long.sbatch        # first submission
sbatch hpc/cells_e8_long.sbatch        # after a timeout — AUTO-RESUMES
```

The job is **resume-by-resubmit**: on restart the binary loads the matching
checkpoint in `results/e8_ckpt/`, recomputes the cheap E7 recursion (~60 s) to
rebuild the W1 star-reps, restores the loop state, truncates the stream to the
checkpointed record count, and replays from the checkpointed rep.  SIGTERM/SIGINT
are not trapped (no `unsafe`, no extra crates); kill-safety rests on
checkpoint-after-every-rep plus the inner layer-log below.

### Layer-granular checkpoint inside `relklpols` (Task Q4)

The driver-level checkpoint recovers *between* reps, but one E8 W1-rep's
induction step can itself run for hours or days — longer than a 4-day SLURM box.
Re-running such a monster rep from scratch on every resume would **livelock**:
the rep never finishes inside one box, so the driver never advances.  Task Q4
removes that wall by checkpointing *inside* the inner `relklpols` call, one
wavefront layer at a time.

When streaming + checkpointing are both on, each rep's inner call writes a
per-rep **block log** at `results/e8_ckpt/relkl/repNNNNNN.{blklog,blkhdr}`: after
each completed layer `y` it appends that layer's finalized off-diagonal blocks
plus the pool deltas (new `rklpols`/`mues` entries), fsyncs the log, then
atomically rewrites a tiny side header (version, a `(group,W1,cell1)`
fingerprint, last completed layer, record count, log byte length, pool sizes).
On resume the deterministic setup is recomputed fresh, the log is replayed up to
the header's byte length (any trailing partial record from a crash mid-append is
ignored), the pool sizes are verified, and the wavefront continues at the next
uncomputed layer.  Output is **byte-identical** to an uninterrupted run for any
number of interruptions and any thread count.

Bounded disk: only the in-flight rep keeps a log — it is deleted on the rep's
clean completion, and stale logs for already-completed reps (`< next_rep`) are
removed on driver resume.  The driver checkpoint binary format and fingerprint
are untouched (additive new files only), so the live E8 run's existing
`results/e8_ckpt/klcells.ckpt` remains compatible.

**Honest limitation:** the checkpoint granularity is one wavefront layer.  A
single layer of a monster rep — `nx = |X1|` layers, with the last layers the
widest — could itself take hours; if a layer alone exceeds the box, the run still
livelocks at the layer level.  If that is ever observed, the next refinement is
intra-layer chunk logging (log partial Case-B block progress within a layer).
That is **not** implemented now; it is recorded here as the planned next step.

Storage to provision (see the header comment in the sbatch for the full
analysis): stream ~2.5–5 GB gz, reps ≤ ~50 GB, checkpoint ~10–20 MB, plus the
in-flight rep's block log (one rep's blocks + pools — bounded, deleted on rep
completion).

The star-rep W-graphs land in `results/e8_reps/reps/NNNNNN.json.gz` — this
W-graph data is the mathematical payload of the computation.  Final
canonicalization of the cell stream (re-reduce + sort) is done offline/post-hoc;
the stream's on-disk order is processing order (rep ascending, then component,
then star-orbit), which is deterministic per the P5 BTreeMap fix.

## XMU cluster access

The H4 / rank-6 numbers in [BENCHMARKS.md](BENCHMARKS.md) were produced on the
Xiamen University HPC. Concrete workflow, recorded so a fresh agent can
reproduce it without re-deriving the cluster layout.

### Cluster facts

- **Scheduler:** SLURM (`sbatch` / `squeue` / `sinfo` in `/usr/bin`).
- **Login node** `mu012` — 2 cores, 8 GB RAM. **Build / sync only; never run a
  KL job here.** It has internet (crates.io + rustup reachable); compute nodes
  do **not**.
- **`cpu` partition** (default): ~389 nodes, ~64 cores/node (2× Xeon Gold 6338),
  usually near-full, so queue waits are normal. Also `fat` (large memory) and
  `gpu`.
- **Account / QOS:** `-A yushilingroup`, `--qos=normal` (or `long`) — already
  baked into the `hpc/*.sbatch` directives.
- **`$HOME = /public/home/majj`** on Lustre (`/public`, ~1 TB, no per-user
  quota), **shared across all nodes** → a binary built once on the login node is
  visible to every compute node with no extra transfer.

### Connection

```bash
ssh majj@10.26.14.64        # login node mu012
```

(rustcox is a private repo, so this host is recorded directly.)

### Workflow — Mac = git authority, HPC = compute copy

The Mac repo is the git authority; the cluster copy is synced with `rsync`
(keeps no GitHub token on the cluster). The Mac's `target/` is a Darwin/arm64
build — **never sync it up**; build fresh on the login node (x86_64 Linux).

```bash
# 1. sync code up (exclude git, the wrong-arch target, and results)
rsync -az --exclude='.git' --exclude='target' --exclude='results' \
  ./ majj@10.26.14.64:/public/home/majj/rustcox/

# 2. on the login node: build once (internet here; binary lands on shared $HOME)
ssh majj@10.26.14.64
cd ~/rustcox
export PATH="$HOME/.cargo/bin:$PATH"   # rustup installed user-space
rustup update stable
cargo build --release                  # → target/release/rustcox, visible to compute nodes

# 3. submit (SBATCH directives already set account/partition/qos/cpus)
sbatch hpc/h4_determinism.sbatch       # H4 seq-vs-parallel determinism proof
sbatch hpc/big_groups.sbatch           # D6 / B6 / E6 rank-6 tables
squeue -u majj                         # monitor; stdout → results/<job-name>-<id>.out

# 4. back on the Mac: pull results down and commit locally
rsync -az majj@10.26.14.64:/public/home/majj/rustcox/results/ ./results/
```

If the login-node linker complains, `module load gcc/12.1` (or 11.4 / 14.2)
before `cargo build`. No conda / Julia is needed — rustcox is pure Rust.

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

- **E7 / E8 full KL tables (`rustcox kl`):** |W| = 2 903 040 / 696 729 600 —
  full pol-id matrices are TB-scale and impractical.  Use `rustcox cells`
  instead (E7 completes in ~71 s; E8 cells run via the streaming long-run job).
- **E8 cells in whole-document `-o` mode (`rustcox cells E8 -o ...`):** the whole
  partition cannot be held in RAM and canonicalized.  Use the streaming path
  (`--stream` + `--checkpoint-dir`, `hpc/cells_e8_long.sbatch`) — see
  *E8 cells — streaming long run* above.
- **Unequal parameters on H4:** not blocked by code, but mu storage cost has
  not been profiled. Use caution and monitor memory.
