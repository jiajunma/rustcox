# rustcox — KL engine benchmarks

## Machine

- **Hardware**: Apple Silicon (M-series)
- **OS**: macOS 14 (Sonoma)
- **Profile**: `cargo bench` (`--release` + `debug = true` symbols per workspace profile)
- **Date**: 2026-06-11

## Results

All times are Criterion **median** from the run recorded in `/tmp/bench.txt`.

### Sequential driver — `klpolynomials_seq`

| Group | \|W\| | Median time | Notes |
|-------|--------|-------------|-------|
| H3    |    120 | 928 µs      | sample_size = 100 |
| B4    |    384 | 6.35 ms     | sample_size = 100 |
| A5    |    720 | 14.9 ms     | sample_size = 10  |
| F4    |  1 152 | 66.4 ms     | sample_size = 10  |

### Parallel driver — `klpolynomials` on F4

| Threads | Median time | Speedup vs seq |
|---------|-------------|----------------|
| seq (1) | 66.4 ms     | 1.00× baseline |
| t = 2   | 50.1 ms     | 1.32×          |
| t = 4   | 45.5 ms     | 1.46×          |
| t = 8   | 44.7 ms     | 1.48×          |

**Note on parallel scaling**: F4 has 25 length layers (longest element has
length 24, so layers 0–24), with a maximum width of 94 elements and only 4 layers
thinner than 8 elements.  The modest 1.46–1.48× speedup has three causes:
(a) the parallel driver serialises across layer boundaries — 25 barriers for F4;
(b) the deterministic intern phase inside each layer is sequential (Amdahl limit);
(c) per-unit work is small at 66 ms total scale.  Speedups will improve for
larger groups (rank ≥ 5) where layers are wider, rows longer, and the
Amdahl fraction shrinks.  F4 is already well within the 1 s gate on a single
thread, so no further parallelism tuning is needed here.

### Unequal-parameter path — `klpolynomials_seq` on B3 [2,1,1]

| Group / weights | \|W\| | Median time |
|-----------------|--------|-------------|
| B3 [2, 1, 1]   |     48 | 218 µs      |

### Cell derivation — `CellData::from_table` on precomputed F4 table

| Input  | Median time |
|--------|-------------|
| F4 KL table | 4.69 ms |

## Acceptance gates (plan §0.4 / Task 16)

| Gate | Target | Result | Status |
|------|--------|--------|--------|
| F4 seq < 1 s | < 1 000 ms | 66 ms | PASS |
| F4 par(t=4) faster than seq | ratio > 1 | 1.46× | PASS |
| No further perf work needed | F4 seq ≥ 1 s trigger | 66 ms ≪ 1 s | no work needed |

## Comparison to PyCox baseline (plan §0.4)

PyCox timings are from a single-threaded CPython 3 run on comparable hardware
(cited from `docs/superpowers/plans/2026-06-10-rustcox-implementation.md §0.4`).

| Group | PyCox (s) | rustcox seq (ms) | Speedup |
|-------|-----------|------------------|---------|
| B4    | 1.0 s     | 6.35 ms          | ~157×   |
| A5    | 2.5 s     | 14.9 ms          | ~168×   |
| F4    | 28.9 s    | 66 ms            | ~438×   |

rustcox is roughly **150–440× faster** than PyCox across these groups,
driven by compiled Rust, integer arithmetic throughout (no Python object
overhead), and the interned polynomial pool avoiding redundant allocation.

## Phase 2 — cells by parabolic induction (`klcells`)

The `klcells` driver computes the left-cell partition by parabolic induction
(`relklpols` + W-graph decomposition + star-orbit expansion) **without
enumerating `W`'s full KL table**.  Times below are Criterion medians on the
same Apple-Silicon machine, `sample_size = 10`.

### `klcells` sequential

| Group | \|W\| | ncells | nstarreps | Median time |
|-------|--------|--------|-----------|-------------|
| H3    |    120 |  22    | 15        | 2.11 ms     |
| B4    |    384 |  50    | 22        | 4.42 ms     |
| F4    |  1 152 |  72    | 29        | 17.1 ms     |

### `klcells` F4 — sequential vs parallel relative-KL wavefront

| Threads | Median time | Speedup vs seq |
|---------|-------------|----------------|
| seq (1) | 17.1 ms     | 1.00× baseline |
| t = 4   | 14.7 ms     | 1.16×          |

The cells path is already ~4× faster than the full-table F4 KL run (66 ms),
because induction touches only the coset-rep × cell grid per star-class rep
rather than the whole `|W|²` Bruhat matrix.  The relative-KL wavefront
parallelises over the coset-rep index `x` within each `y`-layer with a
**deterministic two-phase intern** (compute Case-B blocks in parallel with
inline Laurent values, then intern sequentially in `(x desc, v, u)` order),
so the `cells` and `star_reps` output is **byte-identical** for any thread
count (verified for B4, H3, F4 × threads {2, 4} in `tests/cells_golden.rs`).
The modest F4 speedup reflects the small per-block work at the 17 ms scale and
the sequential intern phase (Amdahl); larger groups (rank ≥ 6) with wider
coset grids show the wavefront's value on the HPC ladder (Task P7).

## XMU HPC runs (2026-06-11, Intel 64-core node, 256 GB)

First full-table runs at scale, on one `cpu`-partition node (SLURM scripts in
`hpc/`). All runs: equal parameters, `checks_ok=true`.

| Group | \|W\| | seq compute | par t=64 | speedup | peak RSS | npols | lcells | tcells |
|-------|-------|-------------|----------|---------|----------|-------|--------|--------|
| H4    | 14 400 | 99.7 s     | 17.2 s   | 5.8×    | 47 GB*   | 726 635 | 206  | 13     |
| D6    | 23 040 | 61.6 s     | 16.3 s   | 3.8×    | 3.4 GB   | 5 836  | 578   | 27     |
| B6    | 46 080 | —          | 74.7 s   | —       | 13.8 GB  | 57 738 | 752   | 26     |
| E6    | 51 840 | —          | 100.8 s  | —       | 17.2 GB  | 46 681 | 652   | 17     |

\* H4 peak RSS is dominated by the canonical-JSON export (a ~2 GB document via
an in-memory `serde_json::Value` tree); the KL computation alone stays in the
single-digit GB range, consistent with the D6/B6/E6 rows (no export).

**Determinism proven at scale**: the H4 sequential and 64-thread parallel runs
produced **byte-identical** 1.96 GB canonical JSON documents (`cmp` clean).

**Independent mathematical cross-checks**: H4 has 206 left cells and 13
two-sided cells (Alvis, *The left cells of the Coxeter group of type H4*,
J. Algebra 107 (1987)); E6 has 17 two-sided cells — all reproduced exactly.

Parallel speedup grows with layer width as predicted (F4 1.5× → D6 3.8× →
H4 5.8× at 64 threads); the sequential intern phase is the remaining
Amdahl bottleneck.

## XMU HPC — `rustcox cells` (Phase 2, 2026-06-11)

Cells by parabolic induction on the same 64-core node.  All runs: equal
parameters, `--threads 64`.  SLURM scripts in `hpc/cells_medium.sbatch`,
`hpc/cells_e7.sbatch`.

| Group | \|W\| | Compute (t=64) | Wall | Peak RSS | Left cells | Star reps | Validation |
|-------|--------|---------------|------|----------|------------|-----------|------------|
| H4    | 14 400 | 1.26 s        | —    | 74 MB    | 206        | 90        | byte-identical to Phase-1 archive |
| D6    | 23 040 | 0.25 s        | —    | 33 MB    | 578        | —         | matches PyCox full-table |
| B6    | 46 080 | 0.88 s        | —    | 74 MB    | 752        | —         | matches PyCox full-table |
| E6    | 51 840 | 0.62 s        | —    | 75 MB    | 652        | 21        | matches PyCox full-table |
| **E7** | **2 903 040** | **61.1 s** | **71 s** | **6.5 GB** | **6364** | **56** | matches PyCox / Geck literature |

**E7 headline**: PyCox needed approximately 4 hours for E7 cells; rustcox
completes in 61.1 s of compute (71 s wall) — a ~235× speedup.  The output
(6364 left cells, 56 star-class representatives) matches the documented PyCox
and Geck values exactly.  Output archived on-cluster at
`results/cells_E7.json.gz` (10 MB).

**Apple-Silicon local benches** (Criterion medians, `klcells`):

| Group | \|W\| | Median time |
|-------|--------|-------------|
| H3    |    120 | 2.1 ms      |
| B4    |    384 | 4.4 ms      |
| F4  (seq)  |  1 152 | 17.1 ms |
| F4  (t=4)  |  1 152 | 14.7 ms |

## Two-sided cell counts from induction data (2026-06-11)

`tcells_from_cells` (examples/) derives two-sided cell counts from a
`rustcox cells` document via union-find (right cells = inverses of left
cells), with no KL recomputation. Verified against full-table `tcells` for
B4 (10), F4 (11), H3 (7), and against the PyCox character-table count of
special characters (a = b) for every group below — all exact:

| Group | left cells | two-sided cells | = #special characters |
|-------|-----------|-----------------|------------------------|
| H4    | 206       | 13              | 13 ✓ |
| D6    | 578       | 27              | 27 ✓ |
| B6    | 752       | 26              | 26 ✓ |
| E6    | 652       | 17              | 17 ✓ |
| **E7**| **6364**  | **35**          | **35 ✓** |

(For Weyl groups these equal the number of special nilpotent orbits via the
Springer correspondence; E8's count is 46 by the same character-table
computation, though its cell partition remains uncomputed.)

## E8 experiment outcome (job 2867149)

`rustcox cells E8` ran for the full 3 h time box at 64 threads and was
cancelled by SLURM (TIMEOUT), with MaxRSS 122.5 GB of the 250 GB allocation.
The failure mode was **time, not memory** — Bruhat/candidate sparsity keeps
the relative-KL matrices below the worst-case estimate. A multi-day run on a
large-memory node may be feasible; treat as an open experiment, not a
deliverable.
