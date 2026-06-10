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
