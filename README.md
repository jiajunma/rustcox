# rustcox

A Rust rewrite of [PyCox](https://github.com/geckmf/PyCox) (M. Geck's Python
port of GAP-CHEVIE) focused on **Kazhdan–Lusztig polynomials, mu-coefficients,
left/right/two-sided cells and Duflo involutions** for finite Coxeter groups,
with equal *and* unequal parameters — designed for multi-threaded computation
on HPC machines.

**Status: planning complete, implementation in progress.**
The authoritative implementation plan lives at
[`docs/superpowers/plans/2026-06-10-rustcox-implementation.md`](docs/superpowers/plans/2026-06-10-rustcox-implementation.md).

## Layout

| Path | Content |
|---|---|
| `crates/rustcox-core` | library: Laurent polynomials, root systems, Coxeter group calculus, KL engine |
| `crates/rustcox-cli` | `rustcox` binary (`info`, `kl`, `verify`, `selftest`) |
| `pycox-ref/` | vendored PyCox reference (the oracle) + golden-data generator |
| `golden/` | canonical JSON golden files generated from PyCox — **never edit by hand** |
| `docs/` | plan, design, verification and HPC guides |

## Verification model

Every observable result (element tables, Bruhat order, every KL polynomial,
every mu slot, cells, Duflo involutions) is compared against golden JSON
produced by running the vendored PyCox under CPython:

```bash
cd pycox-ref
python3 gen_golden.py suite       # regenerate small/medium golden files
python3 gen_golden.py suite-big   # + A5, F4 (gzipped)
```

The Rust test-suite (`cargo test --workspace`) consumes these files. The
parallel KL driver is deterministic: its output is bit-identical to the
sequential reference implementation, which is pinned by tests.

## Quick numbers (PyCox oracle timings on Apple Silicon)

| W | order | distinct KL polys | left cells | PyCox time |
|---|---|---|---|---|
| B4 | 384 | 41 | 50 | 1.0 s |
| A5 | 720 | 17 | 76 | 2.5 s |
| F4 | 1152 | 313 | 72 | 28.9 s |
| H4 | 14400 | — | — | (full table: Rust target) |

## License

GPL-3.0-or-later. rustcox is a derived work of PyCox (GPL-3),
© 2011–2014 Meinolf Geck; see `pycox-ref/README.md` for provenance.
