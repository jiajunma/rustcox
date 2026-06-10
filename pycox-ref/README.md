# pycox-ref — the verification oracle

`pycox_ref.py` is a vendored copy of PyCox by Meinolf Geck (GPL-3),
taken from <https://github.com/geckmf/PyCox> (`pycoxeter.codon`, which is
`chv1r6180.py` version 1r6p180, 27 Jan 2014, adapted for Codon).

Two patches were applied so it runs under plain CPython ≥ 3.9:

1. `mybytes=UInt[8]` → `mybytes=bytes` (Codon-specific type),
2. added `from functools import reduce` (Codon builtin, stdlib in CPython).

No other changes. Reference: M. Geck, *PyCox: Computing with (finite) Coxeter
groups and Iwahori–Hecke algebras*, LMS J. Comput. Math. 15 (2012),
DOI 10.1112/S1461157012001064.

## gen_golden.py

Generates canonical JSON golden files into `../golden/`. The canonicalisation
rules in its module docstring are **normative** — the Rust exporter
(`crates/rustcox-core/src/io.rs`) must match them exactly.

```bash
python3 gen_golden.py suite        # all small/medium files (~35 s)
python3 gen_golden.py suite-big    # + A5, F4 as .json.gz
python3 gen_golden.py kl H3:1 B2:2,1
python3 gen_golden.py basics E6
```
