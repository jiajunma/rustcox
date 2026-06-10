# rustcox — As-Built Architecture

## Module map

| Module | Responsibility |
|--------|---------------|
| `laurent` | Laurent polynomials over i64; pos/nonneg/zero/bar parts; serde |
| `ring` | `RootCoeff` trait; `GoldenInt` (ℤ[φ]); `CycInt` (ℤ[ζ_m]/Φ_m) |
| `cartan` | Named-type Cartan/Coxeter matrices; degrees; parse_type |
| `roots` | Root system BFS; (height, rev-lex) sort; permgens |
| `element` | `Word`, `CoxElm`, `Perm`; composition; length; descents |
| `group` | `CoxeterGroup`; word↔perm; longest element; type-erased after construction |
| `enumerate` | `ElementTable`; canonical order; lft/inva/aw0 maps |
| `bruhat` | Bruhat order test (iterative) |
| `kl/table` | `KlTable`; polynomial and mu pools; `KlRow`; `MuMode` |
| `kl/compute` | Row kernel; sequential driver `klpolynomials_seq` |
| `kl/compute_uneq` | Unequal-parameter mu computation |
| `kl/parallel` | Deterministic layered parallel driver `klpolynomials` |
| `kl/cells` | Arrows; Tarjan SCC cells; Duflo; lorder; rcells/tcells |
| `kl/scc` | Tarjan SCC (generic) |
| `wgraph` | Minimal W-graph per cell; decompose |
| `io` | Canonical JSON export/import (schema `rustcox-golden-v1`, gz-transparent) |

## Element representations and conventions (plan §0.3)

Three representations, always consistent:

- **`Word`** (`Vec<u8>`): reduced expression `[s₁, s₂, …, sₖ]` in generator indices.
- **`CoxElm`** (`Box<[RootIdx]>`, length = rank): images of the `rank` simple roots;
  the prefix of the full permutation.
- **`Perm`** (`Box<[RootIdx]>`, length = 2N): full permutation of all `2N` root indices.

**Composition:** `then(p, q)[i] = q[p[i]]` — "apply p first". PyCox: `permmult(p, q)`.

**Length:** `l(w) = #{i < N : perm[i] ≥ N}` (the number of positive roots sent negative).

**Descents:** `s` is a *left* descent of `w` iff `perm[simple_root[s]] ≥ N`.
Right descents are left descents of the inverse.

**Canonical word:** strip the smallest left descent repeatedly (PyCox `permtoword`).
Canonical element order = sort by `(length, canonical_word lex)`. All golden
indices use this order.

**Product groups:** for a reducible group `A2xA1` the simple roots are laid out
blocked by component. `simple_root[s]` is the global positive-root index of
generator `s`; for irreducible groups `simple_root[s] == s`. Root ordering within
a component uses that component's `n_pos`, not global height — a known deviation
from PyCox's global-height sort (see Known deviations below).

## Root ordering

Positive roots are sorted by `(Σ approx-coordinates, reverse-lex by approx)`.
For crystallographic types the coordinates are exact integers; for H-types they
are `GoldenInt` values evaluated via `approx()`; for I₂(m) types they are
`CycInt` values, also via `approx()`. PyCox uses float comparisons in the same
way (`lpolmod.__gt__`). The precision note in `ring/cyc.rs` explains why f64
is exact for the small algebraic integers that arise.

## KL recursion overview

The normative spec is `pycox-ref/pycox_ref.py` at the lines cited below.

1. **Bruhat flag** (≈10220–10260): fused into the row kernel; four cases
   (y=0 or y=w; same length; aw0 symmetry; descent-based recursion).
2. **P̃_{y,w}** (≈10262–10338): tried in priority order — identity, inverse
   symmetry, Case I, Case II, full recursion (v^{2L(s)} terms and mu sum).
3. **mu** (≈10340–10380): equal-param `MuMode::Implicit` (derived on demand
   from the polynomial via `zero_part`; no array stored); unequal-param
   `MuMode::Stored` (interned into per-generator pools).

Pool insertion order: `w` ascending, `y` descending — the parallel driver
reproduces this exactly in its sequential intern phase.

## Deterministic parallel design

Source: `kl/parallel.rs`. Designed so that `klpolynomials` and
`klpolynomials_seq` return byte-identical `KlTable`s for all inputs.

**Layer classification:** within length layer `l`, rows are partitioned into
*units*:
- *Single*: `inva[w] == w` (self-inverse) — one row.
- *Pair*: `{min, max}` where `inva[min] = max`; min < max.

**Phase 1 (parallel):** Rayon `par_iter` over units. Each unit computes its
row(s) sequentially, storing inline `Laurent` values (not pool ids). A pair
computes `min` first; `max` then reads `min`'s `RowResult` directly (the only
same-layer cross-row dependency). Earlier layers are fully frozen (`&[Laurent]`
reads, no lock).

**Phase 2 (sequential):** Flatten `(w, RowResult)` pairs, sort by `w`
ascending, intern through the shared `Interner` — identical pool growth to the
sequential driver regardless of thread count or `layer_chunk`.

**Two-phase intern:** inline values (phase 1 output) are replaced by u32 pool
ids (phase 2). This is the mechanism that makes the pool order deterministic.

**`layer_chunk`:** optional bound on units-per-Rayon-chunk; result is
independent of chunk size (only bounds peak in-flight memory per layer).

## MuMode

- `MuMode::Implicit` (equal parameters): no mu array stored. The mu value for
  pair `(y, w)` and generator `s` is computed on demand as
  `zero_part(v^{1 + L(y) - L(w)} · P̃_{y,w})`. Slot presence is stored as
  a flat bool array.
- `MuMode::Stored` (unequal parameters): mu values are interned into
  per-generator pools `mues[s]`. Full `u32` index array, sentinel `NO_MU`.

## CycInt design (`ring/cyc.rs`)

`CycInt` = an element of ℤ[ζ_m]/(Φ_m(ζ_m)) for the m-th cyclotomic field.
Used for I₂(m) Cartan entries with m ∉ {3, 4, 5, 6}.

**Sentinel order 0:** constants `from_int(n)` carry order `n = 0` (not bound to
any field). On the first binary operation, a sentinel is promoted to the partner's
field. Two sentinels stay sentinel. Within one root system, all non-constant
values share one `m`, so distinct fields never alias.

**Value-based Eq/Hash:** a constant (degree ≤ 0) compares and hashes equal
regardless of whether it carries field order 0 or a genuine field order. This
is required because root-system BFS stores `Vec<CycInt>` keys that mix
sentinel constants (from `from_int`) with field-bound values (from Cartan
entries).

**Φ_n computation** (`CycInt::phi`): port of PyCox `cyclpol`. Called only at
construction time; n is expected to be small (≤ ~60); no memoisation.

## Known deviations from PyCox

### Product-group root ordering

For a product group `A_n × B_m`, PyCox constructs a root system with roots
sorted globally by height, treating all coordinates from both components as one
vector. Rustcox sorts each component's roots independently within its own
`n_pos`, then concatenates. Simple root index `s` maps to global root index
`simple_root[s]`, not simply `s`. The Coxeter group combinatorics (descents,
length, Bruhat, KL) use `simple_root` consistently, so all results are correct
for each component. Cross-component comparison against PyCox golden files for
product types has not been verified because all committed golden files are
irreducible.

### I₂(5): GoldenInt vs CycInt equivalence

For m=5, the Cartan matrix uses `GoldenInt` (ℤ[φ]), not `CycInt`. The two
representations are numerically equivalent because φ = ζ_{10} + ζ_{10}^{-1}.
The golden files (`kl_I5_w1.json`, `basics_I5.json`) pass with GoldenInt, and
the test verifies byte-identical output to PyCox.
