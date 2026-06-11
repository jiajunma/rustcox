# rustcox — As-Built Architecture

## Module map

### Phase 1 — KL engine

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

### Phase 2 — Cells by parabolic induction

| Module | Responsibility |
|--------|---------------|
| `parabolic` | Parabolic subgroups `W_J`; minimal-length left coset reps (`red_left_coset_reps`); `Parabolic` struct |
| `star` | Star operations, star orbits (`star_orbit_right`); `generalised_tau` pre-partition |
| `cellgraph` | `CellGraph` — W-graph over an induced cell set; `from_relkl`; `decompose`; `to_relkl` |
| `kl/relkl` | `relklpols` — relative KL polynomials by parabolic induction (equal parameters; port of PyCox 10496–10773) |
| `kl/relkl_recur` | Inner recursion helpers: `classify_block`, `compute_caseb_block`, `intern`, `relmue`; `Lft` enum; the five index-space types |
| `kl/klcells` | `klcells` driver — full left-cell partition by induction; size-tier pre-partition; `celms` skip-set |

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

## Phase 2 — Cells by parabolic induction

### Induction pipeline

```
W1 left cells   →  wgraphtoklmat   →  relklpols    →  CellGraph
(klcells_raw)      (to_relkl)         (relkl.rs)      (from_relkl)
                                            |
                                            v
                                    decompose (size tiers)
                                            |
                                            v
                                  star-orbit + w0 expansion
                                            |
                                            v
                                   celms skip-set test
                                   (next star-class rep)
```

1. **`klcells_raw`** recursively computes the left-cell partition of `W1 = W_J`
   (the rank-1 parabolic) using the full KL table.
2. **`to_relkl`** converts each left cell of `W1` to a `RelKlInput` (the
   `cell1` dict in PyCox notation).
3. **`relklpols`** (`kl/relkl.rs`) computes the relative KL polynomials for the
   induced set `X1 · C`, where `X1` is the set of minimal-length left coset
   representatives of `W1` in `W`.  By Geck's theorem, `X1 · C` is a union of
   left cells of `W`.
4. **`CellGraph::from_relkl`** builds the W-graph over the induced set; **`decompose`**
   partitions it into left cells, using a size-tier pre-partition (`generalised_tau`
   for large induced sets, right-descent sets for medium, direct decompose for small).
5. Each new left cell of `W` spawns its full **star orbit** (via
   `star_orbit_right`) together with its **w0-image**, growing the known cell
   partition of `W`.
6. The **`celms` skip-set** (CoxElms of involutions already covered) allows
   `klcells` to skip star-class representatives whose entire induced set is
   already partitioned, avoiding redundant relklpols calls.

The algorithm terminates when all elements of `W` have been assigned a cell
(verified by a coverage assertion).

### The five index spaces (`kl/relkl_recur.rs`)

The relative-KL recursion operates in five distinct index spaces, kept
disjoint by named type aliases and a disciplined naming convention:

| Space | Meaning | Alias / var |
|-------|---------|-------------|
| W-generator | simple generator of `W` | `s` (`Gen`) |
| Coset index | position in `X1` (coset reps) | `x`, `y` (`Cx`) |
| Cell index | position in `cell1.elms` (elements of `C`) | `u`, `v` (`Cu`) |
| Flat `ap` index | position in the induced set `X · C` | `u32` |
| W1 element | perm of `W1` in `W1`'s own root system | `p1[u]` |

The `Lft` enum encodes left-multiplication of a coset rep by a W-generator:
- `Lft::In(x)` — `s·X1[x]` stays in `X1` at coset index `x`
- `Lft::Out(t)` — `s·X1[x]` leaves `X1`; `t` is the W-generator index `J[t']`

### Deterministic wavefront parallelisation

The relative-KL wavefront (`relklpols`) is parallelised with the same
two-phase deterministic design as the Phase-1 KL driver:

1. **Phase 1 (parallel):** Rayon `par_iter` over coset-rep index `x` within
   each `y`-layer.  Each worker computes its Case-B block with inline `Laurent`
   values; no shared pool writes; reads only frozen lower layers.
2. **Phase 2 (sequential):** Flatten `(x, RowResult)` pairs, sort in
   `(x desc, v, u)` order, intern through the shared pool.

This guarantees that `cells` and `star_reps` output is **byte-identical** for
any thread count.

### PyCox reference

The two normative extraction notes are:
- `docs/superpowers/plans/2026-06-11-pycox-relklpols-notes.md` — `relklpols` / `relmue`
- `docs/superpowers/plans/2026-06-11-pycox-klcells-notes.md` — `klcells` driver

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
