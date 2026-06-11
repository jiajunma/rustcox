# Combinatorial invariance of Kazhdan–Lusztig polynomials: a small-group experiment

*Date: 2026-06-11. Driver: `crates/rustcox-core/examples/bruhat_invariance.rs`.
Core machinery + unit tests: `crates/rustcox-core/src/interval.rs`.*

## The conjecture

**Combinatorial invariance conjecture (Lusztig; Dyer).** For a Coxeter group
`W`, the Kazhdan–Lusztig polynomial `P_{y,w}` depends only on the isomorphism
type of the Bruhat interval `[y, w] = {z : y ≤ z ≤ w}` as a partially ordered
set. Equivalently, if `[y, w] ≅ [y', w']` as posets then `P_{y,w} = P_{y',w'}`.

This experiment tests the conjecture **empirically** on small finite Coxeter
groups by:

1. extracting every Bruhat interval of every comparable pair `y <_B w`,
2. classifying intervals up to isomorphism, and
3. checking that each isomorphism class carries a single KL polynomial.

We test **two** flavours of "interval structure":

- **poset / Hasse diagram** (the standard conjecture): the cover graph, with a
  directed edge `z1 → z2` whenever `z1 ⋖ z2` (i.e. `z1 < z2`, `l(z2) = l(z1)+1`);
- **full Bruhat graph** (a stronger / different invariant): a directed edge
  `z1 → z2` for every reflection step `z2 = t·z1` with `l(z2) > l(z1)`.

## Conventions

- **KL polynomials** are equal-parameter (`L(s) = 1` for all `s`), computed by
  the verified `kl::klpolynomials` driver (golden-checked against PyCox).
- **Reflections** are the conjugacy closure of the simple reflections, computed
  by BFS conjugation; `|reflections| = N` (number of positive roots) is asserted.
- **Bruhat-graph edges use the LEFT convention**: `z1 → z2` iff `z2·z1⁻¹` is a
  reflection and `l(z2) > l(z1)`. (The right convention `z2 = z1·t` gives an
  anti-isomorphic graph; either is fine for an isomorphism-class experiment as
  long as it is used consistently. We use left throughout.)
- **Relative levels.** Each interval vertex carries the *relative* length
  `l(z) − l(y)`. The isomorphism test maps vertices only within equal relative
  level. This makes classification **length-shift invariant**: two intervals of
  the same shape sitting in different length windows are identified, as the
  conjecture requires.
- A **cover** is exactly a length-1 Bruhat-graph edge, so the cover graph is a
  subgraph of the Bruhat graph (asserted in tests).

## Method

For each group:

1. Build the full KL table.
2. For every comparable pair `y <_B w` (`y ≠ w`), extract `I = [y, w]` from the
   table's Bruhat flags (`O(|w − y|)` per pair), and build both digraphs.
3. **Two-tier classification.** A cheap, order-invariant key per graph —
   `(|I|, level sizes, edge count, sorted multiset of Weisfeiler–Leman color
   refinements)` — buckets the pairs. Within a bucket, an exact level-respecting
   backtracking digraph-isomorphism test (WL-signature candidate pruning, with a
   loud `10^7`-node iteration cap) resolves true classes.
4. For each class, collect the set of distinct `P_{y,w}`. A class with more than
   one distinct polynomial is a **conjecture violation**.

The Weisfeiler–Leman color refinement is the load-bearing optimization: the
naive `(level, in-deg, out-deg)` signature blows past the iteration cap on the
highly symmetric `B4` intervals, whereas WL signatures keep candidate sets tiny
and let `B4` finish in ~70 s on a laptop.

### Cross-checks asserted in code (not merely reported)

- **Proven case:** any two poset-isomorphic *lower* intervals `[e, w]` must carry
  equal `P_{e,w}` (this direction of the conjecture is a theorem). A mismatch
  would be a canonicalization bug and panics.
- **Classical:** every pair with relative length gap `≤ 2` has `P_{y,w} = 1`.
- **Reflection count:** `|reflections| = N` for every group.
- The covers graph is a subgraph of the Bruhat graph.
- Vertex 0 of every interval is at relative level 0 (length-shift normalization).

These run as `#[cfg(test)]` unit tests on `interval.rs` and as inline asserts in
the driver, so a regression fails loudly rather than printing a wrong table.

## Results

| group | order | comparable pairs | distinct `P` | poset classes | Bruhat-graph classes | poset violations | graph violations |
|-------|------:|-----------------:|-------------:|--------------:|---------------------:|-----------------:|-----------------:|
| A3 | 24  | 189   | 2  | 13   | 13   | **0** | **0** |
| B3 | 48  | 799   | 4  | 71   | 71   | **0** | **0** |
| A4 | 120 | 3661  | 5  | 90   | 90   | **0** | **0** |
| D4 | 192 | 9625  | 10 | 200  | 200  | **0** | **0** |
| H3 | 120 | 5371  | 22 | 886  | 886  | **0** | **0** |
| B4 | 384 | 39865 | 41 | 3974 | 3974 | **0** | **0** |

The comparable-pair and distinct-polynomial counts were independently
cross-checked against the PyCox oracle (`klpolynomials` + `klmat`) and agree
exactly for all six groups.

### Conclusions

1. **No violations.** Across all six groups and ~59,500 comparable pairs, every
   poset-isomorphism class carries a single KL polynomial. The conjecture holds
   empirically everywhere we looked, as expected (no counterexample is known in
   any group). The Bruhat-graph version holds too.

2. **Poset and Bruhat-graph classifications coincide** (`poset classes =
   Bruhat-graph classes`) in every group tested. This is a noteworthy empirical
   observation: for these small groups the two interval invariants induce the
   *same* partition of comparable pairs. (It does not have to hold in general,
   and we make no claim beyond the groups listed.)

3. **Classes vastly outnumber polynomials.** The number of isomorphism classes
   is one to two orders of magnitude larger than the number of distinct KL
   polynomials (e.g. B4: 3974 classes vs 41 polynomials). The conjecture says
   the class *determines* the polynomial — a many-to-one map — so this ratio is
   exactly what we expect.

### Storage verdict

The conjecture suggests a tempting compression: store `pair → class-index` plus
a small `class → polynomial` table instead of `pair → pol-index`. **This saves
nothing.** Because the number of isomorphism classes is always `≥` the number of
distinct polynomials (indeed far larger here), a class index is no cheaper than a
polynomial index, and the extra `class → pol` table only adds overhead:

| group | pair→pol-index (B) | pair→class-index + class table (B) | verdict |
|-------|-------------------:|-----------------------------------:|---------|
| A3 | 756    | 808    | class-indexing larger |
| B3 | 3196   | 3480   | class-indexing larger |
| A4 | 14644  | 15004  | class-indexing larger |
| D4 | 38500  | 39300  | class-indexing larger |
| H3 | 21484  | 25028  | class-indexing larger |
| B4 | 159460 | 175356 | class-indexing larger |

The only scheme that beats `pair → pol-index` on storage is **storing nothing
per pair and recomputing the interval (and thus the polynomial, via the
conjecture or via direct KL recursion) on demand** — a pure storage-for-CPU
trade with `0` bytes/pair. For a static table the direct `pair → pol-index`
scheme remains the right default; combinatorial invariance is mathematically
deep but not a compression win.

## Full per-group output

The following is the verbatim driver output for each group (`cargo run --release
--example bruhat_invariance -- <GROUP>`).

## A3 (order 24)

- positive roots / reflections: 6 / 6
- comparable pairs (y < w): 189
- distinct KL polynomials: 2
- poset (Hasse) iso-classes: 13
- Bruhat-graph iso-classes: 13
- poset classes with >1 distinct P (VIOLATIONS): 0
- Bruhat-graph classes with >1 distinct P (VIOLATIONS): 0
- cross-check: 121 pairs with relative gap ≤ 2 all had P = 1

_Poset (Hasse) violations: none._

_Bruhat-graph violations: none._

### Storage analysis

| scheme | bytes/pair | total stored | note |
|--------|-----------|--------------|------|
| pair → pol-index | 4 | 756 | current scheme |
| pair → class-index + class→pol | 4 (+4·#classes) | 808 | classes (13) ≥ pols (2) ⇒ no win |
| nothing per pair, recompute | 0 | 0 | trades storage for CPU |


## B3 (order 48)

- positive roots / reflections: 9 / 9
- comparable pairs (y < w): 799
- distinct KL polynomials: 4
- poset (Hasse) iso-classes: 71
- Bruhat-graph iso-classes: 71
- poset classes with >1 distinct P (VIOLATIONS): 0
- Bruhat-graph classes with >1 distinct P (VIOLATIONS): 0
- cross-check: 330 pairs with relative gap ≤ 2 all had P = 1

_Poset (Hasse) violations: none._

_Bruhat-graph violations: none._

### Storage analysis

| scheme | bytes/pair | total stored | note |
|--------|-----------|--------------|------|
| pair → pol-index | 4 | 3196 | current scheme |
| pair → class-index + class→pol | 4 (+4·#classes) | 3480 | classes (71) ≥ pols (4) ⇒ no win |
| nothing per pair, recompute | 0 | 0 | trades storage for CPU |


## A4 (order 120)

- positive roots / reflections: 10 / 10
- comparable pairs (y < w): 3661
- distinct KL polynomials: 5
- poset (Hasse) iso-classes: 90
- Bruhat-graph iso-classes: 90
- poset classes with >1 distinct P (VIOLATIONS): 0
- Bruhat-graph classes with >1 distinct P (VIOLATIONS): 0
- cross-check: 1222 pairs with relative gap ≤ 2 all had P = 1

_Poset (Hasse) violations: none._

_Bruhat-graph violations: none._

### Storage analysis

| scheme | bytes/pair | total stored | note |
|--------|-----------|--------------|------|
| pair → pol-index | 4 | 14644 | current scheme |
| pair → class-index + class→pol | 4 (+4·#classes) | 15004 | classes (90) ≥ pols (5) ⇒ no win |
| nothing per pair, recompute | 0 | 0 | trades storage for CPU |


## D4 (order 192)

- positive roots / reflections: 12 / 12
- comparable pairs (y < w): 9625
- distinct KL polynomials: 10
- poset (Hasse) iso-classes: 200
- Bruhat-graph iso-classes: 200
- poset classes with >1 distinct P (VIOLATIONS): 0
- Bruhat-graph classes with >1 distinct P (VIOLATIONS): 0
- cross-check: 2352 pairs with relative gap ≤ 2 all had P = 1

_Poset (Hasse) violations: none._

_Bruhat-graph violations: none._

### Storage analysis

| scheme | bytes/pair | total stored | note |
|--------|-----------|--------------|------|
| pair → pol-index | 4 | 38500 | current scheme |
| pair → class-index + class→pol | 4 (+4·#classes) | 39300 | classes (200) ≥ pols (10) ⇒ no win |
| nothing per pair, recompute | 0 | 0 | trades storage for CPU |


## H3 (order 120)

- positive roots / reflections: 15 / 15
- comparable pairs (y < w): 5371
- distinct KL polynomials: 22
- poset (Hasse) iso-classes: 886
- Bruhat-graph iso-classes: 886
- poset classes with >1 distinct P (VIOLATIONS): 0
- Bruhat-graph classes with >1 distinct P (VIOLATIONS): 0
- cross-check: 1106 pairs with relative gap ≤ 2 all had P = 1

_Poset (Hasse) violations: none._

_Bruhat-graph violations: none._

### Storage analysis

| scheme | bytes/pair | total stored | note |
|--------|-----------|--------------|------|
| pair → pol-index | 4 | 21484 | current scheme |
| pair → class-index + class→pol | 4 (+4·#classes) | 25028 | classes (886) ≥ pols (22) ⇒ no win |
| nothing per pair, recompute | 0 | 0 | trades storage for CPU |


## B4 (order 384)

- positive roots / reflections: 16 / 16
- comparable pairs (y < w): 39865
- distinct KL polynomials: 41
- poset (Hasse) iso-classes: 3974
- Bruhat-graph iso-classes: 3974
- poset classes with >1 distinct P (VIOLATIONS): 0
- Bruhat-graph classes with >1 distinct P (VIOLATIONS): 0
- cross-check: 5616 pairs with relative gap ≤ 2 all had P = 1

_Poset (Hasse) violations: none._

_Bruhat-graph violations: none._

### Storage analysis

| scheme | bytes/pair | total stored | note |
|--------|-----------|--------------|------|
| pair → pol-index | 4 | 159460 | current scheme |
| pair → class-index + class→pol | 4 (+4·#classes) | 175356 | classes (3974) ≥ pols (41) ⇒ no win |
| nothing per pair, recompute | 0 | 0 | trades storage for CPU |


## Limitations

- **Scope.** Only A3, B3, A4, D4, H3, B4 were run (plus A2, B2 in tests). This is
  deliberate: the experiment is a laptop-scale sanity probe, not a search for
  counterexamples. Larger groups (F4 and up, E-series, H4) were **not** attempted;
  **E8 was explicitly not attempted**.
- **Convention dependence.** The Bruhat-graph result uses the left-reflection
  convention. The right convention yields anti-isomorphic graphs; since our
  isomorphism test is direction-sensitive, the *graph-class counts* could in
  principle differ under the other convention (the poset/Hasse result is
  convention-free and is the standard form of the conjecture).
- **Equal parameters only.** All KL polynomials are equal-parameter. The
  unequal-parameter setting is out of scope for this experiment.
- **Isomorphism test is exact but capped.** The backtracking iso test is sound
  and complete, but guarded by a `10^7`-node cap that panics on pathological
  blow-up. WL refinement kept every interval in these groups well under the cap;
  a larger group could in principle trip it, which is the intended loud signal.
- **No claim of generality.** "No violations" here is empirical evidence for the
  named groups only. The conjecture remains open in general (though now proven in
  important cases in the literature).
