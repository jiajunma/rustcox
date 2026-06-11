# PyCox `relklpols` — implementation notes (extraction from source)

Normative source: `pycox-ref/pycox_ref.py` lines 10483–10773 (incl. `relmue`).
These notes are a verified extraction; on any discrepancy THE PYTHON SOURCE WINS.
Equal-parameter case only (weights all 1).

## I/O contract

`relklpols(W, W1, cell1, weightL, q)`:
- `W1 = reflectionsubgroup(W, J)`, parabolic, `J = W1.fusions[W.cartanname]['subJ']`
  (list of W-generator indices; `subJ[i]` = W-index of W1's i-th generator).
- `cell1` = dict from `wgraph.wgraphtoklmat()`:
  - `'elms'`: reduced words in **W1's own labels** `0..|J|-1`, increasing length;
  - `'klmat'`: lower-triangular strings; `'f'` or `'c0' + per-W1-generator 'c<muidx>'`
    (the `'0'` is a placeholder klpol index, never used);
  - `'mpols'`: per-W1-generator mu pools, each starting `[0, 1]`.
- Returns dict:
  - `'elms'` (`ap`): induced set X·C as reduced words in W, sorted by length;
  - `'perm'` (`ap1`): same as full perms;
  - `'elmsX'`: X1 coset-rep words;
  - `'rklpols'`: ONE global pool of relative KL polys, starts `[0, 1]`;
  - `'mpols'` (`mues`): ONE global mu pool, starts `[0, 1]`  ← NOTE: unlike
    klpolynomials, NOT per-generator;
  - `'relklmat'`: dict keyed `(y,x)` coset-rep pairs (x ≤ y), value = |C|×|C|
    string grid; slot = `'f'` or `'c<rkidx>c<muidx>'`;
  - `'klmat'` (`nmat`): the flat lower-triangular matrix in `ap` order
    (consumed by the wgraph dict-path constructor);
  - `'bijection'`: `(y,v) → flat index in ap`, via
    `ap1.index(permmult(X1[y], elms1[v]))`.

## Setup

```
X1w = words of redleftcosetreps(W, J)        # min length left-coset reps, by length
X1  = perms of X1w
Lw[x]  = ℓ_W(X1w[x])
lft[s][x]: sx = permgens[s]∘X1[x] (left mult).
  If sx ∈ X1 → its index (≥ 0). Else sx = X1[x]·t for a unique W1-generator t
  (right mult leaving X); encode as  -t-1   (t in W1-LOCAL index space).
Lw1[u] = Σ poids[J[s]] over cell1['elms'][u]
p1[u]  = W1.wordtoperm(cell1['elms'][u])     # perms in W1's own root system
lft1[J[t]][u]: w1 = W1-left-mult of p1[u] by t.
  If w1 ∈ p1 → its index. Elif t·u < u (p1[u][t] ≥ W1.N) → -1 (descends out).
  Else → len(p1) (ascends out).                     # keyed by J[t], W-index!
bruhatX[y][x] = Bruhat(X1[x] ≤ X1[y])  for x ≤ y    # among coset reps
```

## Matrix init

Pools `rklpols=[0,1]`, `mues=[0,1]`. For `y`, `x<y` with `bruhatX[y][x]`:
grid of `'f'`; slot (v,u) marked `'c'` (pending) iff `x==y and u==v` is false and
`Lw[x]+Lw1[u] < Lw[y]+Lw1[v]` (plus the diagonal u==v when x==y... see source).
Diagonal block `(y,y)`: copied from cell1 — for i, j<i with cell1 klmat 'c':
slot = `'c0'` + mu index where the mu value = `cell1['mpols'][r][idx_r]` for the
FIRST generator r with a non-`''`/`'0'` slot (interned into the global `mues`);
else `'c0c0'`. Diagonal of diagonal: `'c1c0'` (poly 1, mu 0).

## Main recursion

Outer `y` increasing; inner `x` from y−1 down to 0; per slot (v,u).
`ldy = leftdescents(X1[y])`;
`fs  = [s ∈ ldy : lft[s][x] > x]` (s·x ascends, stays in X);
`fs1 = [s ∈ ldy : 0 ≤ lft[s][x] < x]`.

**Case A (fs nonempty, s = fs[0])**: for each pending (v,u):
if `bruhatX[y][sx]` and slot (sx-block) `mat[y,sx][v][u]` is 'c'-real:
copy its rkidx verbatim; mu = `relmue(Lw[y]+Lw1[v], Lw[x]+Lw1[u], rklpols[rkidx])`
(intern). Else slot = `'0c0'` (zero poly, zero mu).

**Case B (fs empty)**: per u first — vanishing test:
if ∃ s ∈ ldy with `lft[s][x] < 0` and `u < lft1[-1-lft[s][x]][u]`
(s exits X at W1-gen t and t ascends u) → all (v,u) slots = `'0c0'`; continue.
Else `s = fs1[0] if fs1 else ldy[0]`, `sx = lft[s][x]`, `sy = lft[s][y]`; per v:

```
h = 0
# subtraction term over z:
for z in range(x, sy):                     # z = x .. sy-1
  sz = lft[s][z]
  if sz < z and bruhatX[sy][z] and bruhatX[z][x]:
    for w in 0..|C|-1:
      if (sz ≥ 0 or lft1[-1-sz][w] < w) and mat[z,x][w][u] real and mat[sy,z][v][w] real:
        m = mues[ muidx of mat[sy,z][v][w] ]
        if m ≠ 0 and rkidx of mat[z,x][w][u] ≠ 0:
          h -= q^(Lw[y]+Lw1[v] − Lw[z] − Lw1[w]) · rklpols[rkidx] · m
# s·x branch:
if sx < 0:                                  # leaves X at W1-gen t = -1-sx (t·u < u here)
  t = -1 - sx
  if mat[sy,x][v][u] real, rkidx≠0:        h += (q² + 1) · rklpols[rkidx]
  if 0 ≤ lft1[t][u] < |C| and mat[sy,x][v][lft1[t][u]] real, rkidx≠0:
                                            h += rklpols[rkidx]
  for w in u+1..|C|-1:
    if lft1[t][w] > w and mat[sy,x][v][w] real and cell1['klmat'][w][u] 'c':
      m = mues[ muidx of mat[0,0][w][u] ]   # W1-cell mu from the diagonal block
      if m ≠ 0 and rkidx of mat[sy,x][v][w] ≠ 0:
        h += q^(Lw1[w] − Lw1[u] + 1) · rklpols[rkidx] · m
else:                                        # sx ≥ 0, s descends both, stays in X
  if mat[sy,sx][v][u] real, rkidx≠0:        h += rklpols[rkidx]
  if x ≤ sy and bruhatX[sy][x] and mat[sy,x][v][u] real, rkidx≠0:
                                            h += q² · rklpols[rkidx]
# store:
if h == 0: slot = '0c0'
else: intern h in rklpols → rkidx; m = relmue(Lw[y]+Lw1[v], Lw[x]+Lw1[u], h),
      intern in mues → slot 'c<rkidx>c<muidx>'
```

`relmue(lw, ly, p)`: coefficient of `v^(lw−ly−1)` in p (0 if degree mismatch;
int p counts as constant, nonzero only when lw−ly == 1).

Dependencies: each (y,x) slot reads blocks (z,x), (sy,z), (sy,x), (sy,sx) with
sy < y, z < sy — i.e. ONLY strictly-smaller first index. Hence: wavefront over y;
ALL (x, v, u) for fixed y are mutually independent (parallelizable); pools are
the only shared write state.

## Relabel

`ap` = X·C words sorted by length; `bij[y,v] = ap1.index(...)`;
`nmat[bij[y,v]][bij[x,u]] = mat[y,x][v][u]` for real slots with flat x ≤ flat y.

## wgraph dict-path constructor (9795–9883) and wgraphtoklmat (9910–9939)

Constructor from `{'elms','klmat','mpols'}`:
`Isets = left descent sets`; for y, x<y with klmat 'c':
slots `ms = split('c')[2:]`; for each W-generator s with `s ∈ I(x) \ I(y)`:
pick slot `ms[s]` if `len(ms) == rank` else `ms[0]` (single-global-pool input,
i.e. relklpols output); if non-empty/non-zero:
**`m = −(−1)^(ℓ(y)+ℓ(x)) · pool[idx]`** (SIGN FLIP by length parity), intern into
per-generator `nmues[s]`. Also generator-bijection entries for s ∉ I(y) with
s·y in the set (lines 9868–9878). `Xrep = coxelms`.

`wgraphtoklmat` is the exact inverse: `eps = −(−1)^(len X[i]+len X[j])`,
`klmat[j][i] = 'c0' + per-generator 'c<idx into rebuilt per-gen pools>'`.

## klcellw0 (11971) / wgraphstarorbit (11989)

klcellw0: right-multiply all elements by w0 (`permmult(p, w0)`); if cell is
w0-stable (np[0] ∈ pc) return unchanged; else new wgraph with recomputed Isets,
**mmat keys transposed** `(y,x)→(x,y)`, same mpols, then `.normalise()`
(sort X by length).
wgraphstarorbit: for each orbit member from `klstarorbitperm(W, wgr.X)`:
new wgraph with SAME Isets/mmat/mpols, relabeled X/Xrep, normalised.

## reflectionsubgroup parabolic (3800–3842)

`cartanJ = [[W.cartan[s][t] for t in J] for s in J]`;
`fusions[W.cartanname] = {'subJ': J, 'parabolic': True}`.
Words W1→W: `[J[s] for s in w]`.

## Rust port advice

- Replace 'c…c…' strings with `struct Slot { rk: u32, mu: u32 }` + sentinels.
- Distinct newtypes for the 5 index spaces (W-gen, W1-gen, coset x/y, cell u/v,
  flat). The `-t-1` encoding → `enum LftX { In(u32), Out(u8 /*W1-gen*/) }`.
- One GLOBAL mu pool inside relklpols; per-generator pools only in wgraph
  (with the parity sign flip) — the round-trip must preserve this exactly.
- Precompute bruhatX fully; wavefront-parallel over y inside relklpols.
