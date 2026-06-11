# PyCox `klcells` driver + star operations — implementation notes

Normative source: `pycox-ref/pycox_ref.py`. THE PYTHON SOURCE WINS on discrepancy.
Companion: `2026-06-11-pycox-relklpols-notes.md`.

## Star operations

`klstaroperation(W, s, t, pcell)` (11359–11397) — RIGHT star op:
- Precondition: m_st = 3 (caller filters `coxetermat[s][t]==3`); all of pcell
  (full perms) share one right descent set.
- Gate (on pcell[0] only): `pw1 = perminverse(p)`; UNDEFINED (return False) iff
  s,t both in or both out of the right descent set (`pw1[s] ≥ N` tests).
- Map per element: `ws = right-mult by s` (= `tuple(permgens[s][r] for r in pw)`);
  if exactly one of {s,t} is a right descent of ws → take ws; else take wt.
- Returns the mapped list (full perms).

`klstarorbitperm(W, l, gens='each')` (11439): BFS orbit of a CELL under all
(s,t) pairs with m=3 (both orders); dedup by first-element coxelm against all
known orbit members. Returns list of cells (perm lists). THIS is what klcells uses.

`leftklstar` variants (11482–11578): left-handed mirror (tests `pw[s] ≥ N`,
left-multiplies). `generalisedtau(W, pw, maxd)` (11670–11689): BFS orbit of ONE
element under right star ops capped at maxd members; returns the tuple of right
descent sets of orbit members — a left-cell invariant used for pre-partitioning.

## klcells (12054–12303), equal parameters

Dispatch: weights ≠ all-1 → klcellsun (out of scope here). Rank 0 → one trivial
cell + trivial wgraph.

Recursive structure:
```
J = all generators minus ONE:
    if cartantype is E with 7 nodes → remove generator 0   (E7 → E6)
    else                            → remove the LAST generator
W1 = reflectionsubgroup(W, J)
X1p = words of redleftcosetreps(W, J)
kk = klcells(W1, ..., allcells=False)      # recursion; kk[1] = W1 star-class rep wgraphs
celms = {}                                  # set of coxelms OF INVOLUTIONS only
nc, cr1, creps = [], [], []; tot = 0
for i over kk[1] while tot < W.order:
  pairs = [wordtoperm(x ++ [J[s] for s in w]) for x in X1p, w in kk[1][i].X]
  skip if ALL pa in pairs: (pa not an involution) OR (coxelm(pa) ∈ celms)
  rk = relklpols(W, W1, kk[1][i].wgraphtoklmat(), 1, v)
  # decompose with size tiers:
  ≤300:        ind = wgraph(W,…,rk,v).decompose()
  301–1500:    pre-partition rk['perm'] by RIGHT descent set, restrict the
               wgraph (X/Xrep/Isets/mmat) per bucket, decompose each, concat
  >1500:       same but key = generalisedtau(p, maxd=3·rank)
  # both invariants are constant on left cells → buckets never split a cell
  for ii in ind:
    if tot < order and no Xrep of ii in celms:
      record rep (creps/cr1); for each orbit cell o in klstarorbitperm(W, ii.X):
        nc.append(words of o)            # allcells=False: only x with x⁻¹ ∈ o
        celms ∪= {coxelm(e) : e ∈ o, e·e == identity}; tot += |o|
    if tot < order:
      ii0 = klcellw0(W, ii)              # right-multiply cell by w0
      if ii0 new: same expansion for ii0
final check (PyCox): len(nc) == Σ degrees of SPECIAL characters (a == b) via
chartable — replace in Rust with: Σ|cell| == |W|, all elements distinct, plus
known cell counts (golden / full-table cross-check).
return [nc, cr1] with cr1 sorted by |X|
```

## Element encodings per stage

words (X1p, cell X, nc output) / full perms (pairs, rk['perm'], star orbits) /
coxelms = perm[:rank] (Xrep, celms keys). celms holds ONLY involutions
(involution test: `permmult(e,e)[:rank] == identity coxelm`).

## Scale facts (from PyCox docs/comments)

- E7 klcells: ~4 h in PyCox; 6364 left cells, 56 star-class reps.
- B8 (|W| = 10,321,920): 58 h, 9 GB in PyCox.
- E8: "seems to remain out of reach" for klcells (PyCox ships precomputed
  E8KLCELLREPS / libdistinv instead; Geck's distinguished-involutions run took
  18 days / 22 GB). Treat E8 as experimental-only.
- Memory: never materialize all |W| full perms. Persistent state = nc (words,
  |W| total) + celms (involution coxelms). Transients per induction step:
  one induced set (|X1|·|C| perms).
- Coxelm entries are ROOT indices < 2N (E7: 126 < 252) — pack as u16/u32, not u8.

## Rust port checklist

- Verbatim gates: star applicability XOR test; the skip-test; tiers >300/>1500
  with maxd = 3·rank; the E7 J rule; allcells=False inverse-closure filter.
- wgraph reuse in orbits: star-equivalent cells SHARE Isets/mmat/mpols — only
  X/Xrep are relabeled (then normalise = sort by length).
- klcellw0: transpose mmat keys, recompute Isets, reuse mpols; no-op if w0-stable.
- Parallelism: per-W1-rep relklpols calls are independent (but the celms skip
  test makes serial iteration cheaper — parallelize INSIDE relklpols instead:
  bruhatX table + the wavefront over y).
