# RustCox Phase 2: Cells by Parabolic Induction (relklpols / klcells)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Phase 1 (full KL tables, Tasks 0–18) is complete and golden-verified; this plan
> ports PyCox's scalable cell machinery so left cells of E7 can be computed
> WITHOUT the impossible |W|² table.

**Goal:** `rustcox cells <TYPE>` computing the full left-cell partition + star-class
representative W-graphs via parabolic induction, star operations and the w0-trick —
validated against PyCox golden data and Phase-1 full-table results, then run on the
XMU HPC: H4 → D6/B6/E6 → **E7** (6364 cells expected). E8 is explicitly experimental
(PyCox itself cannot do it with this algorithm).

**Architecture:** new modules `parabolic.rs`, `star.rs`, `kl/relkl.rs`, `kl/klcells.rs`;
`wgraph.rs` upgraded to full PyCox semantics (Xrep, per-generator mu pools with the
length-parity sign flip, mmat, normalise, wgraphtoklmat, klcellw0, star orbits).
All Phase-1 conventions (canonical words, perm composition, golden pipeline) carry over.

**Normative algorithm references (READ THESE, they are precise extractions):**
- `docs/superpowers/plans/2026-06-11-pycox-relklpols-notes.md`
- `docs/superpowers/plans/2026-06-11-pycox-klcells-notes.md`
- the PyCox source itself (`pycox-ref/pycox_ref.py`) — final authority.

---

## Verification strategy (multi-layered)

1. **Golden** `kind:"cells"` files generated from PyCox `klcells(W,1,v)` for
   A3, B3, A4, D4, B4, H3, F4. Canonical form: each cell → canonical words sorted
   by (len, lex); cells sorted lex; plus `ncells`, `nstarreps`, `order`.
   Reference counts (PyCox, verified 2026-06-11): A3 10/5, B3 14/10, A4 26/7,
   D4 36/12, B4 50/22, H3 22/15 (ncells/nstarreps).
2. **Internal cross-check** (strongest): for every group with a Phase-1 full table
   (A3…F4, H3), the klcells partition must EQUAL `CellData::from_table(...).lcells`.
3. **HPC cross-checks**: H4 partition vs the archived full-table run
   (`results/h4_par.json.gz` on the cluster, 206 cells); E6 652 / D6 578 / B6 752
   cell counts from the Phase-1 HPC runs.
4. **E7**: expected 6364 left cells, 56 star-class reps (PyCox documentation).
5. Structural invariants after every klcells run: Σ|cell| == |W|, elements pairwise
   distinct, every cell's elements share their generalized invariants.

## Memory budget (E7: |W| = 2,903,040, rank 7, N = 63, 2N = 126)

| Structure | Size |
|---|---|
| nc (all cells as words) | ~2.9M words × avg 32 letters ≈ 200–400 MB |
| celms (involution coxelms) | involutions only — small |
| one induced set (X1 × C) transient | |X1| ≤ 56 at top level × cell size |
| star-orbit transient (perms of 252 u32) | per-cell, freed per orbit |

E8 (|W| = 697M): nc alone ≈ 40+ GB; klcells declared out of reach by PyCox — only
attempt as a labeled experiment, never as a deliverable.

---

## Tasks

Execution rules identical to Phase 1 (TDD, fmt+clippy clean, conventional commits,
two-stage review per task, scale ≤ F4/H3 in local tests).

### Task P0: golden generator extension ✅ (done during planning — verify only)
`gen_golden.py` gains `kind:"cells"` (PyCox klcells, canonicalized) and the suite
files `cells_A3, cells_B3, cells_A4, cells_D4, cells_B4, cells_H3, cells_F4`.

### Task P1: parabolic subgroups & coset reps (`parabolic.rs`)
- `pub struct Parabolic { pub group: CoxeterGroup /*W1*/, pub sub_j: Vec<Gen> /*J: W-indices*/ }`
- `Parabolic::new(w: &CoxeterGroup, j: &[Gen])` — restricted Cartan submatrix →
  `CoxeterGroup::from_cartan`?? NO from_cartan in Phase 1 — instead: J is a set of
  simple generators of a NAMED type; the restricted Cartan matrix must be
  type-recognized. Options: (a) port `typecartanmat`/`cartantotype` (heavy), or
  (b) recognize the subdiagram type directly from the Coxeter matrix (rank ≤ 8,
  connected components + bond multiset uniquely determine the series — implement a
  small classifier over the COXETER matrix: components via edges m≥3; per component
  classify by node count + multiset of edge labels + branch structure; A/B-or-C/D/E/F/G/H/I
  — distinguishing B vs C matters only for Cartan-level data; for klcells we need a
  CoxeterGroup for W1 whose permgens/words behave correctly — build W1 from the
  COMPONENT TYPES with the index mapping. CAREFUL: B vs C have the same Coxeter
  matrix but different Cartan matrices; the sub-Cartan of a B_n parabolic IS the
  honest restriction — preserve it: classify from the CARTAN submatrix (which we
  have exactly) by matching the off-diagonal asymmetry: −2 position distinguishes B/C.
  Implement `classify_cartan_component(sub_cartan) -> (Series, Vec<local indices>)`
  for ranks ≤ 8, validated by: for every named type T and every J ⊂ generators,
  Parabolic::new succeeds and |W1| divides |W| with the right index (test over all
  single-generator removals of A4,B4,D4,F4,H4,E6 vs hand-computed orders).
- `pub fn red_left_coset_reps(w: &CoxeterGroup, j: &[Gen]) -> Vec<Word>` — port
  PyCox redleftcosetreps (≈3974–4010; BFS over coxelms by right mult, accept if no
  J-generator is a right descent — READ THE SOURCE), returned as canonical words
  sorted by length. Tests: |reps| = |W|/|W1| for the J-removals above; E6 inside E7
  not testable locally (E7 enumeration too big? E7 coset reps via BFS need only the
  reps, not all of W — the BFS explores ~|X1|·rank coxelms — CHEAP! Test E7/E6:
  |X1| = 56 — uses CoxeterGroup::from_type("E7") which only builds roots/permgens
  (fast), NOT the element table.)
- W1-word ↦ W-word mapping helper (`[J[s] for s in w]`).

### Task P2: star operations (`star.rs`)
Port klstaroperation (right), klstarorbitperm (BFS over cells, dedup by first-element
coxelm), leftklstar + leftklstarorbitelm, generalisedtau(maxd). Perms throughout;
no whole-group enumeration. Tests: A3/B3 — partition the full-table left cells into
star orbits; every orbit member must be exactly a full-table cell (use Phase-1
CellData); generalisedtau constant on each full-table cell (A4); orbit counts:
star-orbit count of cells == nstarreps golden values where available… (defer exact
nstarreps equality to P5; here assert orbit-of-cell = unions of cells).

### Task P3: full W-graph semantics (`wgraph.rs` upgrade)
Extend WGraph with: `xrep: Vec<CoxElm>`, per-generator `mpols: Vec<Vec<Laurent>>`
(pools seeded [0,1]), `mmat: HashMap<(u32,u32), Vec<u32>>` (per-generator mu indices,
sentinel for empty), `weights`. New constructors/methods:
- `from_relkl(group, weights, rk: &RelKlOutput) -> WGraph` — the dict-path semantics
  incl. the **sign flip m = −(−1)^(ℓx+ℓy)·pool[idx]** and the len(ms)==rank vs
  single-pool distinction (relkl output has ONE global pool) and the
  generator-bijection entries (notes §wgraph). 
- `to_klmat(&self) -> CellInput` (wgraphtoklmat: inverse sign flip, per-gen pools,
  'c0'-placeholder semantics → struct CellInput { elms: Vec<Word>, mpols: …, klmat: … }).
- `normalise(&self) -> WGraph` (sort by (len,word), relabel mmat keys).
- `cell_w0(&self, group) -> WGraph` (klcellw0: right-mult by w0, transpose mmat keys,
  recompute Isets, reuse mpols, no-op if stable, normalise).
- `star_orbit(&self, group) -> Vec<WGraph>` (wgraphstarorbit: same graph data,
  relabeled X/Xrep per klstarorbitperm member, normalised).
Existing `of_cell`/`decompose` keep working (all Phase-1 tests stay green); decompose
must now use the full mmat semantics (mu-arrows s ∈ I(x)\I(y) + lft/bijection arrows).
Tests: round-trip from_relkl→to_klmat on a hand-built tiny input; w0/orbit on A3
full-table cells; Phase-1 wgraph tests unchanged.

### Task P4: relative KL polynomials (`kl/relkl.rs`) — THE core
Port relklpols exactly per the notes file (newtypes for the 5 index spaces;
`enum LftX { In(u32), Out(u8) }`; Slot{rk,mu} u32 pairs; ONE global mu pool;
bruhatX precomputed; Case A / Case B with the vanishing test, z-subtraction,
sx<0 (q²+1 / +1 / q^(Lw1 shift) terms) and sx≥0 branches; relmue; final flat
relabel + bijection). Output struct `RelKlOutput { elms: Vec<Word>, perms: …,
rklpols: Vec<Laurent>, mues: Vec<Laurent>, klmat: flat slots, … }`.
Tests (CRITICAL — this is the highest-risk task):
- For A3, B3, B4, H3 with W1 = last-generator-removed parabolic and cell1 = EACH
  star-rep W-graph of W1 (obtained by running the not-yet-existing klcells on W1?
  bootstrap instead: take W1's full KlTable → CellData.lcells → WGraph::of_cell →
  to_klmat as cell1): the resulting wgraph(rk).decompose() components must each be
  exactly a left cell of the FULL-table CellData of W (set equality of element sets).
  This validates relklpols end-to-end without needing klcells.
- Edge case: cell1 = the whole of W1 (all cells) → induced = all of W → components
  == ALL left cells of W (A3 test).

### Task P5: klcells driver (`kl/klcells.rs`)
Port per the notes: recursion with the E7 J-rule, pairs build, involution skip-test
(celms: HashSet<CoxElm> of involutions), relklpols call, decompose with tiers
(>300 right-descent prepartition, >1500 generalisedtau maxd=3·rank), star-orbit +
w0 expansion, tot early-exit, allcells flag (inverse-closure filter when false).
Correctness check replacing chartable: Σ|cell| == |W| + global element distinctness
(+ optional known-count table). Public API:
`pub fn klcells(group: &CoxeterGroup, opts: &CellsOpts) -> KlCellsResult
 { cells: Vec<Vec<Word>>, star_reps: Vec<WGraph> }` (cells canonicalized like golden).
Tests: golden cells_A3…cells_F4 full match (partition + ncells + nstarreps);
internal cross-check vs CellData.lcells for A3,B3,A4,D4,B4,H3,F4 (canonical-word
sets equal); recursion exercised ≥ 3 levels (B4 → B3 → B2 → A1…).

### Task P6: CLI + parallelism + bench
- `rustcox cells <TYPE> [--threads N] [--summary] [-o FILE]` (JSON: cells as words,
  counts; summary: ncells/nstarreps/time). selftest extension for cells_* goldens.
- Parallelize inside relklpols: bruhatX table construction + the y-wavefront
  (all (x,v,u) for fixed y independent; deterministic two-phase intern like Phase 1
  — reuse the pattern). Determinism test: cells output identical seq vs threads=4
  (B4, H3).
- Bench: criterion cells_B4, cells_H3, cells_F4.

### Task P7: HPC validation ladder (operator task, not subagent-codeable)
On XMU (see private notes; repo at ~/rustcox): sbatch scripts `hpc/cells_medium.sbatch`
(H4, D6, B6, E6 — verify 206/578/752/652 and H4 partition equality vs the archived
full-table JSON) and `hpc/cells_e7.sbatch` (E7, 64 threads, --mem=200G, time 24h;
expect 6364 cells / 56 reps; record wall time + RSS). E8: NOT attempted unless E7
finishes in < 1h with < 10GB — then a time-boxed experiment with explicit kill limits.

### Task P8: docs + final review + push
DESIGN/VERIFICATION/HPC/BENCHMARKS updates; whole-phase final review; CI green.
