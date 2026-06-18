# `relklpols` Memory Compression for E8 Cells (Atlas-style sparse extremal mat)

> **Tracked in:** GitHub issue #12 (umbrella) + #13 (sparse mat) + #14 (packed pool).
> **Status:** plan; implementation in progress on `main`.
> **Goal:** make `rustcox cells E8` finish on a single 256 GB `cpu` node, removing
> the current dependency on the 2 TB `fat` partition.

## Background — the OOM and where memory actually goes

The 2026-06-11→12 `cells E8` run on a 64-core 250 GB node was OOM-killed at:

```
Job 2867698  wall=17h30m  Max RSS=235.5 GB / 250 GB cap
killer = single outer rep i=36 with nc=378 induced cells
inner relklpols at layer y=239,  rklpols_len=377,170, mues_len=61
inner block-log (Q4) = 30.6 GB on disk  ← small relative to RSS
```

Source map and memory accounting (cited line ranges in
[`crates/rustcox-core/src/kl/relkl.rs`](../../../crates/rustcox-core/src/kl/relkl.rs)):

- **`mat`** (`Mat = HashMap<(Cx,Cx), Vec<Vec<SlotState>>>`, lines 460–500) — the
  outer (y, x) block grid; each block is a dense `nc × nc` grid of `SlotState`.
  `SlotState` is at least 12 bytes (enum tag + `Done { rk: u32, mu: u32 }`).
  For `nc=378` each block costs `~378² × 12 ≈ 1.7 MB`; with `nx²/2` triangular
  blocks and 240 layers populated, this is the dominant term — confirmed by
  comparing the 30 GB on-disk delta log to the 235 GB live RSS.

- **`rklpols`** (`Vec<Laurent>`, line 516) — deduplicated pool of relative-KL
  Laurents shared across blocks; reached 377,170 entries on rep 36. Each
  `Laurent = { v_exp: i32, coeffs: Vec<i64> }` is ~24 B header + 16 B per
  monomial; pool footprint at OOM ≈ several GB. **Secondary term, not the
  killer.**

- **`mues`** — μ-coefficient pool, 61 entries on rep 36, negligible.

The orthogonal performance roadmap (issue #11 + #1–#10) targets the **inner**
`klpolynomials` call on W₁; this plan targets the **outer** `relklpols`
wavefront on W. The two roadmaps stack.

## Atlas of Lie Groups stand-alone analysis (cited evidence)

`sources/stand-alone/{KLread,coef-merge,matrix-merge}.cpp` describes the
on-disk K-L format used by the Atlas project. Three structural choices, with
file:line citations:

1. **Sparse rows over primitive (== extremal) elements only** —
   `KLread.cpp:162–224`. Each row of the K-L matrix is stored as
   `(bitmap of primitives in row, Vec<poly_id>)`. Non-primitive entries are
   reconstructed at lookup via `primitivise()` — the identity
   `P_{y,w} = P_{sy,w}` for `s ∈ L(w)` with `sy > y` (exactly our
   [`extremal_stats.rs`](../../../crates/rustcox-core/examples/extremal_stats.rs)
   reduction). The primitive table is keyed by descent bitset and cached
   (`KLread.cpp:178–188`), so only `2^rank ≤ 256` tables exist even for E8.

2. **Variable-width packed polynomial pool** — `KLread.cpp:137–159` +
   `coef-merge.cpp:336–337`. A pool is `5 B × N_poly` cumulative-offset index +
   a tightly packed coefficient stream of width chosen from
   `LCM(moduli)`. For char-0 E-type KL, coefficient size is typically 1 byte,
   so a degree-`d` polynomial costs `5 + d` bytes vs our ~`24 + 16·d`.

3. **`coef-merge.cpp:247`** hard-codes `KL poly degree ≤ 31`. We can adopt the
   same assertion for `rklpols` (the Q4 layer log is already deterministic
   about bounds — anything exceeding the cap is a contract bug, not a graceful
   fallback case).

The Atlas stand-alone directory is a *reader* of dumped K-L data — the K-L
computation engine itself is in `sources/kl/` (not analysed here). What is
imported from this directory is the **storage philosophy**, not algorithms.

## Plan

### Phase 0 — Math validation (already done, here for completeness)

`extremal_stats.rs` already proves on H4, D6, E6 that
`distinct_pols == distinct_pols_extremal`. The B3 case explicitly verifies the
push-up reduction identity `P_{y,w} = P_{push_to_extremal(y, w), w}` for
every comparable pair.

For `relklpols` we have to redo the validation against the **relative** K-L
recursion (not the global one), since the storage we're compressing is
`mat[(y,x)][v][u]` not `KlTable.pol[y, w]`. The plan:

1. Add an unimplemented analogue `is_relkl_extremal((y,x,v,u), top)` whose
   correctness follows from Geck–Pfeiffer §6.4 (the same identity holds in the
   relative setting; cf. PyCox `relklpols` source comments). Cite the precise
   line in the PyCox reference.
2. Reuse the `extremal_stats.rs` testing pattern: for every group A3..F4 and
   for every cell representative, the multiset of `mat` slot values restricted
   to extremal pairs equals the full multiset. If any group fails, abort the
   refactor — the math is wrong.

### Phase 1 — Promote extremal helpers from `examples/` into the library

`examples/extremal_stats.rs` currently owns `is_extremal`, `push_to_extremal`,
`desc_mask`, and `PushCtx`. Move these into
`crates/rustcox-core/src/kl/extremal.rs` (new module), gated by no feature
flag — they're general-purpose. The `extremal_stats` binary becomes a thin
caller.

API:

```rust
pub fn is_extremal(y: ElmIdx, w: ElmIdx, ldesc: &[u64], rdesc: &[u64]) -> bool;
pub fn push_to_extremal(y: ElmIdx, w: ElmIdx, ctx: &PushCtx) -> ElmIdx;
pub fn primitive_set(w: ElmIdx, ctx: &PushCtx) -> Bitmap;  // new
```

`primitive_set` enumerates the extremal `y ≤ w` deterministically (length-asc,
lex-tiebreak) — that is the canonical iteration order of a sparse row.

### Phase 2 — Sparse `mat` block (the big win)

Replace `Vec<Vec<SlotState>>` per block with:

```rust
/// One row of a block: extremal positions only.
struct SparseRow {
    /// nc-bit bitmap, set iff (v, u) is extremal for the top of this block.
    extremal_mask: BitVec,
    /// Done states for extremal u, in u-ascending order.
    /// Length == popcount(extremal_mask).
    states: Vec<DoneState>,
}

struct DoneState { rk: RkId, mu: MuId }  // 8 B vs current 12 B SlotState

type Block = Vec<SparseRow>;  // length nc, one row per v
type Mat = HashMap<(Cx, Cx), Block>;
```

The bitmap encodes Atlas's "primitive subset" pre-translated through
`primitivise`. `Absent` slots and `Pending` slots **vanish from storage**:

- `Absent` = bit not set, no `DoneState` allocated.
- `Pending` (the layer's frontier) is reconstructed on the fly during the
  wavefront — Phase 1 produces inline `h` values without needing pre-marked
  pending slots in `mat`. (This was a redundant pre-mark in the old code; the
  `lw[x]+lw1[u] < lw[y]+lw1[v]` condition can be checked at iteration time.)
- `Done` for non-extremal `(v, u)` = bit not set, the value is the same as the
  extremal closure's, looked up by re-running `push_to_extremal` (cheap).

#### Determinism contract

`KlTable` output bytes must be unchanged. The contract is:

- `rklpols` pool grows in the **same order** as today (Phase 2 sequential
  intern walks `x desc, v, u` — `relkl.rs:638–693`). Order is preserved iff
  Phase 2 walks extremal `(v, u)` in the same lex order. **Key invariant:**
  the lex order of extremal `(v, u)` is a sub-order of the dense lex order, so
  filtering out non-extremal slots preserves relative order. Validated by a
  byte-identical golden test on A3..F4.
- The on-disk layer log format (`relkl_ckpt::LayerRecord`) **changes**, so a
  schema bump `BLK_VERSION 1 → 2` is required. Old logs become invalid and
  callers must wipe `e8_ckpt/relkl/`. CLI prints a clear message when it sees
  a stale log.

#### Memory model

For `nc=378`:

- Old: 378² × 12 B = 1.7 MB/block.
- New: 378-bit bitmap (48 B) + `(extremal-density × 378² × 8) B` per row,
  summed over 378 rows.
- E-type relative-KL extremal density empirically <10% on small groups; for E8
  conservative 20% gives 378² × 0.2 × 8 = 230 KB/block — **~7× shrink**.
- Combined with `Absent`-elimination (most blocks are >50% absent at any
  layer), total live `mat` shrinks **15–30×**. Rep 36's 235 GB drops to 8–16 GB.
- Even pessimistic 50% density and 5× overhead: 56× margin in 256 GB.

### Phase 3 — Variable-width packed `rklpols` pool

Independent of Phase 2. Replace `Vec<Laurent>` with:

```rust
struct LaurentPool {
    /// Cumulative byte offset of each Laurent's coefficient stream.
    /// 5 bytes per entry (40-bit), supports 2^40 ≈ 1 TB pool.
    offsets: Vec<u8>,
    /// Packed (v_exp_low_to_high, coef_stream).
    /// coef_size is chosen once per pool from observed coef range.
    bytes: Vec<u8>,
    coef_size: u8,    // 1, 2, 4, or 8 bytes per coefficient
    degree_cap: u32,  // matches coef-merge.cpp:247; default 31
}
```

`Laurent`'s public API is unchanged. Only the pool storage changes. The
intern lookup uses a `HashTable<u32>` keyed by hash-of-coefficients, exactly
parallel to issue #10's plan for the inner KL pool.

Expected shrink: pool drops from ~10 GB (rep 36 peak) to ~50 MB. Secondary
benefit — but combined with Phase 2 it pushes the total budget for E8 well
under 50 GB.

### Phase 4 — Rollout

```text
H4   → small enough to verify locally on a login node
D6   → run from HPC cpu partition, confirm peak RSS drops vs baseline
E6   → check  determinism byte-identically against baseline
E7   → re-run cells E7 (baseline was 10 MB cells_E7.json.gz on HPC); verify
       it still produces the same 6364 cells, 56 star-reps.
E8   → wipe e8_ckpt/relkl/ (incompat log format), resubmit cells_e8_long.sbatch
       on cpu partition with --mem=240G; finish the remaining 8 outer reps.
```

A `--mem-profile` flag is added to `rustcox cells`, emitting per-rep peak RSS
to stderr, so we have data for the next iteration.

## Out of scope

- Multi-node parallelism (would require MPI; the unit of work is shared-memory).
- Inner-`klpolynomials` rewriting (covered by #1–#10).
- Reading/writing the Atlas binary format directly (we keep our gz/JSON for
  cells output; only the in-RAM `mat` is sparsified).
- Cross-platform endianness in the pool format (Lustre + HPC are all
  little-endian; we assert at pool-open).

## Verification gates

- [ ] **G1.** `is_relkl_extremal` definition committed with PyCox reference
  line citation. Math validated on A3..F4 against a brute-force oracle.
- [ ] **G2.** All existing golden tests (full-table + cells) byte-identical.
- [ ] **G3.** `parallel_eq.rs` passes — Phase 1/2 still deterministic.
- [ ] **G4.** New per-rep peak-RSS instrumentation in `cells` CLI.
- [ ] **G5.** D6/E6/E7 HPC runs reproduce prior golden cells output.
- [ ] **G6.** E8 finishes on `cpu` partition, 256 GB cap, in <4 days wall.

## Task list (mirrors GitHub issues)

| # | Issue | Phase | Status |
|---|---|---|---|
| 1 | promote extremal helpers → library | 1 | pending |
| 2 | `is_relkl_extremal` + brute-force oracle on A3..F4 | 2 | pending |
| 3 | sparse `mat` block + `Mat` type swap | 2 | pending |
| 4 | wavefront Phase 1/2 rewrite over sparse rows | 2 | pending |
| 5 | `BLK_VERSION` bump + layer log format | 2 | pending |
| 6 | `LaurentPool` packed storage (gated behind feature flag) | 3 | pending |
| 7 | `--mem-profile` instrumentation | 4 | pending |
| 8 | HPC validation runs D6 → E6 → E7 → E8 | 4 | pending |
