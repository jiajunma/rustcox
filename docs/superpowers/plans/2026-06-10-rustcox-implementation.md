# RustCox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Rust rewrite of PyCox's Kazhdan–Lusztig polynomial machinery for finite Coxeter groups — modular, multi-threaded (HPC-ready), and verified bit-for-bit against PyCox golden data.

**Architecture:** A `rustcox-core` library crate (Laurent polynomials, root systems, Coxeter group element calculus, KL recursion, cells) plus a `rustcox` CLI binary. The KL computation is layered by element length; rows within a length layer are computed in parallel with rayon using a deterministic two-phase (compute-then-intern) scheme, so parallel output is byte-identical to sequential output. Verification is driven by canonical JSON golden files generated from the vendored PyCox reference (`pycox-ref/`).

**Tech Stack:** Rust 2021 (MSRV 1.75), rayon, serde/serde_json, flate2, clap; dev: proptest, criterion. Oracle: Python 3.9+ running vendored `pycox_ref.py` (GPL-3).

**License:** GPL-3.0-or-later (rustcox is a derived work of PyCox, which is GPL-3).

---

## Part 0 — Status, references, and ground truth

### 0.1 What already exists in this repo (done during planning — do NOT redo)

| Path | Content |
|---|---|
| `pycox-ref/pycox_ref.py` | Vendored PyCox (`pycoxeter.codon` from https://github.com/geckmf/PyCox, = `chv1r6180.py` v1r6p180), with **two patches**: line ≈79 `mybytes=UInt[8]` → `mybytes=bytes`, and added `from functools import reduce`. Runs under plain Python ≥3.9. |
| `pycox-ref/gen_golden.py` | Golden-data generator. **The canonicalisation rules in its docstring are normative** — the Rust exporter must match them exactly. |
| `golden/*.json[.gz]` | Generated golden files (see §2.3 for the list). |

The reference for every algorithm in this plan is `pycox-ref/pycox_ref.py`. Line numbers below refer to that file (≈ ±2 lines). When this plan and the PyCox source disagree, **the PyCox source wins** — the golden files are its output.

### 0.2 Background reading (optional)

- M. Geck, *PyCox: Computing with (finite) Coxeter groups and Iwahori–Hecke algebras*, LMS J. Comput. Math. (2012), DOI 10.1112/S1461157012001064 — describes exactly the algorithms being ported.
- The `klpolynomials` docstring (pycox_ref.py ≈10141–10200) documents the output format including a worked B2 unequal-parameter example.

### 0.3 Key conventions (memorize these)

- **Polynomial variable:** everything is a Laurent polynomial in `v`. Equal-parameter classical `P_{y,w}(q)` appears with `q = v²` (e.g. `1+q` is stored as `1+v²`). With unequal weights, coefficients can be **negative** and Duflo signs `n_d` can be `−1`.
- **Weight function:** `weights[s] ≥ 0` per generator; conjugate generators must get equal weights. `L(w) = Σ weights[s_i]` over a reduced word. Equal-parameter case = all weights 1 (`uneq = false`); *any* other vector (even all-2) takes the unequal-parameter code path.
- **Roots:** indices `0..N` are positive roots, `N..2N` negative, `roots[N+i] = −roots[i]`. Positive roots sorted by height (coordinate sum) ascending, then reverse-lex within a height. Simple roots are indices `0..rank`.
- **Element representations:** `Word` (reduced word, `Vec<u8>` of generator indices), `CoxElm` (images of the `rank` simple roots, length-`rank` tuple of root indices), `Perm` (full permutation of all `2N` root indices). Identity perm = `(0,1,…,2N−1)`.
- **Composition:** `permmult(p,q)[i] = q[p[i]]` (apply `p` first). `wordtoperm([s1,…,sk]) = fold(permmult, id, [P_s1,…,P_sk])`.
- **Length:** `l(w) = #{i < N : perm[i] ≥ N}`.
- **Descents:** `s` is a *left* descent of `w` iff `perm[s] ≥ N`; right descents are left descents of the inverse.
- **Canonical word:** strip the smallest left descent repeatedly (PyCox `permtoword`, ≈3180–3200). **Canonical element order = sort by (length, canonical word lex).** All golden indices use this order.

### 0.4 Ground-truth reference values (verified on 2026-06-10 against the vendored PyCox)

Equal parameters (weights all 1), `npols` = number of *distinct* KL polynomials, `ncells` = number of left cells:

| W | order | N | degrees | npols | ncells | PyCox time |
|---|---|---|---|---|---|---|
| A1 | 2 | 1 | [2] | 1 | 2 | — |
| A2 | 6 | 3 | [2,3] | 1 | 4 | — |
| A3 | 24 | 6 | [2,3,4] | 2 | 10 | — |
| A4 | 120 | 10 | [2,3,4,5] | 5 | 26 | 0.1 s |
| A5 | 720 | 15 | [2,..,6] | 17 | 76 | 2.5 s |
| B2 | 8 | 4 | [2,4] | 1 | 4 | — |
| B3 | 48 | 9 | [2,4,6] | 4 | 14 | — |
| B4 | 384 | 16 | [2,4,6,8] | 41 | 50 | 1.0 s |
| D4 | 192 | 12 | [2,4,4,6]* | 10 | 36 | 0.2 s |
| F4 | 1152 | 24 | [2,6,8,12] | 313 | 72 | 28.9 s |
| G2 | 12 | 6 | [2,6] | 1 | 4 | — |
| H3 | 120 | 15 | [2,6,10] | 22 | 22 | 0.13 s |
| I2(5) | 10 | 5 | [2,5] | 1 | 4 | — |

\* D4 degrees as PyCox reports them sorted: `[2,4,4,6]`.

More A3 facts: distinct pols `{1, 1+v²}`; 213 Bruhat-comparable pairs `(y ≤ w)`; exactly 6 pairs have `P = 1+v²`; 54 arrows; cell sizes sorted `[1,1,2,2,3,3,3,3,3,3]`; `lorder` has 39 `true` entries; Duflo a-values multiset `{0,1,1,1,2,2,3,3,3,6}`.

B3 distinct pols: `{1, 1+v², 1+v⁴, 1+v²+v⁴}`.

Unequal parameters:

| W, weights | npols | distinct pols | ncells | duflo (canonical idx, a, n) |
|---|---|---|---|---|
| B2, [2,1] | 3 | `{1, 1−v², 1+v²}` | 6 | `[[0,0,1],[1,2,1],[2,1,1],[6,2,1],[5,3,−1],[7,6,1]]` |
| B2, [1,2] | 3 | `{1, 1+v², 1−v²}` | 6 | — |
| G2, [1,3] | 4 | `{1, 1+v², 1−v², 1−v²+v⁴}` | 6 | — |

B2 [2,1] left cells (canonical indices): `[[0],[1,4],[2],[3,6],[5],[7]]` — matches the PyCox docstring example.

H3 has 22 distinct pols; the largest is `1+v²+v⁴+v⁶+v⁸`-type degree 8 entries (full list pinned in `golden/kl_H3_w1.json`).

### 0.5 Scaling expectations (sets perf targets and memory budgets)

Full-table `klpolynomials` is Θ(|W|²) memory. With u32 pol-ids and implicit mu (equal-parameter mode):

| W | \|W\| | pol-id matrix | feasibility |
|---|---|---|---|
| F4 | 1 152 | ~2.7 MB | trivial; Rust target ≤ 1 s sequential (Python: 29 s) |
| H4 | 14 400 | ~415 MB | fine on a laptop; minutes parallel |
| D6 | 23 040 | ~1.1 GB | HPC node |
| E6 | 51 840 | ~5.4 GB | fat node, experimental |
| E7/E8 | 2.9M/697M | — | **out of scope** for full tables (needs relklpols/cells induction, Part 5) |

---

## Part 1 — Architecture & design

### 1.1 Workspace layout

```text
rustcox/
├── Cargo.toml                  # [workspace] members = ["crates/rustcox-core", "crates/rustcox-cli"]
├── rust-toolchain.toml         # stable
├── LICENSE                     # GPL-3.0
├── README.md
├── CLAUDE.md                   # agent guide (conventions, commands)
├── .github/workflows/ci.yml
├── crates/
│   ├── rustcox-core/
│   │   ├── Cargo.toml          # deps: rayon, serde, serde_json, flate2, thiserror
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── laurent.rs      # Laurent polynomials over i64
│   │   │   ├── ring.rs         # RootCoeff trait, GoldenInt (ℤ[φ]); later CycInt
│   │   │   ├── cartan.rs       # named-type Cartan/Coxeter matrices, degrees
│   │   │   ├── roots.rs        # root system BFS, ordering, permgens
│   │   │   ├── element.rs      # Word/CoxElm/Perm + conversions
│   │   │   ├── group.rs        # CoxeterGroup
│   │   │   ├── bruhat.rs       # Bruhat order test
│   │   │   ├── enumerate.rs    # element table, lft/inva/aw0 maps
│   │   │   ├── kl/
│   │   │   │   ├── mod.rs      # public API, KlOpts
│   │   │   │   ├── table.rs    # KlTable / KlRow storage + accessors
│   │   │   │   ├── compute.rs  # row kernel (Bruhat init + P + mu), sequential driver
│   │   │   │   ├── parallel.rs # layered rayon driver (two-phase intern)
│   │   │   │   └── cells.rs    # arrows, SCC cells, duflo, lorder, rcells, tcells
│   │   │   ├── wgraph.rs       # minimal W-graph per cell + decompose
│   │   │   └── io.rs           # canonical JSON (golden format) import/export
│   │   ├── tests/
│   │   │   ├── golden_basics.rs
│   │   │   ├── golden_kl.rs
│   │   │   ├── parallel_eq.rs
│   │   │   └── props.rs
│   │   └── benches/kl.rs
│   └── rustcox-cli/
│       ├── Cargo.toml          # deps: rustcox-core, clap, anyhow
│       └── src/main.rs
├── pycox-ref/                  # oracle (exists)
└── golden/                     # golden JSON (exists)
```

Rules: every file ≤ ~600 lines; split before it grows past that. No `unsafe`. `cargo clippy -D warnings` clean.

### 1.2 Core types

```rust
// element.rs
pub type Gen = u8;            // generator index (rank ≤ 64 in practice)
pub type RootIdx = u32;       // index into the root list, < 2N
pub type ElmIdx = u32;        // index into the canonical element table
pub type Word = Vec<Gen>;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Perm(pub Box<[RootIdx]>);    // length 2N

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CoxElm(pub Box<[RootIdx]>);  // length rank (= first rank entries of Perm)
```

```rust
// laurent.rs — invariant: coeffs is empty (zero poly) or first and last entries are nonzero
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Laurent {
    val: i32,          // exponent of coeffs[0]
    coeffs: Vec<i64>,  // coeffs[i] = coefficient of v^(val+i)
}
```

```rust
// group.rs
pub enum Series { A, B, C, D, E, F, G, H, I(u32) }   // I(m) = dihedral I2(m)
pub struct TypeComponent { pub series: Series, pub indices: Vec<usize> }

pub struct CoxeterGroup {
    pub rank: usize,
    pub n_pos: u32,                       // N
    pub order: u128,
    pub degrees: Vec<u32>,                // sorted ascending
    pub coxmat: Vec<Vec<u32>>,            // Coxeter matrix m_st (diag 1)
    pub permgens: Vec<Perm>,              // rank generators as root permutations
    pub components: Vec<TypeComponent>,
    pub roots_int: Option<Vec<Vec<i64>>>, // 2N coordinate vectors, crystallographic only
    longest: std::sync::OnceLock<Perm>,
}
```

The group is **type-erased**: root coordinates over ℤ[φ] are used only during construction (in `roots.rs`, generic over `RootCoeff`); the stored group is purely combinatorial (`permgens` + metadata), so the entire KL layer is monomorphic.

### 1.3 PyCox → Rust API mapping

| PyCox (pycox_ref.py ≈line) | Rust | Notes |
|---|---|---|
| `lpol` class (588) | `laurent::Laurent` | zero = empty coeffs (PyCox returns int 0) |
| `pospart/nonnegpart/zeropart/barpart` (10104–10138) | `Laurent::{pos_part, nonneg_part, zero_part, bar}` | `zero_part` returns `i64` |
| `cartanmat(typ,n)` (2145) | `cartan::cartan_mat(series, rank) -> CartanMat` | data tables copied verbatim |
| `degreesdata` (2677) | `cartan::degrees(series, rank)` | |
| `roots(cmat)` (2728) | `roots::generate::<R: RootCoeff>(...) -> RootSystem<R>` | BFS + (height, rev-lex) sort |
| `coxeter(typ,n)` (2790) | `CoxeterGroup::from_type(&str)` / `from_components(&[(Series, usize)])` | |
| `permmult` (1488) | `Perm::then(&self, q) ` | `then(p,q)[i] = q[p[i]]` |
| `perminverse` (1506) | `Perm::inverse` | |
| `wordtoperm` / `permtoword` (3061–3234) | `group.word_to_perm` / `group.perm_to_word` | `perm_to_word` = canonical word |
| `wordtocoxelm`, `coxelmtoword` | `group.word_to_coxelm`, `group.coxelm_to_word` | |
| `permlength` (3089) | `group.perm_length` | |
| `leftdescentsetperm` / `rightdescentsetperm` (3263) | `group.left_descents` / `right_descents` | |
| `longestperm` (3522) | `group.longest_perm()` | cached via OnceLock |
| `bruhatperm` (3622) | `bruhat::leq(&group, x, y)` | iterative variant only |
| `allcoxelms` (3925) | `enumerate::ElementTable::build(&group)` | w0 symmetry trick kept |
| `klpolynomials(W, weightL, v)` (10141) | `kl::klpolynomials(&group, &KlOpts) -> KlTable` | parallel by default; `kl::klpolynomials_seq` reference |
| cells/duflo block inside klpolynomials (10380–10468) | `kl::cells::CellData::from_table(&KlTable)` | SCC instead of O(n³) closure |
| `klpoly1` (10470) | `KlTable::cell_tables()` | per-cell view |
| `wgraph` class (9698) | `wgraph::WGraph` | minimal: X, Isets, mues, decompose |

### 1.4 The KL recursion (normative spec)

Precomputation (`enumerate.rs`): canonical element table `elms` sorted by (length, word); arrays `lengths[i]`, `lweights[i] = L(w_i)`, `coxelms[i]`; maps `inva[i]` (index of inverse), `aw0[i]` (index of `w0·w_i`), and the left-multiplication table `lft[i][s]` (index of `s·w_i`). Key invariant: `lft[w][s] < w` ⟺ `s` is a left descent of `w` — pinned by a test.

For each pair `(w, y)`, `y ≤ w` as indices, the entry holds: Bruhat flag, pol-id of `P̃_{y,w}`, and (uneq only) per-generator mu-ids.

**Bruhat flag** (PyCox ≈10220–10260, fused into the row kernel): `y=0` or `y=w` → comparable. Same length, `y≠w` → not. If `lengths[w]+lengths[y] > N` → copy flag from `(aw0[w], aw0[y])` (longer-row symmetry; that row has strictly smaller length, see §1.5). Otherwise pick the first left descent `s` of `w`, `sw = lft[w][s]`:
- if `s` is a descent of `y`: comparable iff `sy ≤_B sw`;
- else: comparable iff `y ≤_B sw`.

**P̃_{y,w}** (PyCox ≈10262–10338): in priority order —
1. `y == w` → `1`.
2. *Inverse symmetry*: if `inva[w] < w`, or (`inva[w] == w` and `inva[y] > y`): `P̃_{y,w} = P̃_{inva[y], inva[w]}` (already computed).
3. *Case I*: first `s` with `lft[y][s] > y` and `lft[w][s] < w` (left descent of `w`, not of `y`). If `weights[s] == 0`: `P̃ = P̃_{sy,sw}` if `sy ≤_B sw` else `0`. Else `P̃_{y,w} = P̃_{sy,w}` (same row, higher index — already computed in this row's descending-y loop).
4. *Case II*: same as Case I applied to `(inva[y], inva[w])`; result read from `P̃_{inva(sy), w}` (same row, higher index).
5. *Full recursion*: pick `s` = first left descent of `w` (uneq: among left descents, one with minimal weight). `sw = lft[w][s]`, `sy = lft[y][s]`. If `weights[s] == 0`: `P̃ = P̃_{sy,sw}`. Else:

   `P̃_{y,w} = P̃_{sy,sw} + v^{2·weights[s]}·P̃_{y,sw}·[y ≤_B sw] − Σ_z mu^s_{z,sw} · v^{L(w)−L(z)} · P̃_{y,z}`

   sum over indices `z ∈ [y, sw)` with `lft[z][s] < z`, `y ≤_B z`, `z ≤_B sw`, `mu^s_{z,sw} ≠ 0`. (The index range is a superset of the Bruhat interval; the flags filter it. The result is independent of intra-length index order.)

**mu** (PyCox ≈10340–10380):
- Equal parameters: `m = zero_part(v^{1+L(y)−L(w)} · P̃_{y,w})` — the same scalar for every `s`; the slot for `s` is *present* iff `lft[y][s] < y` and `lft[w][s] > w`. **Do not store**: recompute on demand from the pol-id (`MuMode::Implicit`).
- Unequal, slot present iff `weights[s] > 0`, `lft[y][s] < y`, `lft[w][s] > w`:
  - if `lengths[y]+lengths[w] > N`: `m = ±mu^s_{aw0[y], aw0[w]}` (sign `−` iff `lengths[w]−lengths[y]` even);
  - elif `weights[s] == 1`: `m = zero_part(v^{1+L(y)−L(w)} · P̃)`;
  - else: `m = nonneg_part(v^{weights[s]+L(y)−L(w)} · P̃)`; then for `z` from `w−1` down to `y+1` with `lft[z][s] < z`, `y ≤_B z`, `z ≤_B w`: `m −= nonneg_part(pos_part(mu^s_{z,w}) · v^{L(y)−L(z)} · P̃_{y,z})`; finally if `m ≠ 0`: `m = bar(m) + m − zero_part(m)`.
  - Stored in per-generator pools (`MuMode::Stored`).

**Pools:** `pols: Vec<Laurent>` with `pols[0] = 1`; `mues[s]: Vec<Laurent>` with `mues[s][0] = 0`. Dedup via `HashMap<Laurent, u32>` beside each Vec. Pool insertion order in the sequential driver: `w` ascending, `y` descending — the parallel driver must reproduce exactly this order (see §1.5).

### 1.5 Parallel design (deterministic, layered)

Observed dependency structure (verified against the source):
- Computing row `w` reads: rows `z` with `lengths[z] < lengths[w]` (always fully complete), entries of **row `w` itself at higher `y`** (the descending-y loop order), and — only in the inverse-symmetry shortcut — row `inva[w]` (same length).
- The `aw0` symmetry lookups (`Bruhat flag` and uneq-mu) read row `aw0[y]` under a guard `lengths[y]+lengths[w] > N` ⟺ `lengths[aw0[y]] = N − lengths[y] < lengths[w]` — strictly shorter, safe.

Therefore:

```text
for l in 1..=N:                                   # length layers, sequential
    units = group {w : lengths[w]==l} into {w} if inva[w]==w else {min(w,w⁻¹), max}
    PHASE 1 (rayon par_iter over units):
        for each unit: compute its row(s) sequentially (y from w down to 0);
        the second member of a pair uses the inverse-symmetry shortcut on the first;
        new polynomials are kept INLINE (Laurent values) in the row buffer;
        reads of earlier layers go through the frozen pools (&[Laurent], no lock)
    PHASE 2 (sequential, w ascending / y descending):
        intern inline polynomials into the global pools (dedup via HashMap),
        replacing values by u32 ids  →  identical pool order to the sequential driver
```

- Within a unit, the row kernel is **identical** to the sequential one (same code path, `compute.rs`).
- Determinism: phase 2 ordering reproduces the sequential pool order exactly, so `klpolynomials` and `klpolynomials_seq` return identical `KlTable`s — pinned by `tests/parallel_eq.rs`.
- Memory: transient inline values live only for one layer; for huge groups add `KlOpts::layer_chunk` (process units of a layer in bounded chunks with an intern pass after each chunk — still deterministic if chunks are formed in unit order).
- Shared state during phase 1 is read-only (`&KlTable` of completed layers + this layer's per-unit buffers). No locks, no atomics.

### 1.6 Cells extraction (`cells.rs`)

Port of PyCox ≈10380–10468 with better asymptotics, same output:

1. `adelta[w]`, `ndelta[w]`: from `p = v^{−L(w)}·P̃_{0,w}`: if `p == 0` → `(−1, 0)`, else `(−deg p, leading coeff)`.
2. Arrows `pp`: for every `w` and `s`: if `weights[s]==0` or `lft[w][s] > w` → arrow `(w, lft[w][s])`; for every comparable pair `y < w`: arrow `(w,y)` iff ∃`s`: `weights[s]>0`, `lft[y][s]<y`, `lft[w][s]>w`, `mu^s_{y,w} ≠ 0`.
3. Left cells = SCCs of the arrow digraph (Tarjan, O(V+E)) — *not* PyCox's O(n³) closure; results provably identical (mutual reachability classes).
4. `duflo` per cell: among members with `ndelta ≠ 0`, pick `d` minimizing `adelta[d]`; record `[d, adelta[d], ndelta[d]]`. Sanity checks (same as PyCox): minimizer unique, `n_d = ±1`; expose as `CellData::checks_ok: bool`.
5. `lorder[c1][c2]` = reachability of `duflo[c2]`'s cell from `duflo[c1]`'s cell in the condensation DAG (BFS per cell over the cell-level DAG).
6. `rcells = {inva(cell)}`; `tcells` = connected components of "same left cell OR same right cell" (union–find).

### 1.7 Canonical JSON / golden format

Normative spec lives in `pycox-ref/gen_golden.py`'s docstring. Summary:

```jsonc
{
  "schema": "rustcox-golden-v1",
  "kind": "kl",                              // or "basics"
  "type": [{"series":"B","rank":2}],          // I2(m): {"series":"I","rank":2,"m":7}
  "weights": [2,1],
  "rank": 2, "order": 8, "N": 4,
  "elms": [[],[0],[1],[0,1],[1,0],...],       // canonical words, sorted (len, lex)
  "pols": [{"v":0,"c":[1]}, ...],             // dedup, sorted by (val, coeffs lex)
  "mues": [[...], ...],                       // one pool per generator, same sort
  "klmat": [[0],[0,0],[0,-1,0],...],          // klmat[w][y]: pol index, -1 = incomparable
  "mumat": [[[-1,-1]],...],                   // mumat[w][y][s]: mu index, -1 = no slot
  "arrows": [[w,y],...],                      // sorted
  "lcells": [[0],[1,4],...],                  // each sorted; list sorted lex
  "duflo":  [[d,a,n],...],                    // aligned with lcells order
  "lorder": [[1,0,...],...],                  // aligned with lcells order
  "rcells": [...], "tcells": [...]
}
```

`kind: "basics"` files carry: `order, N, rank, degrees, coxetermat`, `roots` (crystallographic only), `length_histogram` and `longest_word` (when `|W| ≤ 10000`).

Rust side: `io::to_canonical_json(&KlTable, &CellData) -> serde_json::Value` must reproduce these bytes (after JSON key-sorted, compact serialization). Comparison in tests is on parsed `serde_json::Value`s, not raw bytes.

### 1.8 Error handling & input validation

- `CoxeterGroup::from_type` returns `Err` for unknown series/rank out of range (B needs n≥2, D n≥4, E∈{6,7,8}, F=4, G=2, H∈{3,4}, I m≥3).
- `KlOpts` validation: `weights.len() == rank`; weights equal on conjugate generators (check: `weights[s] == weights[t]` whenever `coxmat[s][t]` odd); error otherwise (PyCox would silently compute nonsense).
- Coefficient overflow: all Laurent arithmetic in `i64` with `debug_assert!` checked ops; tests for medium groups guarantee headroom (largest observed coefficients are tiny).

---

## Part 2 — Golden data pipeline (exists; how to use and extend)

### 2.1 Regenerating

```bash
cd pycox-ref
python3 gen_golden.py suite        # everything small/medium (< 30 s)
python3 gen_golden.py suite-big    # + A5, F4 as .json.gz (~1 min)
python3 gen_golden.py kl H3:1 B2:2,1   # individual files
python3 gen_golden.py basics E6
```

Golden files are **generated artifacts — never edit by hand**. To change the format, change `gen_golden.py` AND `io.rs` together, regenerate everything, and bump `schema`.

### 2.2 What each kind verifies

| Golden kind | Verifies (Rust side) |
|---|---|
| `basics_*` | Cartan/Coxeter matrices, degrees, order, N, root coordinates & ordering, element enumeration counts, longest element |
| `kl_*_w*` | element table order, Bruhat order (−1 pattern), every `P̃_{y,w}`, every mu slot & value, arrows, left/right/two-sided cells, Duflo involutions with (a, n), cell order matrix |

### 2.3 Committed suite

Basics: A1–A5, B2–B4, C3, D4, D5, G2, F4, H3, H4, I5, I7, I8, E6.
KL equal: A1–A4, B2–B4, C3, D4, G2, H3, I5, I7.
KL unequal: B2:[2,1], B2:[1,2], B3:[2,1,1], G2:[1,3], G2:[3,1], I8:[1,2].
Big (gz): A5, F4. — I7/I8 KL files exist but their Rust tests stay `#[ignore]`d until Task 18 (CycInt) lands.

---

## Part 3 — Task breakdown

Execution rules for every task:
- TDD: write the failing test, see it fail, implement, see it pass, `cargo fmt && cargo clippy --all-targets -- -D warnings`, commit.
- Commit messages: conventional commits (`feat:`, `test:`, `fix:`, `docs:`, `chore:`).
- Tests live next to the module (`#[cfg(test)]`) for units, in `crates/rustcox-core/tests/` for golden/integration.
- A tiny shared test util `tests/common/mod.rs` provides `golden(name) -> serde_json::Value` (reads `golden/NAME.json[.gz]`, relative to workspace root via `env!("CARGO_MANIFEST_DIR")/../..`).

### Task 0: Workspace scaffolding  ✅ (done during planning — verify, don't redo)

`Cargo.toml` workspace, both crates compiling empty, CI (`fmt`+`clippy`+`test`), LICENSE, README, CLAUDE.md, `.gitignore`. Verify with `cargo test --workspace`.

### Task 1: Laurent polynomials (`laurent.rs`)

**Files:** Create `crates/rustcox-core/src/laurent.rs`; modify `lib.rs` (add `pub mod laurent;`).

- [ ] **Step 1 — failing tests** (in-module). Cover, with exact values:

```rust
#[test]
fn arithmetic_and_normalization() {
    let p = Laurent::from_coeffs(-2, vec![-1, 6, -12, 8]);  // -v⁻² + 6v⁻¹ - 12 + 8v
    assert_eq!(p.val(), -2);
    assert_eq!(p.degree(), Some(1));
    assert!(Laurent::zero().is_zero());
    assert_eq!(Laurent::from_coeffs(0, vec![0, 0]), Laurent::zero()); // strips to zero
    let q = Laurent::monomial(1, 2);                         // v²
    assert_eq!(&q + &Laurent::one(), Laurent::from_coeffs(0, vec![1, 0, 1]));
    assert_eq!(&q - &q, Laurent::zero());
    // (1+v²)·(1−v²) = 1−v⁴
    let a = Laurent::from_coeffs(0, vec![1, 0, 1]);
    let b = Laurent::from_coeffs(0, vec![1, 0, -1]);
    assert_eq!(&a * &b, Laurent::from_coeffs(0, vec![1, 0, 0, 0, -1]));
    // cancellation inside add: (v + 1) + (-v) = 1
    let c = Laurent::from_coeffs(0, vec![1, 1]);
    assert_eq!(&c + &Laurent::monomial(-1, 1), Laurent::one());
}
#[test]
fn parts_and_bar() {
    // f = v⁻¹ + 2 + 3v  (PyCox pospart/nonnegpart/zeropart/barpart, ≈10104–10138)
    let f = Laurent::from_coeffs(-1, vec![1, 2, 3]);
    assert_eq!(f.pos_part(), Laurent::monomial(3, 1));
    assert_eq!(f.nonneg_part(), Laurent::from_coeffs(0, vec![2, 3]));
    assert_eq!(f.zero_part(), 2);
    assert_eq!(f.bar(), Laurent::from_coeffs(-1, vec![3, 2, 1]));
    assert_eq!(Laurent::zero().zero_part(), 0);
}
#[test]
fn shift_and_eval() {
    let p = Laurent::from_coeffs(0, vec![1, 0, 1]);          // 1+v²
    assert_eq!(p.shifted(-3), Laurent::from_coeffs(-3, vec![1, 0, 1]));
    assert_eq!(p.eval_i64(2), 5);
    assert_eq!(p.scaled(-2), Laurent::from_coeffs(0, vec![-2, 0, -2]));
}
```

- [ ] **Step 2** `cargo test -p rustcox-core laurent` → compile failure (expected).
- [ ] **Step 3 — implement.** API: `zero, one, monomial(c, exp), from_coeffs(val, Vec<i64>)` (normalizes: strips leading/trailing zeros, zero ⇒ `{val:0, coeffs:[]}`), `is_zero, val, degree() -> Option<i32>, leading_coeff() -> i64, coeff(exp) -> i64`, ops `Add/Sub/Mul/Neg` on `&Laurent`, `shifted(d)` (multiply by `v^d`), `scaled(k)`, `pos_part, nonneg_part, zero_part() -> i64, bar` (reverse coeffs, `val' = −degree`), `eval_i64`. Addition aligns by `min(val)`, multiplication is convolution. ~180 lines.
- [ ] **Step 4** tests pass.
- [ ] **Step 5 — serde**: `impl Serialize/Deserialize` to/from `{"v": val, "c": [..]}` (zero ⇒ `{"v":0,"c":[]}`); test round-trip and that `serde_json::to_value(Laurent::from_coeffs(0, vec![1,0,-1]))` equals `json!({"v":0,"c":[1,0,-1]})`.
- [ ] **Step 6** Commit `feat: Laurent polynomial ring with KL part-operations`.

### Task 2: Root-coefficient rings (`ring.rs`)

**Files:** Create `crates/rustcox-core/src/ring.rs`; modify `lib.rs`.

- [ ] **Step 1 — failing tests:**

```rust
#[test]
fn golden_int_arithmetic() {
    let phi = GoldenInt::new(0, 1);                    // φ, φ² = φ+1
    assert_eq!(phi.mul(&phi), GoldenInt::new(1, 1));
    let x = GoldenInt::new(-1, 1);                     // φ−1 = 1/φ ≈ 0.618 > 0
    assert!(x.is_nonneg());
    assert!(!x.neg().is_nonneg());
    assert!(GoldenInt::new(2, -1).is_nonneg());        // 2−φ ≈ 0.382
    assert!(!GoldenInt::new(1, -1).is_nonneg() == false); // 1−φ < 0 → is_nonneg false
    assert!(GoldenInt::new(0, 0).is_nonneg());
    assert!((GoldenInt::new(-1, 1).approx() - 0.618033988749895).abs() < 1e-12);
}
```

- [ ] **Step 2** run, fails.
- [ ] **Step 3 — implement.** Trait:

```rust
pub trait RootCoeff: Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug {
    fn zero() -> Self;
    fn from_int(n: i64) -> Self;
    fn add(&self, o: &Self) -> Self;
    fn sub(&self, o: &Self) -> Self;
    fn mul(&self, o: &Self) -> Self;
    fn neg(&self) -> Self;
    fn is_zero(&self) -> bool;
    fn is_nonneg(&self) -> bool;   // exact
    fn approx(&self) -> f64;       // for the (height, rev-lex) root sort
}
```

`impl RootCoeff for i64` (trivial). `GoldenInt { a: i64, b: i64 }` = `a + bφ`: `mul` = `(ac+bd) + (ad+bc+bd)φ` (use i128 intermediates); `is_nonneg`: value = `(x + y√5)/2` with `x = 2a+b, y = b`; sign by quadrant, comparing `(x as i128)²` vs `5(y as i128)²` in mixed quadrants. `approx` = `a + b·1.618033988749895`.
- [ ] **Step 4** pass. **Step 5** Commit `feat: RootCoeff trait with i64 and GoldenInt(ℤ[φ])`.

### Task 3: Cartan data (`cartan.rs`)

**Files:** Create `crates/rustcox-core/src/cartan.rs`; modify `lib.rs`.

- [ ] **Step 1 — failing golden test** `tests/golden_basics.rs::cartan_data`: for each `basics_*.json`, compare `cartan::coxeter_mat(&components)` with `"coxetermat"`, `cartan::degrees_of` with `"degrees"`, derived order with `"order"`. (Use the shared `golden()` helper; build `components` from the `"type"` field.)
- [ ] **Step 2** fails. **Step 3 — implement** by transcribing PyCox `cartanmat` (≈2145–2282) and `degreesdata` (≈2677–2724):

```rust
pub enum CartanMat { Int(Vec<Vec<i64>>), Golden(Vec<Vec<GoldenInt>>) }
pub fn cartan_mat(series: Series, rank: usize) -> Result<CartanMat, Error>;
pub fn degrees_of(series: Series, rank: usize) -> Vec<u32>;     // A:2..=n+1, B/C:2,4..2n, D:2,4..2n-2,n, E6:[2,5,6,8,9,12], E7:[2,6,8,10,12,14,18], E8:[2,8,12,14,18,20,24,30], F4:[2,6,8,12], G2:[2,6], H3:[2,6,10], H4:[2,12,20,30], I(m):[2,m]
pub fn coxeter_mat_from_cartan(&CartanMat) -> Vec<Vec<u32>>;    // c_st·c_ts: 0→2, 1→3, 2→4, 3→6; golden entries → 5; I(m) set directly
```

**Copy the index conventions from the PyCox source exactly** (which node carries the −2 in type B vs C, the D fork labels, E numbering, H golden entries `−φ`). The golden `coxetermat`/`roots` comparisons will catch any slip. For `Series::I(m)`, `m ∈ {3,4,6}` still constructs the integer Cartan PyCox uses; `m = 5` golden; other `m` → return `Error::NeedsCyc` until Task 18.
- [ ] **Step 4** pass (I7/I8 basics assertions behind `if` on series for now, marked `// TODO(Task 18)`). **Step 5** Commit `feat: Cartan matrices, Coxeter matrices, degree data for all finite types`.

### Task 4: Root systems (`roots.rs`)

**Files:** Create `crates/rustcox-core/src/roots.rs`; modify `lib.rs`.

- [ ] **Step 1 — failing tests:** unit: A2 root system = positive roots `[[1,0],[0,1],[1,1]]` in that order, N=3, `permgens[0] = (3,2,1,0,5,4)` *(derive the expected perm by hand from the conventions in §0.3 — s0 sends α0→−α0, α1→α0+α1)*; golden: for `basics_A3/B3/F4/E6/H3`, compare `"roots"` (when present) and `N`.
- [ ] **Step 2** fails. **Step 3 — implement** PyCox `roots()` (≈2728–2755) generically:

```rust
pub struct RootSystem { pub n_pos: u32, pub roots_int: Option<Vec<Vec<i64>>>, pub permgens: Vec<Perm> }
pub fn build(cmat: &CartanMat) -> RootSystem
```

Internal generic fn over `R: RootCoeff`: BFS from simple roots; reflection step `nr[s] -= Σ_t c[s][t]·nr[t]`; keep if coordinate `nr[s]` stays nonneg; collect positive roots; sort by `(Σ approx, reverse-lex by approx)` — for `i64` use exact ints for the sort key; append negatives; build `permgens[s][i] = index of s(roots[i])` via a HashMap from coordinate vector to index (PyCox `permroots`, ≈2779).
- [ ] **Step 4** pass. **Step 5** Commit `feat: root system generation and generator permutations`.

### Task 5: CoxeterGroup + element calculus (`group.rs`, `element.rs`)

**Files:** Create both; modify `lib.rs`.

- [ ] **Step 1 — failing tests:**

```rust
#[test]
fn b2_element_calculus() {
    let w = CoxeterGroup::from_type("B2").unwrap();
    assert_eq!((w.rank, w.n_pos, w.order), (2, 4, 8));
    let p = w.word_to_perm(&[0, 1, 0]);
    assert_eq!(w.perm_length(&p), 3);
    assert_eq!(w.perm_to_word(&p), vec![0, 1, 0]);          // canonical
    assert_eq!(w.perm_to_word(&w.word_to_perm(&[1, 0, 1, 1, 0])), vec![0]); // reduces
    assert_eq!(w.left_descents(&p), vec![0]);
    assert_eq!(w.right_descents(&p), vec![0]);
    let w0 = w.longest_perm();
    assert_eq!(w.perm_length(w0), 4);
    assert_eq!(w.perm_to_word(w0), vec![0, 1, 0, 1]);
}
#[test]
fn longest_words_golden() { /* for basics_* with "longest_word": compare w.perm_to_word(w.longest_perm()) */ }
```

- [ ] **Step 2** fails. **Step 3 — implement:** constructor assembles `RootSystem` per component into block product (offset root indices per component; permgens act trivially outside their block); `order = Π degrees`; conversions/length/descents per §0.3 formulas; `perm_to_word` = greedy smallest-left-descent strip (PyCox `permtoword`); `longest_perm` = greedy left-descent strip from identity upward (PyCox `longestperm` ≈3522: start at id, repeatedly left-multiply by any `s` with `p[s] < N` until none). `Perm::then`, `Perm::inverse` free functions on slices.
- [ ] **Step 4** pass. **Step 5** Commit `feat: CoxeterGroup with word/coxelm/perm element calculus`.

### Task 6: Element enumeration (`enumerate.rs`)

**Files:** Create `crates/rustcox-core/src/enumerate.rs`; modify `lib.rs`.

- [ ] **Step 1 — failing tests:** golden `length_histogram` for every basics file that has it; golden `elms` equality against `kl_A3_w1.json` / `kl_B3_w1.json`; invariants on A4: `lft[w][s] < w ⟺ s ∈ left_descents(w)`, `inva[inva[w]] == w`, `lengths[aw0[w]] == N − lengths[w]`.
- [ ] **Step 2** fails. **Step 3 — implement:**

```rust
pub struct ElementTable {
    pub elms: Vec<Word>,              // canonical order (length, lex)
    pub coxelms: Vec<CoxElm>,
    pub index: HashMap<CoxElm, ElmIdx>,
    pub lengths: Vec<u32>,
    pub inva: Vec<ElmIdx>,
    pub aw0: Vec<ElmIdx>,
    pub lft: Vec<Vec<ElmIdx>>,        // lft[w][s] = index of s·w
}
impl ElementTable { pub fn build(w: &CoxeterGroup) -> Self; pub fn lweights(&self, weights: &[u32]) -> Vec<u32>; }
```

BFS by right multiplication on coxelms with the w0 half-way symmetry trick (PyCox `allcoxelms` ≈3925–3971: lengths > N/2 produced as `w0 · (mirror element)`); after collecting all coxelms, compute canonical words, sort, build maps. `lft` via applying `permgens[s]` to the *inverse* coxelm trick (PyCox ≈10210) or directly: `s·w = (w⁻¹·s)⁻¹`; simplest correct route: store full perms during build or recompute — keep it simple first (`Perm` per element is 2N·4 bytes; A5 720×60 fine; for big groups build `lft` from coxelm of `s·w` computed via `permgens[s]` composed with stored perm of `w`). Guard memory: only materialize full perms transiently.
- [ ] **Step 4** pass. **Step 5** Commit `feat: canonical element table with lft/inva/aw0 maps`.

### Task 7: Bruhat order (`bruhat.rs`)

**Files:** Create `crates/rustcox-core/src/bruhat.rs`.

- [ ] **Step 1 — failing test:** brute-force cross-check on A3 and B3: for all pairs `(y,w)`, `bruhat::leq` agrees with the subword criterion evaluated naively (implement the naive subword check inside the test); plus `leq(id, w)`, antisymmetry on same-length pairs.
- [ ] **Step 2** fails. **Step 3 — implement** the *iterative* `bruhatperm` (PyCox ≈3622–3652): while `l(x) < l(y)`: take a left descent `s` of `y`; if `s` descends `x` too, `x ← sx`; `y ← sy`; finally `x == y` or `l(x)==0`.
- [ ] **Step 4** pass. **Step 5** Commit `feat: Bruhat order test (iterative descent stripping)`.

### Task 8: KL storage (`kl/table.rs`, `kl/mod.rs`)

**Files:** Create `crates/rustcox-core/src/kl/{mod.rs,table.rs}`; modify `lib.rs`.

- [ ] **Step 1 — failing tests:** construct a tiny hand-built table (3 elements) and exercise accessors: `bruhat_leq`, `pol(y,w)`, `set/get` round-trips, `mu` in both modes (Implicit computes `zero_part(v^{1+L(y)−L(w)}·P)`).
- [ ] **Step 2** fails. **Step 3 — implement:**

```rust
pub struct KlOpts { pub weights: Vec<u32>, pub threads: Option<usize>, pub layer_chunk: Option<usize> }
pub enum MuMode { Implicit, Stored }
pub struct KlTable {
    pub elms: ElementTable, pub weights: Vec<u32>, pub lweights: Vec<u32>,
    pub pols: Vec<Laurent>, pub mues: Vec<Vec<Laurent>>,   // mues empty in Implicit mode
    pub mu_mode: MuMode,
    rows: Vec<KlRow>,            // rows[w].pol[y]: u32 (u32::MAX = incomparable)
}                                 // rows[w].mu: Option<Vec<u32>>, len (w+1)*rank, u32::MAX = no slot
impl KlTable {
    pub fn bruhat_leq(&self, y: ElmIdx, w: ElmIdx) -> bool;
    pub fn pol(&self, y: ElmIdx, w: ElmIdx) -> Option<&Laurent>;
    pub fn mu(&self, s: usize, y: ElmIdx, w: ElmIdx) -> Laurent;  // zero if no slot
    pub fn mu_is_nonzero(&self, s: usize, y: ElmIdx, w: ElmIdx) -> bool;
}
```

- [ ] **Step 4** pass. **Step 5** Commit `feat: KL table storage with implicit/stored mu modes`.

### Task 9: Sequential KL — equal parameters (`kl/compute.rs`)

The heart of the port. Reference: PyCox `klpolynomials` ≈10141–10380 and §1.4. Equal-parameter mode: `MuMode::Implicit`, mu slots tracked as *presence bits* only (presence ⟺ `lft[y][s] < y && lft[w][s] > w`; value derived).

**Files:** Create `crates/rustcox-core/src/kl/compute.rs`; create `crates/rustcox-core/tests/golden_kl.rs` + `tests/common/mod.rs`.

- [ ] **Step 1 — failing tests:**

```rust
// tests/golden_kl.rs
fn check_kl_golden(name: &str) {
    let g = common::golden(name);
    let table = /* build group from g["type"], run klpolynomials_seq with g["weights"] */;
    let ours = rustcox_core::io::to_canonical_json(&table, &CellData::from_table(&table));
    for key in ["elms", "pols", "klmat", "mumat"] {       // cells keys added in Task 11
        assert_eq!(ours[key], g[key], "{name}:{key}");
    }
}
#[test] fn kl_a3() { check_kl_golden("kl_A3_w1"); }
#[test] fn kl_b2() { check_kl_golden("kl_B2_w1"); }
#[test] fn kl_b3() { check_kl_golden("kl_B3_w1"); }
#[test] fn kl_h3() { check_kl_golden("kl_H3_w1"); }
// unit, in compute.rs: A3 has exactly 2 distinct pols {1, 1+v²}, 213 comparable pairs,
// exactly 6 pairs with P = 1+v²  (§0.4)
```

(For this task, `io::to_canonical_json` may be a minimal stub emitting only `elms/pols/klmat/mumat`; full version in Task 14. Canonicalization: sort pol pool by `(val, coeffs)`, remap.)
- [ ] **Step 2** fails. **Step 3 — implement** the row kernel exactly per §1.4 as a function reusable by the parallel driver:

```rust
pub(crate) struct RowBuf { pol: Vec<PolSlot>, mu_present: Vec<bool> /* (w+1)*rank */ }
pub(crate) enum PolSlot { Incomparable, Pooled(u32), Fresh(Laurent) }
pub(crate) fn compute_row(w: ElmIdx, ctx: &KlCtx<'_>, same_layer: &SameLayerView<'_>) -> RowBuf;
pub fn klpolynomials_seq(group: &CoxeterGroup, opts: &KlOpts) -> KlTable;
```

The sequential driver: `for w in 1..n { let row = compute_row(...); intern(row); }` with intern order `y` descending (matching PyCox pool order: it appends while looping y from w down to 0). Implement the Bruhat flag fused at the top of the per-`y` iteration. Mu presence bits per §1.4.
- [ ] **Step 4** A3/B2/B3/H3 golden tests pass. Run also `kl_A4_w1`, `kl_D4_w1`, `kl_C3_w1`, `kl_G2_w1`, `kl_I5_w1`.
- [ ] **Step 5** Commit `feat: sequential KL polynomial computation, equal parameters`.

### Task 10: Unequal parameters

**Files:** Modify `kl/compute.rs`, `kl/table.rs`; extend `tests/golden_kl.rs`.

- [ ] **Step 1 — failing tests:** `check_kl_golden` over `kl_B2_w2_1`, `kl_B2_w1_2`, `kl_B3_w2_1_1`, `kl_G2_w1_3`, `kl_G2_w3_1`; unit test pinning §0.4: B2 [2,1] pols sorted = `[1, 1−v², 1+v²]` as canonical JSON, duflo with the `n=−1` entry.
- [ ] **Step 2** fails. **Step 3 — implement:** `uneq = !(weights all 1)` selects `MuMode::Stored`; weight-0 branches in Cases I/II/recursion; minimal-weight descent choice; the three-branch mu computation incl. the `aw0` sign rule and the `bar(m)+m−zero_part(m)` symmetrization; per-generator mu pools with `mues[s][0] = 0`, PyCox insertion order.
- [ ] **Step 4** pass. **Step 5** Commit `feat: unequal-parameter KL polynomials and mu pools`.

### Task 11: Cells, Duflo, orders (`kl/cells.rs`)

**Files:** Create `crates/rustcox-core/src/kl/cells.rs`; extend `tests/golden_kl.rs` comparisons to all keys.

- [ ] **Step 1 — failing tests:** extend `check_kl_golden` to also compare `arrows, lcells, duflo, lorder, rcells, tcells`; unit: A3 cell sizes `[1,1,2,2,3,3,3,3,3,3]`, 54 arrows, 39 true entries in lorder; B2[2,1] lcells = `[[0],[1,4],[2],[3,6],[5],[7]]`, `checks_ok == true` everywhere.
- [ ] **Step 2** fails. **Step 3 — implement** per §1.6 (Tarjan SCC; condensation BFS for lorder; union-find for tcells):

```rust
pub struct CellData { pub arrows: Vec<(ElmIdx, ElmIdx)>, pub lcells: Vec<Vec<ElmIdx>>,
    pub duflo: Vec<(ElmIdx, i32, i64)>, pub lorder: Vec<Vec<bool>>,
    pub rcells: Vec<Vec<ElmIdx>>, pub tcells: Vec<Vec<ElmIdx>>, pub checks_ok: bool }
impl CellData { pub fn from_table(t: &KlTable) -> Self }
```

Canonical ordering of cells (sort members, sort cells lex, permute duflo/lorder along) happens here so golden comparison is direct.
- [ ] **Step 4** pass for all committed kl goldens. **Step 5** Commit `feat: left/right/two-sided cells, Duflo involutions, cell order`.

### Task 12: Parallel driver (`kl/parallel.rs`)

**Files:** Create `crates/rustcox-core/src/kl/parallel.rs`; create `tests/parallel_eq.rs`.

- [ ] **Step 1 — failing tests:**

```rust
// tests/parallel_eq.rs — KlTable must derive PartialEq (rows, pools, everything)
#[test]
fn parallel_equals_sequential() {
    for spec in ["B3", "D4", "H3", "A4"] {
        let g = CoxeterGroup::from_type(spec).unwrap();
        let opts = |t| KlOpts { weights: vec![1; g.rank], threads: Some(t), layer_chunk: None };
        let seq = kl::klpolynomials_seq(&g, &opts(1));
        for t in [2, 4, 8] {
            assert_eq!(kl::klpolynomials(&g, &opts(t)), seq, "{spec} threads={t}");
        }
    }
}
#[test]
fn parallel_uneq_equals_sequential() { /* B3 weights [2,1,1], same pattern */ }
#[test]
fn layer_chunking_is_deterministic() { /* H3, layer_chunk: Some(7) == seq */ }
```

- [ ] **Step 2** fails. **Step 3 — implement** §1.5: length layers; inverse-pair units; `rayon::ThreadPoolBuilder` honoring `opts.threads`; phase-1 `par_iter` over units calling the *same* `compute_row` kernel with a `SameLayerView` that exposes only the unit's own first row (for the pair shortcut); phase-2 sequential intern in (w asc, y desc) order. `klpolynomials` = parallel entry point (falls back to seq when `threads == Some(1)`).
- [ ] **Step 4** pass; also rerun all golden tests through the parallel path (parametrize `check_kl_golden` to run both).
- [ ] **Step 5** Commit `feat: deterministic layered parallel KL driver (rayon)`.

### Task 13: W-graphs (`wgraph.rs`)

**Files:** Create `crates/rustcox-core/src/wgraph.rs`.

- [ ] **Step 1 — failing tests:** for A3 and B3 (equal params): each left cell's W-graph from `KlTable` has vertex descent sets = left descent sets of its members; `decompose()` of each left-cell W-graph returns exactly 1 component; total vertices across cells = |W|; for B2[2,1] the mu-edge multiset is consistent with `mumat` golden (spot-check one cell by hand if needed — cell `[1,4]`: single edge pair with mu = 1).
- [ ] **Step 2** fails. **Step 3 — implement** (PyCox `wgraph` class ≈9698–10050, dict-construction path only):

```rust
pub struct WGraph { pub vertices: Vec<ElmIdx>, pub isets: Vec<Vec<Gen>>,
    pub edges: HashMap<(u32, u32), Vec<Laurent>> /* per-generator mu, indices into vertices */ }
impl WGraph { pub fn of_cell(t: &KlTable, cell: &[ElmIdx]) -> Self;
              pub fn decompose(&self) -> Vec<WGraph>; }   // SCC on nonzero-mu digraph
```

- [ ] **Step 4** pass. **Step 5** Commit `feat: per-cell W-graphs with decomposition`.

### Task 14: Canonical JSON I/O (`io.rs`)

**Files:** Create `crates/rustcox-core/src/io.rs` (replacing the Task-9 stub); extend `tests/golden_kl.rs` to full-document equality.

- [ ] **Step 1 — failing test:** for every committed `kl_*` golden: `to_canonical_json(...) == golden` as whole `serde_json::Value` documents (all keys incl. `schema/kind/type/weights/order/N/rank`); same for `basics_*` via `basics_json(&group)`. Gz support: `common::golden` must transparently read `.json.gz` (flate2), so `kl_A5_w1` and `kl_F4_w1` join the suite (mark `#[ignore = "slow"]` if > 30 s in debug; CI runs `cargo test --release -- --include-ignored`).
- [ ] **Step 2** fails. **Step 3 — implement** exporter (pool canonical sort + remap of klmat/mumat; cells already canonical from Task 11) and `basics_json`. Also `from_type_json(&Value) -> CoxeterGroup` used by tests/CLI verify.
- [ ] **Step 4** pass incl. F4 in release. **Step 5** Commit `feat: canonical golden-format JSON export/import`.

### Task 15: CLI (`rustcox-cli`)

**Files:** Modify `crates/rustcox-cli/src/main.rs`, `Cargo.toml`.

- [ ] **Step 1 — failing tests** (`assert_cmd`-style via `std::process::Command` on the built binary, or unit-test the arg-parsing + a smoke integration test):
  - `rustcox info B4` prints order=384, N=16, degrees.
  - `rustcox kl B3 -o /tmp/b3.json && rustcox verify /tmp/b3.json --against golden/kl_B3_w1.json` exits 0.
  - `rustcox kl B2 --weights 2,1 --summary` prints `npols=3 ncells=6`.
  - `rustcox kl B2 --weights 2,2,1` exits non-zero (rank mismatch).
- [ ] **Step 2** fails. **Step 3 — implement** with clap derive:

```text
rustcox info <TYPE>
rustcox kl <TYPE> [--weights w0,w1,...|k] [--threads N] [--summary] [-o FILE[.gz]]
rustcox verify <FILE> --against <GOLDEN>      # canonical-JSON equality, reports first diff
rustcox selftest [--golden-dir golden/]       # runs every golden comparison, prints PASS/FAIL table
```

`<TYPE>`: `A5`, `B4`, `H3`, `I7`, or products `A2xA1`. Exit codes: 0 ok, 1 mismatch, 2 usage.
- [ ] **Step 4** pass. **Step 5** Commit `feat: rustcox CLI (info, kl, verify, selftest)`.

### Task 16: Benchmarks & performance pass

**Files:** Create `crates/rustcox-core/benches/kl.rs`; modify `Cargo.toml` (criterion, `[[bench]]`).

- [ ] **Step 1:** criterion benches: `kl_seq_B4`, `kl_seq_F4`, `kl_par_F4_t4`, `kl_par_F4_t8`, `kl_par_H3`. Record baseline numbers in `docs/BENCHMARKS.md`.
- [ ] **Step 2:** acceptance: F4 sequential < 1 s release on Apple Silicon; F4 parallel speedup ≥ 2.5× at 8 threads; H4 equal-param parallel completes < 30 min / < 8 GB (run once manually, record; not in CI).
- [ ] **Step 3:** profile (`cargo flamegraph` or Instruments) and fix the top hotspots only if targets missed — candidate wins: avoid Laurent clones in the z-loop (borrow from pool), reuse scratch buffers per worker, `SmallVec` for short coeff vectors. No speculative tuning.
- [ ] **Step 4** Commit `perf: criterion benches and baseline numbers`.

### Task 17: Docs & release hygiene

**Files:** Modify `README.md`; create `docs/DESIGN.md`, `docs/VERIFICATION.md`, `docs/HPC.md`.

- [ ] README: quick start, CLI examples, feature matrix (what's ported vs PyCox), scaling table from §0.5.
- [ ] `docs/DESIGN.md`: distill Part 1 of this plan (update to as-built reality).
- [ ] `docs/VERIFICATION.md`: golden pipeline (Part 2), how to add a new golden case, the canonicalisation spec.
- [ ] `docs/HPC.md`: thread control (`--threads`/`RAYON_NUM_THREADS`), memory budgeting table, sample SLURM script (`#SBATCH --cpus-per-task=64`, `rustcox kl H4 --threads $SLURM_CPUS_PER_TASK -o h4.json.gz`), determinism note.
- [ ] Commit `docs: design, verification, and HPC guides`.

### Task 18 (optional, unblocks I2(m) for general m): Cyclotomic ring `CycInt`

**Files:** Modify `ring.rs`, `cartan.rs`; un-ignore I7/I8 tests.

- [ ] Port `cyclpol` (Φ_n, PyCox ≈857–874) to compute Φ_{2m}; `CycInt { m2: u32, coeffs: Vec<i64> }` = ℤ[ζ_{2m}]/(Φ_{2m}) with convolution + exact monic reduction; `2cos(π/m) = ζ + ζ⁻¹`; `is_nonneg`/`approx` via f64 evaluation at `ζ = e^{iπ/m}` (real part; PyCox uses float comparisons too — document the precision argument: root coordinates stay tiny).
- [ ] Cartan for `I(m)`: `[[2, −1], [−2cos²(π/m)·2…]]` — transcribe PyCox exactly (≈2145ff uses `ir(m) = ζ+ζ⁻¹` entries `[[2,−1],[−(ir(m)²−… )]]`; read the source).
- [ ] Tests: `basics_I7`, `basics_I8`, `kl_I7_w1`, `kl_I8_w1_2` golden all pass. Commit `feat: cyclotomic root coefficients enabling all I2(m)`.

---

## Part 4 — Verification matrix (single view)

| Layer | Test | Oracle |
|---|---|---|
| Laurent ring | unit + proptest (ring axioms vs naive `(val,BTreeMap)` model) | hand values |
| GoldenInt | unit (sign exactness incl. boundary `±(2−φ)`, `±(1−φ)`) | hand values |
| Cartan/degrees | `golden_basics` | PyCox |
| Roots/permgens | `golden_basics` roots + N | PyCox |
| Words/length/descents | unit B2 + `longest_word` golden + proptest roundtrips (A4/B3 random words) | PyCox + invariants |
| Enumeration | `length_histogram` + `elms` golden + `lft/inva/aw0` invariants | PyCox + invariants |
| Bruhat | brute subword cross-check (A3, B3) | independent algorithm |
| KL equal | `kl_*_w1` golden full-document | PyCox |
| KL unequal | `kl_*_w{2_1,...}` golden full-document | PyCox |
| Cells/Duflo | golden + `checks_ok` flags + §0.4 pinned facts | PyCox + internal checks |
| Parallel | `parallel_eq` (KlTable bit-equality, threads ∈ {2,4,8}, chunked) | sequential self |
| W-graph | structural tests vs cells | internal consistency |
| IO/CLI | golden round-trip, `verify`/`selftest` smoke | PyCox |
| Perf | criterion baselines, acceptance thresholds | recorded numbers |

Definition of done for the project: `cargo test --workspace --release -- --include-ignored` green; `rustcox selftest` prints all-PASS over `golden/`; clippy/fmt clean; docs tasks merged.

## Part 5 — Out of scope (future work, do not start without a new plan)

1. **relklpols / klcells parabolic induction + star operations** (PyCox ≈10496–11070, 11358–11560, 12053–12302) — the scalable route to cells of E6–E8 without full tables. Design notes: induction over a parabolic chain; independent `relklpols` calls per cell are the natural rayon unit; `wgraphtoklmat` round-trip needed.
2. Distinguished involutions library (`libdistinv`), `klcellreps`, two-sided cell representatives.
3. Checkpoint/restart for multi-hour runs; MPI multi-node decomposition.
4. Hecke algebra characters, class polynomials, Schur elements (separate subsystem of PyCox).
5. `from_cartan` with type recognition (`typecartanmat`/`cartantotype`, ≈2378–2650).
