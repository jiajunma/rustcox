# rustcox — agent guide

Rust rewrite of PyCox's Kazhdan–Lusztig machinery.
**Status: Phase 1 (tasks 0–18) + Phase 2 (tasks P1–P8) complete.**

Read this first, then the plan:
`docs/superpowers/plans/2026-06-10-rustcox-implementation.md` (task list,
detailed design, and conventions). For as-built architecture see `docs/DESIGN.md`;
for the oracle pipeline see `docs/VERIFICATION.md`; for HPC notes see
`docs/HPC.md`.

## Ground rules

1. **The oracle is PyCox.** `pycox-ref/pycox_ref.py` is the normative
   reference; cited line numbers in the plan refer to that file. When in doubt
   about an algorithm, read the Python source — do not improvise mathematics.
2. **Golden files are generated, never hand-edited.** Regenerate with
   `cd pycox-ref && python3 gen_golden.py suite`. If you change the canonical
   format, change `gen_golden.py` and `crates/rustcox-core/src/io.rs`
   together, regenerate everything, and bump the `schema` field.
3. **TDD.** Write the failing test first (golden-backed where possible), then
   implement. Expected values for small groups are pinned in plan §0.4.
4. **Determinism.** The parallel KL driver must produce byte-identical
   `KlTable`s to `klpolynomials_seq` (including pool order). Never introduce
   nondeterministic pool/interning order.
5. **Style:** no `unsafe`; files ≤ ~600 lines; `cargo fmt` +
   `cargo clippy --all-targets -- -D warnings` clean before every commit;
   conventional commit messages (`feat:`, `test:`, `fix:`, `docs:`, `chore:`).
6. **License:** GPL-3.0-or-later. Do not vendor non-GPL-compatible code.

## Toolchain on this machine

The MacPorts rustc in `/opt/local/bin` is ancient (1.71). Use the rustup
toolchain (installed user-space, no shell-profile changes):

```bash
export PATH="$HOME/.cargo/bin:$PATH"   # required in every shell/session
```

## Commands

```bash
cargo test --workspace                       # unit + golden tests
cargo test --workspace --release -- --include-ignored   # + slow golden (A5, F4)
cargo clippy --all-targets -- -D warnings
cd pycox-ref && python3 gen_golden.py kl B3:2,1,1   # one-off KL golden file
cd pycox-ref && python3 gen_golden.py cells B4       # cells golden file (Phase 2)
python3 -c "from pycox_ref import *; ..."    # interrogate the oracle (run inside pycox-ref/)
```

## HPC (XMU cluster)

Big groups (H4, rank-6) run on the **XMU HPC via SLURM**. Login node `mu012`
(2 c / 8 GB) is for build + sync only; `$HOME=/public/home/majj` on Lustre is
shared to all compute nodes, which have **no internet** — so build the binary
on the login node first. Full details (cluster facts, account/qos, rsync
workflow) are in `docs/HPC.md` § *XMU cluster access*; the submit scripts are
versioned in `hpc/`.

```bash
# sync up → build on login node → submit → pull results down
rsync -az --exclude='.git' --exclude='target' --exclude='results' ./ majj@10.26.14.64:/public/home/majj/rustcox/
ssh majj@10.26.14.64 'cd rustcox && export PATH=$HOME/.cargo/bin:$PATH && cargo build --release && sbatch hpc/h4_determinism.sbatch'
rsync -az majj@10.26.14.64:/public/home/majj/rustcox/results/ ./results/
```

## Key conventions (details in plan §0.3)

- Polynomials in `v`; classical `q = v²`. Unequal parameters ⇒ negative
  coefficients are legal.
- Roots: `0..N` positive, `N..2N` negative; perm composition
  `then(p,q)[i] = q[p[i]]`; length = #{i<N : perm[i] ≥ N}; left descent `s` ⟺
  `perm[s] ≥ N`.
- Canonical word = strip smallest left descent repeatedly; canonical element
  order = (length, word lex). All golden indices use it.
