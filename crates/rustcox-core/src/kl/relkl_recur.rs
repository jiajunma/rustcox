//! Inner machinery of [`relklpols`](super::relklpols): the shared index/slot
//! types, pool interning, `relmue`, the diagonal-block mu extraction, and the
//! Case-B `h` computation.
//!
//! Split out of `relkl.rs` so each file stays focused; this module is a faithful
//! transcription of the corresponding PyCox lines (see per-item docs).  On any
//! discrepancy the Python source (`pycox-ref/pycox_ref.py` 10483–10749) wins.

use std::collections::HashMap;

use crate::{
    cellgraph::{MuPools, RelKlInput, SlotData},
    element::Gen,
    laurent::Laurent,
};

// ---------------------------------------------------------------------------
// Index-space type aliases (naming discipline; not full newtypes to keep the
// arithmetic-heavy recursion readable, but every binding is named per space).
// ---------------------------------------------------------------------------

/// Coset index: a position in `X1` (the coset representatives).
pub(super) type Cx = usize;
/// Cell index: a position in `cell1.elms` (the elements of `C`).
pub(super) type Cu = usize;

/// Left-multiplication of a coset rep `X1[x]` by a W-generator `s`.
///
/// See the [`relkl` module docs](super) for the keying convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lft {
    /// `s·X1[x]` stays in `X1`, at coset index `x`.
    In(Cx),
    /// `s·X1[x]` leaves `X1`; `t` is the **W-generator** index (`J[t']`) such
    /// that `s·X1[x] = X1[x]·t'` for the W1-generator `t'`.
    Out(Gen),
}

// ---------------------------------------------------------------------------
// Working slot type (replaces PyCox 'c…c…' / 'f' / '0c0' strings).
// ---------------------------------------------------------------------------

/// A working slot in the relative-KL matrix during the recursion.
///
/// Replaces PyCox's slot strings:
/// - [`SlotState::Absent`]   ⇔ `'f'` (no entry);
/// - [`SlotState::Pending`]  ⇔ `'c'` (marked but not yet computed);
/// - [`SlotState::Done`]`{rk, mu}` ⇔ `'c<rk>c<mu>'` (a completed slot pointing
///   `rk` into `rklpols` and `mu` into `mues`).
///
/// The PyCox "zero" outcome `'0c0'` is `Done { rk: 0, mu: 0 }` (the pools are
/// seeded so index 0 is the zero polynomial in both).  Diagonal-block slots use
/// the placeholder `rk = 0` exactly as PyCox writes `'c0c<mu>'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SlotState {
    Absent,
    Pending,
    Done { rk: u32, mu: u32 },
}

impl SlotState {
    /// PyCox `slot[0] == 'c'`: a slot that is either pending or completed.  Both
    /// the initial `'c'` mark and a `'c<rk>c<mu>'` completed string start with
    /// `'c'`, so this is "marked", i.e. **not** `'f'`.
    #[inline]
    pub(super) fn is_marked(self) -> bool {
        !matches!(self, SlotState::Absent)
    }

    /// The `rk` index of a completed slot, or `None` if not yet completed.
    #[inline]
    pub(super) fn rk(self) -> Option<u32> {
        match self {
            SlotState::Done { rk, .. } => Some(rk),
            _ => None,
        }
    }

    /// The `mu` index of a completed slot, or `None`.
    #[inline]
    pub(super) fn mu(self) -> Option<u32> {
        match self {
            SlotState::Done { mu, .. } => Some(mu),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Pool interning (append-if-absent, preserving PyCox order)
// ---------------------------------------------------------------------------

/// Intern `val` into `pool`, returning its index (append if absent).  Preserves
/// PyCox's `if x in pool: pool.index(x) else: pool.append(x)` ordering exactly.
#[inline]
pub(super) fn intern(pool: &mut Vec<Laurent>, val: Laurent) -> u32 {
    if let Some(i) = pool.iter().position(|q| *q == val) {
        i as u32
    } else {
        pool.push(val);
        (pool.len() - 1) as u32
    }
}

// ---------------------------------------------------------------------------
// relmue
// ---------------------------------------------------------------------------

/// `relmue(lw, ly, p)` — coefficient of `v^(lw−ly−1)` in `p`.
///
/// PyCox (10483–10494): for the zero polynomial → 0; otherwise the leading
/// coefficient when `degree(p) == lw − ly − 1`, else 0.  A PyCox plain integer
/// `p` is a degree-0 constant here (we represent every mu as a [`Laurent`]), so
/// the int branch (`lw−ly == 1 ⇒ p`) coincides with the polynomial branch.
pub(super) fn relmue(lw: u32, ly: u32, p: &Laurent) -> Laurent {
    if p.is_zero() {
        return Laurent::zero();
    }
    // Target degree lw - ly - 1 (use i64; lw >= ly is not guaranteed).
    let target = lw as i64 - ly as i64 - 1;
    match p.degree() {
        Some(d) if d as i64 == target => Laurent::monomial(p.leading_coeff(), 0),
        _ => Laurent::zero(),
    }
}

// ---------------------------------------------------------------------------
// Diagonal-block mu extraction
// ---------------------------------------------------------------------------

/// Extract the diagonal-block mu for one `cell1.klmat[i][j]` slot.
///
/// PyCox (10637–10649): split the slot's per-generator indices; find the FIRST
/// generator `r` whose index is neither `''` nor `'0'`; intern that generator's
/// mu value `cell1.mpols[r][idx_r]` into the global `mues` pool.  If no such
/// generator exists, the mu is the zero pool index `0`.
pub(super) fn diag_block_mu(slot: &SlotData, mpols: &MuPools, mues: &mut Vec<Laurent>) -> u32 {
    let pools = match mpols {
        MuPools::PerGen(p) => p,
        MuPools::Global(g) => {
            // A Global cell1 is not the wgraphtoklmat path; fall back to the
            // single index (treat slot.mu[0] like generator 0).  Documented but
            // not exercised by the supported (PerGen) input.
            let idx = slot.mu.first().copied().unwrap_or(0);
            if idx == 0 {
                return 0;
            }
            let m = g[idx as usize].clone();
            return intern(mues, m);
        }
    };
    // Find the first generator r with a real index (PyCox skips '' and '0';
    // here the empty field is NO_SLOT and the zero index is 0 — skip both).
    for (r, &idx) in slot.mu.iter().enumerate() {
        if idx != 0 && idx != crate::cellgraph::NO_SLOT {
            let m = pools[r][idx as usize].clone();
            return intern(mues, m);
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Case-B `h` computation (the heart of the recursion)
// ---------------------------------------------------------------------------

/// Read-only context bundle for [`compute_h`] — keeps the borrow set explicit.
pub(super) struct CaseBCtx<'a> {
    pub y: Cx,
    pub x: Cx,
    pub u: Cu,
    pub v: Cu,
    pub s: usize,
    pub sx: Lft,
    pub sy: Cx,
    pub lw: &'a [u32],
    pub lw1: &'a [u32],
    pub nc: usize,
    pub bx: &'a [Vec<bool>],
    pub lft: &'a [Vec<Lft>],
    pub lft1: &'a [Vec<i64>],
    pub mat: &'a HashMap<(Cx, Cx), Vec<Vec<SlotState>>>,
    pub mues: &'a [Laurent],
    pub rklpols: &'a [Laurent],
    pub cell1: &'a RelKlInput,
}

/// `q^k` as a Laurent monomial in `v` (PyCox `q = v`, so `q^k = v^k`).
#[inline]
fn qpow(k: i64) -> Laurent {
    Laurent::monomial(1, k as i32)
}

/// Compute the relative-KL polynomial `h` for one Case-B slot `(y, x, v, u)`.
///
/// Ports PyCox 10695–10735 verbatim: the z-subtraction term, then the `sx<0`
/// (three terms) or `sx>=0` (two terms) branch.  `s` is unused directly (it is
/// encoded in `sx`/`sy`/`lft`), kept in the context for symmetry with the source.
pub(super) fn compute_h(ctx: CaseBCtx<'_>) -> Laurent {
    let CaseBCtx {
        y,
        x,
        u,
        v,
        s,
        sx,
        sy,
        lw,
        lw1,
        nc,
        bx,
        lft,
        lft1,
        mat,
        mues,
        rklpols,
        cell1,
    } = ctx;

    let bxq = |a: Cx, b: Cx| -> bool {
        // bruhatX is only defined for b <= a.
        if b <= a {
            bx[a][b]
        } else {
            false
        }
    };
    let slot =
        |a: Cx, b: Cx, vv: Cu, uu: Cu| -> Option<SlotState> { mat.get(&(a, b)).map(|g| g[vv][uu]) };

    let mut h = Laurent::zero();

    // --- z-subtraction term: for z in range(x, sy) ---------------------------
    for z in x..sy {
        let sz = lft[s][z];
        // sz < z (descending) test: In(zi) with zi < z, OR Out(_) (leaves X,
        // which PyCox treats as sz<0 < z, always "descends" for this guard).
        let sz_descends = match sz {
            Lft::In(zi) => zi < z,
            Lft::Out(_) => true, // sz < 0 < z
        };
        if sz_descends && bxq(sy, z) && bxq(z, x) {
            for ww in 0..nc {
                // (sz >= 0 or lft1[-1-sz][w] < w)
                let first_guard = match sz {
                    Lft::In(_) => true, // sz >= 0
                    Lft::Out(t) => (lft1[t as usize][ww]) < (ww as i64),
                };
                if !first_guard {
                    continue;
                }
                let (Some(zx), Some(syz)) = (slot(z, x, ww, u), slot(sy, z, v, ww)) else {
                    continue;
                };
                if !(zx.is_marked() && syz.is_marked()) {
                    continue;
                }
                // m = mues[ muidx of mat[sy,z][v][w] ]
                let Some(mu_idx) = syz.mu() else { continue };
                let m = &mues[mu_idx as usize];
                if m.is_zero() {
                    continue;
                }
                let Some(rk) = zx.rk() else { continue };
                if rk == 0 {
                    continue;
                }
                // h -= q^(Lw[y]+Lw1[v] - Lw[z] - Lw1[w]) * rklpols[rk] * m
                let exp = lw[y] as i64 + lw1[v] as i64 - lw[z] as i64 - lw1[ww] as i64;
                let term = &(&qpow(exp) * &rklpols[rk as usize]) * m;
                h = &h - &term;
            }
        }
    }

    // --- s·x branch ----------------------------------------------------------
    match sx {
        Lft::Out(t) => {
            // sx < 0: leaves X at W1-gen t (with t'·u < u).
            // term 1: (q²+1) * rklpols[rk]  from mat[sy,x][v][u]
            if let Some(st) = slot(sy, x, v, u) {
                if let Some(rk) = st.rk() {
                    if rk != 0 {
                        let q2p1 = &qpow(2) + &Laurent::one();
                        h = &h + &(&q2p1 * &rklpols[rk as usize]);
                    }
                }
            }
            // term 2: + rklpols[rk]  from mat[sy,x][v][ lft1[t][u] ]
            let l = lft1[t as usize][u];
            if (0..nc as i64).contains(&l) {
                let lu = l as usize;
                if let Some(st) = slot(sy, x, v, lu) {
                    if let Some(rk) = st.rk() {
                        if rk != 0 {
                            h = &h + &rklpols[rk as usize];
                        }
                    }
                }
            }
            // term 3: for w in u+1..nc with lft1[t][w] > w and mat[sy,x][v][w]
            // real and cell1.klmat[w][u] filled: += q^(Lw1[w]-Lw1[u]+1) * rk * m
            for ww in (u + 1)..nc {
                if lft1[t as usize][ww] <= ww as i64 {
                    continue;
                }
                let Some(st) = slot(sy, x, v, ww) else {
                    continue;
                };
                if !st.is_marked() {
                    continue;
                }
                if cell1.klmat[ww][u].is_none() {
                    continue;
                }
                // m = mues[ muidx of mat[0,0][w][u] ] (the W1-cell diagonal block)
                let Some(diag00) = slot(0, 0, ww, u) else {
                    continue;
                };
                let Some(mu_idx) = diag00.mu() else { continue };
                let m = &mues[mu_idx as usize];
                if m.is_zero() {
                    continue;
                }
                let Some(rk) = st.rk() else { continue };
                if rk == 0 {
                    continue;
                }
                let exp = lw1[ww] as i64 - lw1[u] as i64 + 1;
                let term = &(&qpow(exp) * &rklpols[rk as usize]) * m;
                h = &h + &term;
            }
        }
        Lft::In(sxi) => {
            // sx >= 0: s descends both, stays in X.  (Here sxi < x by fs1/ldy[0].)
            // term 1: + rklpols[rk] from mat[sy, sx][v][u]
            if let Some(st) = slot(sy, sxi, v, u) {
                if let Some(rk) = st.rk() {
                    if rk != 0 {
                        h = &h + &rklpols[rk as usize];
                    }
                }
            }
            // term 2: + q² * rklpols[rk] from mat[sy,x][v][u] (if x <= sy & bruhat)
            if x <= sy && bxq(sy, x) {
                if let Some(st) = slot(sy, x, v, u) {
                    if let Some(rk) = st.rk() {
                        if rk != 0 {
                            h = &h + &(&qpow(2) * &rklpols[rk as usize]);
                        }
                    }
                }
            }
        }
    }

    h
}
