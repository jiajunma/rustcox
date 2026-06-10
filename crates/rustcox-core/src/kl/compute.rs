//! Sequential Kazhdan–Lusztig polynomial computation.
//!
//! This is the heart of the port.  Reference:
//! `pycox-ref/pycox_ref.py::klpolynomials` (≈10141–10470).  The PyCox source
//! is normative; this module is a faithful translation of **both** the
//! `uneq == False` (equal-parameter, Task 9) and `uneq == True`
//! (unequal-parameter, Task 10) paths.
//!
//! # Algorithm overview
//!
//! Elements are processed in canonical order (rows `w = 1..n`).  For each row
//! `w`, the inner loop runs `y` from `w` **down to** `0`; that descending
//! order is what makes the same-row shortcut reads (inverse symmetry, Case I,
//! Case II) always reference an already-computed higher index, what makes the
//! polynomial pool grow in PyCox's exact insertion order, and what makes the
//! within-row mu reads of the unequal-parameter mu phase reference an
//! already-computed column `z > y`.
//!
//! Each `(w, y)` entry gets three things, in order:
//!
//! 1. **Bruhat flag** ([`bruhat_flag`]) — comparable (`c`) or not (`f`).
//! 2. **P̃_{y,w}** ([`compute_h`] / [`compute_h_uneq`]) — the KL polynomial.
//! 3. **mu** — in equal-parameter mode a presence flag (value derived on
//!    demand from the polynomial, [`MuMode::Implicit`]); in unequal-parameter
//!    mode an explicit Laurent value interned into per-generator pools
//!    ([`MuMode::Stored`]).
//!
//! # Equal vs unequal parameters
//!
//! `uneq = not all(weights == 1)`.  The two paths diverge in three places,
//! mirroring PyCox:
//!
//! - **MuMode.**  Equal ⇒ `Implicit` (only presence flags stored, mu derived
//!   from `P̃`).  Unequal ⇒ `Stored` (Laurent mu values interned per generator).
//! - **P̃ recursion.**  Unequal adds `poids[s] == 0` shortcut branches, a
//!   recursion-generator swap to a left descent of minimal weight, a
//!   `v^{2·poids[s]}` term, and reads stored mu values from completed rows.
//! - **mu phase.**  Unequal computes a full Laurent mu (PyCox ≈10346–10370)
//!   with a `bar`-symmetrisation and a within-row `pos_part` z-loop, instead
//!   of the equal-parameter presence flag.
//!
//! # Interface for the parallel driver (Task 12)
//!
//! [`compute_row`] is the single reusable row kernel.  It is parameterised
//! over a read-only [`KlCtx`] (borrows of the element table, weights,
//! `lweights`, the **completed** rows, the **frozen** global polynomial pool,
//! and the **frozen** mu pools) and returns a [`RowResult`]: the per-`y`
//! polynomial *values* of this row, the mu presence flags (equal) or mu
//! *values* (unequal).  The kernel never touches the global pools for
//! interning — same-row shortcut reads come from the row's own freshly
//! computed values.  This makes the kernel pure (no shared mutable state),
//! which is exactly what the layered parallel driver needs: many rows of a
//! length layer can be computed concurrently against the frozen lower layers,
//! then interned sequentially afterwards in `(w asc, y desc)` order to
//! reproduce the sequential pools exactly.
//!
//! The sequential driver ([`klpolynomials_seq`]) calls [`compute_row`] for
//! each `w` and immediately interns its [`RowResult`] in descending-`y` order.

use crate::{
    element::ElmIdx,
    enumerate::ElementTable,
    group::CoxeterGroup,
    kl::{
        table::{KlRow, KlTable, MuMode, NOT_LEQ, NO_MU},
        KlError, KlOpts,
    },
    laurent::Laurent,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Read-only context for the row kernel
// ---------------------------------------------------------------------------

/// Read-only context shared by every invocation of [`compute_row`].
///
/// Holds borrows of the immutable element table, the weight data, the
/// **completed** rows, the **frozen** global polynomial pool, and (in
/// unequal-parameter mode) the **frozen** per-generator mu pools that the
/// kernel reads from.  All reads target strictly-shorter rows (full recursion,
/// `aw0` symmetry, Cases I/II shorter-row reads) or the inverse row (inverse
/// symmetry, same length — completed earlier in the ascending-`w` driver loop).
/// Same-row reads do **not** use this context; they read the in-progress row's
/// own computed values.
pub(crate) struct KlCtx<'a> {
    /// Canonical element table (lengths, `lft`, `inva`, `aw0`).
    pub elms: &'a ElementTable,
    /// `N` = number of positive roots = maximum element length.
    pub n_pos: u32,
    /// `lweights[i] = L(w_i)`.  Equal parameters ⇒ `lweights == lengths`.
    pub lweights: &'a [u32],
    /// Generator weights `L(s)`.  Equal parameters ⇒ all `1`.
    pub weights: &'a [u32],
    /// Whether any weight differs from `1` (unequal-parameter path).
    pub uneq: bool,
    /// Completed rows (`rows[z]` for every `z` already finished).
    pub rows: &'a [KlRow],
    /// The frozen global polynomial pool (`pols[0] == 1`).
    pub pols: &'a [Laurent],
    /// The frozen per-generator mu pools (`Stored` mode only; empty otherwise).
    pub mues: &'a [Vec<Laurent>],
}

impl KlCtx<'_> {
    /// Pool lookup for a completed entry `mat[w][y]`.  `None` when `y ≰ w`.
    #[inline]
    pub(crate) fn pol_at(&self, w: ElmIdx, y: ElmIdx) -> Option<&Laurent> {
        if y == w {
            return Some(&self.pols[0]);
        }
        let idx = self.rows[w as usize].pol[y as usize];
        if idx == NOT_LEQ {
            None
        } else {
            Some(&self.pols[idx as usize])
        }
    }

    /// Whether `y ≤ w` in Bruhat order, reading the completed flag matrix.
    #[inline]
    pub(crate) fn leq(&self, y: ElmIdx, w: ElmIdx) -> bool {
        if y == w {
            return true;
        }
        self.rows[w as usize].pol[y as usize] != NOT_LEQ
    }

    /// `μ^s_{z,w}` as a scalar (equal-parameter derivation), reading a
    /// completed row `w`.  Equals the coefficient of `v^{L(w)−L(z)−1}` in
    /// `P̃_{z,w}` — i.e. `zero_part(v^{1+L(z)−L(w)} · P̃_{z,w})`.  `0` if `z ≰ w`.
    #[inline]
    fn mu_scalar(&self, z: ElmIdx, w: ElmIdx) -> i64 {
        let Some(h) = self.pol_at(w, z) else {
            return 0;
        };
        let shift = 1i32 + self.lweights[z as usize] as i32 - self.lweights[w as usize] as i32;
        h.coeff(-shift)
    }
}

// ---------------------------------------------------------------------------
// Row result
// ---------------------------------------------------------------------------

/// A polynomial slot in a freshly computed row.
///
/// `Incomparable` means `y ≰ w`; `Value(p)` holds the computed `P̃_{y,w}`
/// before interning.  In unequal-parameter mode `Value` may carry the zero
/// polynomial (PyCox can produce `P̃ == 0`).
#[derive(Clone, Debug)]
pub(crate) enum PolSlot {
    Incomparable,
    Value(Laurent),
}

/// The output of [`compute_row`]: per-`y` polynomial values plus mu data, not
/// yet interned into the global pools.
pub(crate) struct RowResult {
    /// `pol[y]` is the computed `P̃_{y,w}` (or `Incomparable`), `y` in `0..=w`.
    pub pol: Vec<PolSlot>,
    /// Equal-parameter (`Implicit`) mode: flat `(w+1) * rank` mu-slot presence
    /// flags.  `None` in unequal-parameter mode.
    pub mu_present: Option<Vec<bool>>,
    /// Unequal-parameter (`Stored`) mode: flat `(w+1) * rank` mu values, where
    /// `None` is an absent slot (`NO_MU`) and `Some(p)` is a present value (`p`
    /// may be zero).  `None` (the outer `Option`) in equal-parameter mode.
    pub mu_vals: Option<Vec<Option<Laurent>>>,
}

// ---------------------------------------------------------------------------
// Bruhat flag (Phase A)
// ---------------------------------------------------------------------------

/// Compute the Bruhat-comparability flag for `(w, y)` with `0 < y < w`.
///
/// Faithful to PyCox ≈10263–10273.  Reads only completed rows: the `aw0`
/// branch reads `mat[aw0[y]][aw0[w]]` (length `N − l(y) < l(w)`); the descent
/// branch reads `mat[sw][..]` with `sw = lft(w,s) < w`.
fn bruhat_flag(w: ElmIdx, y: ElmIdx, first_desc: usize, ctx: &KlCtx<'_>) -> bool {
    let elms = ctx.elms;
    let lw = &elms.lengths;

    // Same length, distinct elements ⇒ never comparable.
    if lw[y as usize] == lw[w as usize] {
        return false;
    }
    // Long-element symmetry: copy the flag from the shorter mirror row.
    // PyCox reads mat[aw0[y]][aw0[w]] = row aw0[y], column aw0[w]; since
    // l(aw0[y]) = N − l(y) < l(w) the row is already complete, and
    // aw0[w] ≤ aw0[y] indexwise (l(aw0[w]) = N − l(w) ≤ N − l(y)).
    if lw[w as usize] + lw[y as usize] > ctx.n_pos {
        let ay = elms.aw0[y as usize]; // row
        let aw = elms.aw0[w as usize]; // column ≤ row
        return ctx.leq(aw, ay);
    }
    // Otherwise strip the first left descent `s` of `w`.
    let sw = elms.lft(w, first_desc);
    let sy = elms.lft(y, first_desc);
    if sy < y {
        // s descends y too: comparable iff sy ≤ sw and sy ≤_B sw.
        sy <= sw && ctx.leq(sy, sw)
    } else {
        // s ascends y: comparable iff y ≤ sw and y ≤_B sw.
        y <= sw && ctx.leq(y, sw)
    }
}

// ---------------------------------------------------------------------------
// KL polynomial (Phase B), equal parameters
// ---------------------------------------------------------------------------

/// Find the first generator `s` (in `0..rank`) such that `lft(y, s) > y` and
/// `lft(w, s) < w`.  Returns `Some(s)` on the first match, `None` if none.
///
/// Used by both Case I (with the original pair `(y, w)`) and Case II (with the
/// inverse pair `(iy, iw)`).
#[inline]
fn find_desc_asc(elms: &ElementTable, y: ElmIdx, w: ElmIdx, rank: usize) -> Option<usize> {
    (0..rank).find(|&s| elms.lft(y, s) > y && elms.lft(w, s) < w)
}

/// Compute `P̃_{y,w}` for a comparable pair `(w, y)`, **equal parameters**.
///
/// `cur` holds this row's already-computed polynomial slots for indices
/// `y' > y` (filled by the descending loop); same-row shortcut reads come
/// from there.  All other reads go through `ctx` (completed rows).
///
/// Faithful to PyCox ≈10286–10337 with every `poids[s] == 0` branch omitted
/// (unreachable when all weights are `1`).
fn compute_h(w: ElmIdx, y: ElmIdx, first_desc: usize, ctx: &KlCtx<'_>, cur: &[PolSlot]) -> Laurent {
    let elms = ctx.elms;
    let rank = elms.rank;

    // 1. Diagonal.
    if y == w {
        return Laurent::one();
    }

    let iw = elms.inva[w as usize];
    let iy = elms.inva[y as usize];

    // 2. Inverse symmetry: h = P̃_{inva[y], inva[w]}.
    if iw < w || (iw == w && iy > y) {
        if iw == w {
            // same row, higher index iy > y.
            return same_row_value(cur, iy);
        }
        // shorter (or earlier) row inva[w].
        return ctx
            .pol_at(iw, iy)
            .cloned()
            .expect("inverse-symmetry entry must be comparable");
    }

    // 3. Case I: first s with lft(y,s) > y and lft(w,s) < w.
    // PyCox ≈10302–10309
    if let Some(s) = find_desc_asc(elms, y, w, rank) {
        let sy = elms.lft(y, s); // > y, same row
        return same_row_value(cur, sy);
    }

    // 4. Case II: same search on the inverses.
    // PyCox ≈10310–10318
    if let Some(s) = find_desc_asc(elms, iy, iw, rank) {
        let sy = elms.lft(iy, s); // > iy
        let idx = elms.inva[sy as usize]; // length l(y)+1 ⇒ > y, same row
        return same_row_value(cur, idx);
    }

    // 5. Full recursion.  s = first left descent of w (== first_desc); since
    //    Cases I/II failed, this s also descends y.
    let s = first_desc;
    let sw = elms.lft(w, s);
    let sy = elms.lft(y, s);

    // h = P̃_{sy,sw}  (shorter row, comparable in this branch).
    let mut h = ctx
        .pol_at(sw, sy)
        .cloned()
        .expect("recursion base P̃_{sy,sw} must be comparable");

    // + v^{2·weights[s]} · P̃_{y,sw}  if y ≤ sw and y ≤_B sw.
    if y <= sw && ctx.leq(y, sw) {
        if let Some(p) = ctx.pol_at(sw, y) {
            let shift = 2 * ctx.weights[s] as i32;
            h += &p.shifted(shift);
        }
    }

    // − Σ_{z = sw−1 down to y}  μ^s_{z,sw} · v^{L(w)−L(z)} · P̃_{y,z}
    //   over z with lft(z,s) < z and y ≤_B z and z ≤_B sw.
    let lw_w = ctx.lweights[w as usize] as i32;
    for z in (y..sw).rev() {
        if elms.lft(z, s) >= z {
            continue;
        }
        if !ctx.leq(y, z) || !ctx.leq(z, sw) {
            continue;
        }
        // PyCox reads the stored slot mues[s][mat[sw][z]...]; that slot exists
        // iff lft(z,s) < z (loop guard) and lft(sw,s) > sw.  The latter holds
        // because s descends w (so s·sw = w > sw, i.e. s ascends sw); pin it.
        debug_assert!(
            elms.lft(sw, s) > sw,
            "mu slot (sw={sw}, z={z}, s={s}) absent: s does not ascend sw"
        );
        let m = ctx.mu_scalar(z, sw);
        if m == 0 {
            continue;
        }
        // P̃_{y,z}: shorter row z.
        let Some(pyz) = ctx.pol_at(z, y) else {
            continue;
        };
        let shift = lw_w - ctx.lweights[z as usize] as i32;
        // h -= m · v^shift · P̃_{y,z}  (single-pass: no intermediate allocation)
        h -= &pyz.shift_scaled(shift, m);
    }

    h
}

/// Read a same-row entry `mat[w][idx]` from this row's computed slots.
#[inline]
pub(crate) fn same_row_value(cur: &[PolSlot], idx: ElmIdx) -> Laurent {
    match &cur[idx as usize] {
        PolSlot::Value(p) => p.clone(),
        PolSlot::Incomparable => {
            unreachable!("same-row read of an incomparable entry at index {idx}")
        }
    }
}

// ---------------------------------------------------------------------------
// Row kernel (Phases A + B + C)
// ---------------------------------------------------------------------------

/// Compute one row `w` of the KL table as pure data.
///
/// Returns the per-`y` polynomial values plus mu data (presence flags in
/// equal-parameter mode, explicit values in unequal-parameter mode).  The
/// caller (sequential or parallel driver) interns the values into the global
/// pools.
pub(crate) fn compute_row(w: ElmIdx, ctx: &KlCtx<'_>) -> RowResult {
    let elms = ctx.elms;
    let rank = elms.rank;
    let w_us = w as usize;

    // First left descent of w (always exists for w > 0).
    let mut first_desc = 0usize;
    while elms.lft(w, first_desc) > w {
        first_desc += 1;
    }

    let mut pol = vec![PolSlot::Incomparable; w_us + 1];

    // Equal mode tracks presence flags; unequal mode tracks explicit values.
    let mut mu_present = if ctx.uneq {
        None
    } else {
        Some(vec![false; (w_us + 1) * rank])
    };
    let mut mu_vals: Option<Vec<Option<Laurent>>> = if ctx.uneq {
        Some(vec![None; (w_us + 1) * rank])
    } else {
        None
    };

    // Diagonal: P̃_{w,w} = 1.
    pol[w_us] = PolSlot::Value(Laurent::one());

    // Per-y scratch buffer for unequal-parameter mu values.  Hoisted out of
    // the loop to avoid repeated allocation; reset to None at the top of each
    // comparable-y iteration.
    let mut computed: Vec<Option<Laurent>> = vec![None; rank];

    // y from w−1 down to 0.
    for y in (0..w).rev() {
        // ---- Phase A: Bruhat flag ----
        let comparable = if y == 0 {
            true
        } else {
            bruhat_flag(w, y, first_desc, ctx)
        };
        if !comparable {
            continue; // PolSlot::Incomparable already in place; flags false.
        }

        // ---- Phase B: P̃_{y,w} ----
        let h = if ctx.uneq {
            crate::kl::compute_uneq::compute_h_uneq(w, y, first_desc, ctx, &pol)
        } else {
            compute_h(w, y, first_desc, ctx, &pol)
        };

        // ---- Phase C: mu ----
        if ctx.uneq {
            // Stored mode: compute explicit Laurent values for present slots.
            // The mu values of this row's columns z > y are already filled in
            // mu_vals; pass an immutable view to compute_mu_uneq, then write.
            let row_mu = mu_vals.as_ref().expect("uneq: mu_vals present");
            let base = y as usize * rank;
            // Columns z > y of this row are already filled (descending-y loop);
            // they live at flat indices `z*rank + s` > `base`.  Pass the whole
            // buffer so the z-loop (z > y) reads those completed columns; the
            // current column's `base..base+rank` slots are still None and are
            // never read (z > y strictly).
            // Reset the scratch buffer for this y.
            for slot in computed.iter_mut() {
                *slot = None;
            }
            for (s, slot) in computed.iter_mut().enumerate() {
                if ctx.weights[s] > 0 && elms.lft(y, s) < y && elms.lft(w, s) > w {
                    let m =
                        crate::kl::compute_uneq::compute_mu_uneq(w, y, s, &h, ctx, row_mu, rank);
                    *slot = Some(m);
                }
            }
            // Now write h and the computed mu values into the row.
            pol[y as usize] = PolSlot::Value(h);
            let muv = mu_vals.as_mut().expect("uneq: mu_vals present");
            for (s, c) in computed.iter_mut().enumerate() {
                muv[base + s] = c.take();
            }
        } else {
            pol[y as usize] = PolSlot::Value(h);
            let present = mu_present.as_mut().expect("equal: mu_present present");
            for s in 0..rank {
                if elms.lft(y, s) < y && elms.lft(w, s) > w {
                    present[y as usize * rank + s] = true;
                }
            }
        }
    }

    RowResult {
        pol,
        mu_present,
        mu_vals,
    }
}

// ---------------------------------------------------------------------------
// Sequential driver
// ---------------------------------------------------------------------------

/// Compute the full KL table sequentially.
///
/// Validates `opts` against `group`, selects [`MuMode`] from the
/// equal/unequal-parameter split (`uneq = not all weights == 1`), builds the
/// element table and an empty table, seeds row `0`, and fills rows `1..n` via
/// [`compute_row`], interning each row's values in `(w asc, y desc)` order so
/// the pools grow in PyCox's exact insertion order.
pub fn klpolynomials_seq(group: &CoxeterGroup, opts: &KlOpts) -> Result<KlTable, KlError> {
    opts.validate(group)?;

    let uneq = opts.weights.iter().any(|&wt| wt != 1);
    let mu_mode = if uneq {
        MuMode::Stored
    } else {
        MuMode::Implicit
    };

    let elms = ElementTable::build(group);
    let n = elms.len();
    let n_pos = elms.lengths.iter().copied().max().unwrap_or(0);
    let mut table = KlTable::new_empty(elms, opts.weights.clone(), mu_mode);
    let rank = table.elms.rank;

    // Dedup map beside the pol pool.  Seeded with pols[0] = one at id 0.
    let mut pool_index: HashMap<Laurent, u32> = HashMap::new();
    pool_index.insert(Laurent::one(), 0);

    // Dedup maps beside each mu pool (Stored mode).  Seeded with zero at id 0.
    let mut mu_index: Vec<HashMap<Laurent, u32>> = (0..rank)
        .map(|_| {
            let mut m = HashMap::new();
            if uneq {
                m.insert(Laurent::zero(), 0);
            }
            m
        })
        .collect();

    // Row 0: the identity.  Only y == 0 (itself); P̃ = 1; no mu slots.
    let row0 = if uneq {
        KlRow {
            pol: vec![0],
            mu: Some(vec![NO_MU; rank]),
            mu_present: None,
        }
    } else {
        KlRow {
            pol: vec![0],
            mu: None,
            mu_present: Some(vec![false; rank]),
        }
    };
    table.rows.push(row0);

    for w in 1..n {
        // Build the read-only context over the completed rows + frozen pools.
        let ctx = KlCtx {
            elms: &table.elms,
            n_pos,
            lweights: &table.lweights,
            weights: &table.weights,
            uneq,
            rows: &table.rows,
            pols: &table.pols,
            mues: &table.mues,
        };
        let result = compute_row(w as ElmIdx, &ctx);

        // Intern the row's values in descending-y order (matches PyCox pool
        // growth).  The diagonal value (y == w) is the constant 1 = pols[0].
        let row = intern_row(
            &mut table.pols,
            &mut pool_index,
            &mut table.mues,
            &mut mu_index,
            result,
            rank,
        );
        table.rows.push(row);
    }

    Ok(table)
}

/// Intern a [`RowResult`] into the global pools in descending-`y` order,
/// returning the storage [`KlRow`].
fn intern_row(
    pols: &mut Vec<Laurent>,
    pool_index: &mut HashMap<Laurent, u32>,
    mues: &mut [Vec<Laurent>],
    mu_index: &mut [HashMap<Laurent, u32>],
    result: RowResult,
    rank: usize,
) -> KlRow {
    let len = result.pol.len();
    let mut pol_ids = vec![NOT_LEQ; len];

    // Optional stored-mu id buffer, parallel to the value buffer.
    let mut mu_ids: Option<Vec<u32>> = result.mu_vals.as_ref().map(|_| vec![NO_MU; len * rank]);

    // Descending y so pool insertion order matches PyCox exactly.  For each y
    // we first intern the polynomial, then (Stored mode) intern its rank mu
    // slots in ascending-s order, matching PyCox's per-(w,y) `for s in rank`.
    let RowResult {
        pol,
        mu_present,
        mu_vals,
    } = result;

    for y in (0..len).rev() {
        match &pol[y] {
            PolSlot::Incomparable => {} // leave NOT_LEQ
            PolSlot::Value(p) => {
                pol_ids[y] = intern_pol(pols, pool_index, p.clone());
            }
        }
        if let (Some(vals), Some(ids)) = (mu_vals.as_ref(), mu_ids.as_mut()) {
            for s in 0..rank {
                if let Some(m) = &vals[y * rank + s] {
                    ids[y * rank + s] = intern_mu(&mut mues[s], &mut mu_index[s], m.clone());
                }
            }
        }
    }

    KlRow {
        pol: pol_ids,
        mu: mu_ids,
        mu_present,
    }
}

/// Intern `p` into the polynomial pool, deduplicating via `pool_index`.
fn intern_pol(pols: &mut Vec<Laurent>, pool_index: &mut HashMap<Laurent, u32>, p: Laurent) -> u32 {
    if let Some(&id) = pool_index.get(&p) {
        return id;
    }
    let id = pols.len() as u32;
    pool_index.insert(p.clone(), id);
    pols.push(p);
    id
}

/// Intern `m` into a per-generator mu pool, deduplicating via `mu_index`.
///
/// The zero polynomial is seeded at index `0` by the driver, so a zero value
/// always resolves to id `0` (preserving the pool invariant that zero never
/// lives at an index ≥ 1).
fn intern_mu(pool: &mut Vec<Laurent>, index: &mut HashMap<Laurent, u32>, m: Laurent) -> u32 {
    if let Some(&id) = index.get(&m) {
        return id;
    }
    let id = pool.len() as u32;
    index.insert(m.clone(), id);
    pool.push(m);
    id
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a KlTable for the named group using equal parameters.
    fn table_for(spec: &str) -> KlTable {
        let g = CoxeterGroup::from_type(spec).unwrap();
        let opts = KlOpts::equal(g.rank);
        klpolynomials_seq(&g, &opts).unwrap()
    }

    /// A3: exactly 2 distinct pols {1, 1+v²}; 213 comparable pairs;
    /// exactly 6 pairs with P = 1+v².  (Plan §0.4.)
    #[test]
    fn a3_facts() {
        let t = table_for("A3");
        let n = t.elms.len();

        let one = Laurent::one();
        let one_plus_v2 = Laurent::from_coeffs(0, vec![1, 0, 1]);
        assert_eq!(t.pols.len(), 2, "A3 should have exactly 2 distinct pols");
        assert!(t.pols.contains(&one), "pool must contain 1");
        assert!(t.pols.contains(&one_plus_v2), "pool must contain 1+v²");

        let mut comparable = 0usize;
        let mut count_1v2 = 0usize;
        for w in 0..n {
            for y in 0..=w {
                let p = t.pol(y as u32, w as u32);
                if p.is_some() {
                    comparable += 1;
                    if p == Some(&one_plus_v2) {
                        count_1v2 += 1;
                    }
                }
            }
        }
        assert_eq!(comparable, 213, "A3 comparable pairs");
        assert_eq!(count_1v2, 6, "A3 pairs with P = 1+v²");
    }

    /// B2: every comparable pair has P = 1 (single pol in the pool).
    #[test]
    fn b2_all_trivial() {
        let t = table_for("B2");
        assert_eq!(t.pols.len(), 1, "B2 should have a single pol (= 1)");
        assert_eq!(t.pols[0], Laurent::one());

        let n = t.elms.len();
        let mut comparable = 0usize;
        for w in 0..n {
            for y in 0..=w {
                if let Some(p) = t.pol(y as u32, w as u32) {
                    comparable += 1;
                    assert_eq!(*p, Laurent::one(), "B2 P_{{{y},{w}}} should be 1");
                }
            }
        }
        // B2: 33 comparable pairs (incl. diagonal) — confirmed against
        // golden/kl_B2_w1.json's klmat.
        assert_eq!(comparable, 33, "B2 comparable pairs");
    }

    /// Row shape: diagonal pol-id is 0, lengths are `w+1`, ids in range.
    #[test]
    fn row_shape_invariants() {
        let t = table_for("B3");
        let n = t.elms.len();
        for w in 0..n {
            let row = &t.rows[w];
            assert_eq!(row.pol.len(), w + 1, "row {w} length");
            assert_eq!(row.pol[w], 0, "diagonal pol-id is 0 at w={w}");
            for y in 0..w {
                let idx = row.pol[y];
                assert!(
                    idx == NOT_LEQ || (idx as usize) < t.pols.len(),
                    "row {w} y {y} pol id {idx} out of range"
                );
            }
        }
    }
}
