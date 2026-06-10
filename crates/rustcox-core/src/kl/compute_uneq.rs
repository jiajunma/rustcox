//! Unequal-parameter KL helpers: `compute_h_uneq`, `compute_mu_uneq`, and
//! `mu_stored`.
//!
//! Factored out of `compute.rs` (was >800 lines) to keep file sizes in budget.
//! Public API (compute_row, klpolynomials_seq, RowResult, KlCtx) stays in
//! compute.rs.  All items here are `pub(crate)` — internal to rustcox-core.

// TODO(drift-risk): steps 1-4 duplicated with compute_h — see review Task 10

use crate::{
    element::ElmIdx,
    enumerate::ElementTable,
    kl::{
        compute::{same_row_value, KlCtx, PolSlot},
        table::NO_MU,
    },
    laurent::Laurent,
};

// ---------------------------------------------------------------------------
// mu_stored  (moved from KlCtx impl in compute.rs)
// ---------------------------------------------------------------------------

/// `μ^s_{z,w}` as a stored Laurent value (unequal-parameter path), reading
/// a completed row `w`.  `None` when the slot is absent (`NO_MU`); the
/// returned reference may be the zero polynomial.
#[inline]
pub(crate) fn mu_stored<'a>(
    ctx: &'a KlCtx<'_>,
    s: usize,
    z: ElmIdx,
    w: ElmIdx,
) -> Option<&'a Laurent> {
    if z == w {
        return None;
    }
    let row = &ctx.rows[w as usize];
    let mu_vec = row.mu.as_ref()?;
    let rank = ctx.elms.rank;
    let idx = mu_vec[z as usize * rank + s];
    if idx == NO_MU {
        None
    } else {
        Some(&ctx.mues[s][idx as usize])
    }
}

// ---------------------------------------------------------------------------
// KL polynomial (Phase B), unequal parameters
// ---------------------------------------------------------------------------

/// Compute `P̃_{y,w}` for a comparable pair `(w, y)`, **unequal parameters**.
///
/// Faithful to PyCox ≈10286–10337 *including* every `poids[s] == 0` branch,
/// the recursion-generator swap to a left descent of minimal weight, the
/// `v^{2·poids[s]}` term, and the stored-mu z-loop.  Same-row shortcut reads
/// come from `cur`; all other reads go through `ctx` (completed rows / pools).
pub(crate) fn compute_h_uneq(
    w: ElmIdx,
    y: ElmIdx,
    first_desc: usize,
    ctx: &KlCtx<'_>,
    cur: &[PolSlot],
) -> Laurent {
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
            return same_row_value(cur, iy);
        }
        return ctx
            .pol_at(iw, iy)
            .cloned()
            .expect("inverse-symmetry entry must be comparable");
    }

    // 3. Case I.  PyCox ≈10293–10302.
    if let Some(s) = find_desc_asc(elms, y, w, rank) {
        let sw = elms.lft(w, s);
        let sy = elms.lft(y, s);
        if ctx.weights[s] == 0 {
            // poids[s] == 0: h = P̃_{sy,sw} (shorter row) if comparable, else 0.
            if sy <= sw && ctx.leq(sy, sw) {
                return ctx
                    .pol_at(sw, sy)
                    .cloned()
                    .expect("Case I weight-0 base must be comparable");
            }
            return Laurent::zero();
        }
        // poids[s] > 0: h = P̃_{sy,w} (same row, sy > y).
        return same_row_value(cur, sy);
    }

    // 4. Case II: search on the inverses.  PyCox ≈10304–10317.
    if let Some(s) = find_desc_asc(elms, iy, iw, rank) {
        let sw = elms.lft(iw, s); // inverse index
        let sy = elms.lft(iy, s); // inverse index, > iy
        if ctx.weights[s] == 0 {
            if sy <= sw && ctx.leq(sy, sw) {
                return ctx
                    .pol_at(sw, sy)
                    .cloned()
                    .expect("Case II weight-0 base must be comparable");
            }
            return Laurent::zero();
        }
        let idx = elms.inva[sy as usize]; // length l(y)+1 ⇒ > y, same row
        return same_row_value(cur, idx);
    }

    // 5. Full recursion.  PyCox ≈10318–10337.
    //    s = first left descent of w; then (uneq) replace by any left descent
    //    of minimal weight: the first t (in scan order) with poids[t] minimal.
    let mut s = first_desc;
    for t in 0..rank {
        if elms.lft(w, t) < w && ctx.weights[t] < ctx.weights[s] {
            s = t;
        }
    }
    let sw = elms.lft(w, s);
    let sy = elms.lft(y, s);

    // h = P̃_{sy,sw}  (shorter row, comparable in this branch).
    let mut h = ctx
        .pol_at(sw, sy)
        .cloned()
        .expect("recursion base P̃_{sy,sw} must be comparable");

    if ctx.weights[s] == 0 {
        // poids[s] == 0: no v-term and no z-sum — just the base.
        return h;
    }

    // + v^{2·poids[s]} · P̃_{y,sw}  if y ≤ sw and y ≤_B sw.
    if y <= sw && ctx.leq(y, sw) {
        if let Some(p) = ctx.pol_at(sw, y) {
            let shift = 2 * ctx.weights[s] as i32;
            h += &p.shifted(shift);
        }
    }

    // − Σ_{z = sw−1 down to y}  μ^s_{z,sw} · v^{L(w)−L(z)} · P̃_{y,z}
    //   over z with lft(z,s) < z and y ≤_B z and z ≤_B sw, reading the stored
    //   mu pool of the completed row sw.
    let lw_w = ctx.lweights[w as usize] as i32;
    for z in (y..sw).rev() {
        if elms.lft(z, s) >= z {
            continue;
        }
        if !ctx.leq(y, z) || !ctx.leq(z, sw) {
            continue;
        }
        let Some(m) = mu_stored(ctx, s, z, sw) else {
            continue;
        };
        if m.is_zero() {
            continue;
        }
        let Some(pyz) = ctx.pol_at(z, y) else {
            continue;
        };
        let shift = lw_w - ctx.lweights[z as usize] as i32;
        // h -= m · v^shift · P̃_{y,z}  (general Laurent mu ⇒ full product).
        let term = &m.shifted(shift) * pyz;
        h -= &term;
    }

    h
}

/// Find the first generator `s` (in `0..rank`) such that `lft(y, s) > y` and
/// `lft(w, s) < w`.  Returns `Some(s)` on the first match, `None` if none.
///
/// Used by both Case I (with the original pair `(y, w)`) and Case II (with the
/// inverse pair `(iy, iw)`).
#[inline]
fn find_desc_asc(elms: &ElementTable, y: ElmIdx, w: ElmIdx, rank: usize) -> Option<usize> {
    (0..rank).find(|&s| elms.lft(y, s) > y && elms.lft(w, s) < w)
}

// ---------------------------------------------------------------------------
// mu value (Phase C), unequal parameters
// ---------------------------------------------------------------------------

/// Compute the explicit Laurent mu value `μ^s_{y,w}` for a present slot,
/// **unequal parameters**.  Faithful to PyCox ≈10349–10365.
///
/// - `h` is `P̃_{y,w}` (this row, already computed).
/// - `row_view` is this row's flat mu buffer, indexed as `row_view[z*rank+s]`.
///   The `poids[s] ≥ 2` z-loop reads `pos_part` of `μ^s_{z,w}` from it for
///   `z > y` only — those columns are already filled by the descending-`y`
///   loop, and they live at flat indices *above* the current column's
///   `y*rank` block, so the whole buffer is passed (the current and lower
///   columns are never read here, since `z > y` strictly).
/// - All other reads go through `ctx` (completed rows / pools); the
///   `aw0`-symmetry branch reads `mues[s]` of the completed row `aw0[y]`.
pub(crate) fn compute_mu_uneq(
    w: ElmIdx,
    y: ElmIdx,
    s: usize,
    h: &Laurent,
    ctx: &KlCtx<'_>,
    row_view: &[Option<Laurent>],
    rank: usize,
) -> Laurent {
    let elms = ctx.elms;
    let lw = &elms.lengths;
    let lw_y = ctx.lweights[y as usize] as i32;
    let lw_w = ctx.lweights[w as usize] as i32;
    let poids = ctx.weights[s];

    if lw[y as usize] + lw[w as usize] > ctx.n_pos {
        // aw0 symmetry: m = ±μ^s_{aw0[y], aw0[w]}, sign MINUS iff
        // (l(w) − l(y)) even.  Row aw0[y] is strictly shorter, slot present.
        let ay = elms.aw0[y as usize];
        let aw = elms.aw0[w as usize];
        // The aw0 slot must be present; a missing value is a PyCox invariant
        // violation, not a legitimate zero-fallback.
        let base = mu_stored(ctx, s, aw, ay)
            .cloned()
            .expect("aw0-symmetry mu slot must be present (PyCox invariant)");
        // Invariant: l(w) >= l(y) because (w, y) is a comparable pair in
        // canonical order (y < w), so l(w) >= l(y) always holds.
        debug_assert!(
            lw[w as usize] >= lw[y as usize],
            "len_diff underflow: l(w)={lw_w} < l(y)={lw_y} for comparable pair (w={w}, y={y})"
        );
        let len_diff = lw[w as usize] - lw[y as usize];
        if len_diff % 2 == 0 {
            -&base
        } else {
            base
        }
    } else if poids == 1 {
        // m = zeropart(v^{1 + L(y) − L(w)} · h), as a constant.
        let shift = 1i32 + lw_y - lw_w;
        let c = h.coeff(-shift);
        Laurent::monomial(c, 0)
    } else {
        // poids[s] ≥ 2.
        // m = nonnegpart(v^{poids[s] + L(y) − L(w)} · h).
        let shift = poids as i32 + lw_y - lw_w;
        let mut m = h.shifted(shift).nonneg_part();
        // − Σ_{z = w−1 down to y+1}  nonnegpart( pospart(μ^s_{z,w}) ·
        //     v^{L(y) − L(z)} · P̃_{y,z} )  over z with lft(z,s) < z,
        //   mat[z][y] comparable and mat[w][z] comparable.
        //
        // PyCox reads `mues[s][int(mat[w][z].split('c')[s+2])]` for the CURRENT
        // row w at column z (already computed, z > y) — `row_view[z*rank+s]`.
        // A present (Some) mu value there exists exactly when (w, z) is
        // comparable and the geometric mu condition for s holds, so the
        // `mat[w][z][0]=='c'` guard is subsumed by `Some(_)`; we still need the
        // shorter-row guard `mat[z][y][0]=='c'` (`ctx.leq(y, z)`).
        for z in ((y + 1)..w).rev() {
            if elms.lft(z, s) >= z {
                continue;
            }
            if !ctx.leq(y, z) {
                continue;
            }
            let Some(Some(mu_zw)) = row_view.get(z as usize * rank + s) else {
                continue;
            };
            let mp = mu_zw.pos_part();
            if mp.is_zero() {
                continue;
            }
            let Some(pyz) = ctx.pol_at(z, y) else {
                continue;
            };
            let sh = lw_y - ctx.lweights[z as usize] as i32;
            // mp · v^sh · P̃_{y,z}
            let prod = &mp.shifted(sh) * pyz;
            m -= &prod.nonneg_part();
        }
        // Symmetrise: m = barpart(m) + m − zeropart(m).
        if !m.is_zero() {
            let bar = m.bar();
            let c = m.zero_part();
            m = &(&bar + &m) - &Laurent::monomial(c, 0);
        }
        m
    }
}

// ---------------------------------------------------------------------------
// Tests (unequal-parameter unit tests, moved from compute.rs)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::{
        group::CoxeterGroup,
        kl::{
            compute::klpolynomials_seq,
            table::{KlTable, MuMode, NO_MU},
            KlOpts,
        },
        laurent::Laurent,
    };

    /// Build a KlTable for the named group using the given weights.
    fn table_for_weights(spec: &str, weights: Vec<u32>) -> KlTable {
        let g = CoxeterGroup::from_type(spec).unwrap();
        let opts = KlOpts {
            weights,
            threads: None,
            layer_chunk: None,
        };
        klpolynomials_seq(&g, &opts).unwrap()
    }

    /// B2 with weights [2, 1]: Stored mode; pol pool has exactly the three
    /// canonical polynomials {1, 1−v², 1+v²}.  Negative coefficients are
    /// correct here (unequal parameters).
    #[test]
    fn b2_w2_1_pol_pool() {
        let t = table_for_weights("B2", vec![2, 1]);
        assert_eq!(t.mu_mode, MuMode::Stored, "uneq ⇒ Stored mu mode");

        let one = Laurent::one();
        let one_minus_v2 = Laurent::from_coeffs(0, vec![1, 0, -1]);
        let one_plus_v2 = Laurent::from_coeffs(0, vec![1, 0, 1]);

        assert_eq!(t.pols.len(), 3, "B2[2,1] pol pool size == 3");
        assert!(t.pols.contains(&one), "pool contains 1");
        assert!(t.pols.contains(&one_minus_v2), "pool contains 1−v²");
        assert!(t.pols.contains(&one_plus_v2), "pool contains 1+v²");
    }

    /// B2 with weights [2, 1]: the mu pool for generator 0 contains the
    /// non-constant value `v⁻¹ + v` (PyCox `mpols[0] = [0, v**(-1)+v]`).  This
    /// is the hand-checked negative/non-constant mu fact for the golden test.
    #[test]
    fn b2_w2_1_mu_pool() {
        let t = table_for_weights("B2", vec![2, 1]);
        // mues[0] must contain v⁻¹ + v (and zero); mues[1] only zero.
        let vinv_plus_v = Laurent::from_coeffs(-1, vec![1, 0, 1]);
        assert!(
            t.mues[0].contains(&vinv_plus_v),
            "mues[0] must contain v⁻¹+v, got {:?}",
            t.mues[0]
        );
        assert!(
            t.mues[0].contains(&Laurent::zero()),
            "mues[0] must contain zero"
        );
        // mues[1]: generator 1 has weight 1; for B2[2,1] its only mu is zero.
        assert!(
            t.mues[1].iter().all(|m| m.is_zero()),
            "mues[1] should be all zero for B2[2,1], got {:?}",
            t.mues[1]
        );
    }

    /// B2 with weights [0, 1]: weight-0 generator.  PyCox accepts this and the
    /// pol pool then contains the zero polynomial (e.g. P̃_{e,[0]} = 0).  No mu
    /// slots exist for the weight-0 generator 0.
    #[test]
    fn b2_w0_1_zero_pol_and_no_mu_for_weight0() {
        let t = table_for_weights("B2", vec![0, 1]);
        assert_eq!(t.mu_mode, MuMode::Stored);
        // The zero polynomial is in the pool (P̃ can vanish at weight 0).
        assert!(
            t.pols.contains(&Laurent::zero()),
            "weight-0 pol pool must contain zero, got {:?}",
            t.pols
        );
        // No present mu slot exists for the weight-0 generator s=0.
        let n = t.elms.len();
        for w in 0..n {
            let row = &t.rows[w];
            let mu = row.mu.as_ref().expect("Stored row has mu");
            let rank = t.rank();
            // generator s = 0 has weight 0 ⇒ slot index `y * rank + 0`.
            for y in 0..w {
                let id = mu[y * rank];
                assert_eq!(
                    id, NO_MU,
                    "weight-0 generator slot must be NO_MU at (w={w}, y={y})"
                );
            }
        }
    }

    /// Cross-check: `table.mu(s, y, w)` resolves a known negative-sign /
    /// non-constant mu.  For B2[2,1], μ^0 of the slot carrying `v⁻¹+v` must be
    /// recoverable via the public accessor.  We locate it by scanning for the
    /// unique present non-zero μ^0 value and assert it equals `v⁻¹+v`.
    #[test]
    fn b2_w2_1_mu_accessor() {
        let t = table_for_weights("B2", vec![2, 1]);
        let vinv_plus_v = Laurent::from_coeffs(-1, vec![1, 0, 1]);
        let n = t.elms.len();
        let mut found = false;
        for w in 0..n {
            for y in 0..w {
                if !t.bruhat_leq(y as u32, w as u32) {
                    continue;
                }
                let m = t.mu(0, y as u32, w as u32);
                if m == vinv_plus_v {
                    found = true;
                }
            }
        }
        assert!(found, "expected some μ^0_{{y,w}} == v⁻¹+v in B2[2,1]");
    }
}
