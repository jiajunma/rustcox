//! Dihedral / cyclotomic helpers for `cartan.rs`.
//!
//! These functions handle the I₂(m) cases that require [`CycInt`] arithmetic:
//! [`cartan_i_cyc`], [`coxeter_from_cyc`], and the supporting [`bezout_coeff1`].

use crate::ring::{CycInt, RootCoeff};

// ---------------------------------------------------------------------------
// bezout_coeff1
// ---------------------------------------------------------------------------

/// First Bézout coefficient `s` in `gcd(a, b) = s·a + t·b`, matching PyCox's
/// `gcdex(a, b)['coeff1']` (extended Euclid, GAP-style sign conventions).
pub(crate) fn bezout_coeff1(a: i64, b: i64) -> i64 {
    // f = |a|, fm = sign(a); g = |b|, gm = 0   (PyCox initialisation)
    let (mut f, mut fm) = if a >= 0 { (a, 1_i64) } else { (-a, -1_i64) };
    let (mut g, mut gm) = if b >= 0 { (b, 0_i64) } else { (-b, 0_i64) };
    while g != 0 {
        let q = f.div_euclid(g);
        let (h, hm) = (g, gm);
        g = f - q * g;
        gm = fm - q * gm;
        f = h;
        fm = hm;
    }
    fm
}

// ---------------------------------------------------------------------------
// cartan_i_cyc
// ---------------------------------------------------------------------------

/// Build the cyclotomic I₂(m) Cartan matrix for m ∉ {3,4,5,6}.
///
/// See `cartan_i` (in `cartan.rs`) for the even/odd construction rules.
pub(crate) fn cartan_i_cyc(m: u32) -> Vec<Vec<CycInt>> {
    let two = CycInt::from_int(2);
    let neg_one = CycInt::from_int(-1);
    // ζ_m^k as a CycInt (negative k handled via ζ^{-1} = ζ^{m-1}).
    let zeta_pow = |k: i64| -> CycInt {
        let e = k.rem_euclid(m as i64) as u32;
        let mut coeffs = vec![0_i64; e as usize + 1];
        coeffs[e as usize] = 1;
        CycInt::new(m, coeffs)
    };

    if m % 2 == 0 {
        // \[\[2, -1\], \[-2 - ir(m/2), 2\]\], ir(m/2) = ζ_m + ζ_m^{-1}.
        let ir = zeta_pow(1).add(&zeta_pow(-1));
        let c10 = CycInt::from_int(-2).sub(&ir);
        vec![vec![two.clone(), neg_one], vec![c10, two]]
    } else {
        // d = gcdex(2+m, 2m)\['coeff1'\]; z1 = ζ^d + ζ^{-d} (negate if d even).
        let d = bezout_coeff1(2 + m as i64, 2 * m as i64);
        let base = zeta_pow(d).add(&zeta_pow(-d));
        let z1 = if d % 2 == 0 { base.neg() } else { base };
        vec![vec![two.clone(), z1.clone()], vec![z1, two]]
    }
}

// ---------------------------------------------------------------------------
// coxeter_from_cyc
// ---------------------------------------------------------------------------

/// Coxeter matrix for a cyclotomic (`CartanMat::Cyc`) Cartan matrix.
///
/// The `Cyc` variant is produced solely by `cartan_i` for the dihedral group
/// I₂(m) with m ∉ {3,4,5,6}, so it is always a 2×2 matrix whose single edge has
/// Coxeter order m. The order m equals the cyclotomic field order carried by any
/// off-diagonal entry (`CycInt::order`); diagonal `2`s are sentinel constants.
pub(crate) fn coxeter_from_cyc(mat: &[Vec<CycInt>]) -> Vec<Vec<u32>> {
    let rows = mat.len();
    let cols = mat.first().map(|r| r.len()).unwrap_or(0);
    assert_eq!(
        rows, 2,
        "Cyc Cartan matrices are dihedral (2×2); got {rows}×{cols}"
    );
    // Recover m from whichever off-diagonal entry carries a real field order.
    let m = mat[0][1].order().max(mat[1][0].order());
    assert!(
        m >= 3,
        "Cyc dihedral Coxeter order must be ≥ 3, got {m} (entries carry no field order)"
    );
    vec![vec![1, m], vec![m, 1]]
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartan::{cartan_mat, coxeter_mat_from_cartan, Series};

    // -- I2(m) cyclotomic Cartan matrices (transcribed from PyCox) ----------

    /// Dense (low-degree-first) coefficient vectors of the two off-diagonal
    /// entries of the I₂(m) cyclotomic Cartan matrix, for cross-checking
    /// against PyCox `cartanmat("I?", 2)`.
    fn cyc_off_diags(m: u32) -> (Vec<i64>, Vec<i64>) {
        use crate::cartan::CartanMat;
        match cartan_mat(Series::I(m), 2).unwrap() {
            CartanMat::Cyc(mat) => (mat[0][1].coeffs().to_vec(), mat[1][0].coeffs().to_vec()),
            other => panic!("I{m} should be Cyc, got {other:?}"),
        }
    }

    #[test]
    fn i7_cyc_matches_pycox() {
        // PyCox I7: symmetric, c01 = c10 = ζ^3 + ζ^4 → coeffs [0,0,0,1,1].
        let (c01, c10) = cyc_off_diags(7);
        assert_eq!(c01, vec![0, 0, 0, 1, 1]);
        assert_eq!(c10, vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn i8_cyc_matches_pycox() {
        // PyCox I8: asymmetric, c01 = -1, c10 = -2 - ζ + ζ^3 → coeffs [-2,-1,0,1].
        let (c01, c10) = cyc_off_diags(8);
        assert_eq!(c01, vec![-1]); // bare integer -1
        assert_eq!(c10, vec![-2, -1, 0, 1]);
    }

    #[test]
    fn i9_i12_cyc_match_pycox() {
        // Odd m=9: ζ^4 + ζ^5 → [0,0,0,0,1,1].
        let (c01, _) = cyc_off_diags(9);
        assert_eq!(c01, vec![0, 0, 0, 0, 1, 1]);
        // Even m=12: c10 = -2 - 2ζ + ζ^3 → [-2,-2,0,1]; c01 = -1.
        let (c01_12, c10_12) = cyc_off_diags(12);
        assert_eq!(c01_12, vec![-1]);
        assert_eq!(c10_12, vec![-2, -2, 0, 1]);
    }

    #[test]
    fn i7_i8_coxeter_mats() {
        let cox7 = coxeter_mat_from_cartan(&cartan_mat(Series::I(7), 2).unwrap());
        assert_eq!(cox7, vec![vec![1, 7], vec![7, 1]]);
        let cox8 = coxeter_mat_from_cartan(&cartan_mat(Series::I(8), 2).unwrap());
        assert_eq!(cox8, vec![vec![1, 8], vec![8, 1]]);
    }

    #[test]
    fn bezout_coeff1_matches_pycox() {
        // PyCox gcdex(2+m, 2m)['coeff1'] values for odd m (validated against
        // the reference): m=7→-3, m=9→5, m=11→-5, m=13→7, m=15→-7.
        assert_eq!(bezout_coeff1(9, 14), -3);
        assert_eq!(bezout_coeff1(11, 18), 5);
        assert_eq!(bezout_coeff1(13, 22), -5);
        assert_eq!(bezout_coeff1(15, 26), 7);
        assert_eq!(bezout_coeff1(17, 30), -7);
        // gcdex doc example: 1 = 4*4 + (-1)*15 ⇒ coeff1(4,15) = 4.
        assert_eq!(bezout_coeff1(4, 15), 4);
    }

    // -- coxeter_from_cyc directly -------------------------------------------

    #[test]
    fn coxeter_from_cyc_i7() {
        let mat = cartan_i_cyc(7);
        let cox = coxeter_from_cyc(&mat);
        assert_eq!(cox, vec![vec![1, 7], vec![7, 1]]);
    }

    #[test]
    fn coxeter_from_cyc_i8() {
        let mat = cartan_i_cyc(8);
        let cox = coxeter_from_cyc(&mat);
        assert_eq!(cox, vec![vec![1, 8], vec![8, 1]]);
    }
}
