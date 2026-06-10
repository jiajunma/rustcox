//! Ring coefficient types for root systems.
//!
//! Root coordinate vectors use one of three coefficient rings:
//!
//! - ℤ (`i64`) for crystallographic types and dihedral I₂(m) with m ∈ {3,4,6};
//! - ℤ\[φ\] ([`GoldenInt`]) for types H₃, H₄, I₂(5), where φ = (1+√5)/2 is the
//!   golden ratio satisfying φ² = φ + 1;
//! - ℤ\[ζ\]/(Φ_m(ζ)) ([`CycInt`]) for dihedral I₂(m) with m ∉ {3,4,5,6}, where
//!   ζ = ζ_m = e^{2πi/m} and Φ_m is the m-th cyclotomic polynomial.

mod cyc;
pub use cyc::CycInt;

/// The golden ratio φ = (1+√5)/2 ≈ 1.6180339887498948…
const PHI: f64 = 1.618_033_988_749_895;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Coefficient ring for root coordinate vectors.
///
/// Implementors must support exact arithmetic and exact sign determination.
pub trait RootCoeff: Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug {
    fn zero() -> Self;
    fn from_int(n: i64) -> Self;
    fn add(&self, o: &Self) -> Self;
    fn sub(&self, o: &Self) -> Self;
    fn mul(&self, o: &Self) -> Self;
    fn neg(&self) -> Self;
    fn is_zero(&self) -> bool;
    /// Exact (no rounding) non-negativity test: returns `true` iff `self >= 0`.
    fn is_nonneg(&self) -> bool;
    /// Floating-point approximation used for the (height, rev-lex) root sort.
    fn approx(&self) -> f64;
}

// ---------------------------------------------------------------------------
// i64 impl
// ---------------------------------------------------------------------------

impl RootCoeff for i64 {
    fn zero() -> Self {
        0
    }
    fn from_int(n: i64) -> Self {
        n
    }
    fn add(&self, o: &Self) -> Self {
        self + o
    }
    fn sub(&self, o: &Self) -> Self {
        self - o
    }
    fn mul(&self, o: &Self) -> Self {
        self * o
    }
    fn neg(&self) -> Self {
        -self
    }
    fn is_zero(&self) -> bool {
        *self == 0
    }
    fn is_nonneg(&self) -> bool {
        *self >= 0
    }
    fn approx(&self) -> f64 {
        *self as f64
    }
}

// ---------------------------------------------------------------------------
// GoldenInt — ℤ[φ]
// ---------------------------------------------------------------------------

/// An element a + bφ of ℤ\[φ\], where φ = (1+√5)/2.
///
/// Arithmetic rule: φ² = φ + 1, so
///   (a + bφ)(c + dφ) = ac + (ad+bc)φ + bd·φ²
///                     = ac + (ad+bc)φ + bd(φ+1)
///                     = (ac+bd) + (ad+bc+bd)φ.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct GoldenInt {
    pub a: i64,
    pub b: i64,
}

impl GoldenInt {
    pub fn new(a: i64, b: i64) -> Self {
        Self { a, b }
    }
}

impl RootCoeff for GoldenInt {
    fn zero() -> Self {
        Self::new(0, 0)
    }

    fn from_int(n: i64) -> Self {
        Self::new(n, 0)
    }

    fn add(&self, o: &Self) -> Self {
        Self::new(
            self.a.checked_add(o.a).expect("GoldenInt add/sub overflow"),
            self.b.checked_add(o.b).expect("GoldenInt add/sub overflow"),
        )
    }

    fn sub(&self, o: &Self) -> Self {
        Self::new(
            self.a.checked_sub(o.a).expect("GoldenInt add/sub overflow"),
            self.b.checked_sub(o.b).expect("GoldenInt add/sub overflow"),
        )
    }

    /// Multiply two ℤ\[φ\] elements using i128 intermediates to avoid overflow.
    fn mul(&self, o: &Self) -> Self {
        let a = self.a as i128;
        let b = self.b as i128;
        let c = o.a as i128;
        let d = o.b as i128;
        let new_a = a * c + b * d;
        let new_b = a * d + b * c + b * d;
        // Values stay tiny in root-system construction; panic on overflow is fine.
        Self::new(
            i64::try_from(new_a).expect("GoldenInt::mul overflow on a component"),
            i64::try_from(new_b).expect("GoldenInt::mul overflow on b component"),
        )
    }

    fn neg(&self) -> Self {
        Self::new(-self.a, -self.b)
    }

    fn is_zero(&self) -> bool {
        self.a == 0 && self.b == 0
    }

    /// Exact non-negativity test using integer arithmetic.
    ///
    /// Write `v = a + bφ = (2a + b + b√5) / 2`, so sign equals sign of
    /// `x + y√5` where `x = 2a + b`, `y = b`.
    ///
    /// Cases:
    /// - x ≥ 0, y ≥ 0 → v ≥ 0 → true
    /// - x < 0, y < 0 → v < 0 → false
    /// - x ≥ 0, y < 0 → v ≥ 0 iff x ≥ |y|√5 iff x² ≥ 5y²
    /// - x < 0, y ≥ 0 → v ≥ 0 iff |x| ≤ y√5 iff 5y² ≥ x²
    /// - zero: a = 0, b = 0 → true
    fn is_nonneg(&self) -> bool {
        let x: i128 = 2 * (self.a as i128) + self.b as i128;
        let y: i128 = self.b as i128;
        match (x >= 0, y >= 0) {
            (true, true) => true,
            (false, false) => false,
            (true, false) => {
                // y < 0, x ≥ 0: nonneg iff x² ≥ 5y²
                x * x >= 5 * y * y
            }
            (false, true) => {
                // x < 0, y ≥ 0: nonneg iff 5y² ≥ x²
                5 * y * y >= x * x
            }
        }
    }

    fn approx(&self) -> f64 {
        self.a as f64 + self.b as f64 * PHI
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // i64 basics
    // -----------------------------------------------------------------------

    #[test]
    fn i64_zero_and_from_int() {
        assert_eq!(<i64 as RootCoeff>::zero(), 0);
        assert_eq!(<i64 as RootCoeff>::from_int(42), 42);
        assert_eq!(<i64 as RootCoeff>::from_int(-7), -7);
    }

    #[test]
    fn i64_arithmetic() {
        let a: i64 = 3;
        let b: i64 = -5;
        assert_eq!(a.add(&b), -2);
        assert_eq!(a.sub(&b), 8);
        assert_eq!(a.mul(&b), -15);
        assert_eq!(a.neg(), -3);
    }

    #[test]
    fn i64_is_zero_and_is_nonneg() {
        assert!(<i64 as RootCoeff>::zero().is_zero());
        assert!(!5_i64.is_zero());
        assert!(5_i64.is_nonneg());
        assert!(0_i64.is_nonneg());
        assert!(!(-1_i64).is_nonneg());
    }

    #[test]
    fn i64_approx() {
        assert_eq!(7_i64.approx(), 7.0_f64);
        assert_eq!((-3_i64).approx(), -3.0_f64);
    }

    // -----------------------------------------------------------------------
    // GoldenInt: from the task spec
    // -----------------------------------------------------------------------

    #[test]
    fn golden_int_arithmetic() {
        let phi = GoldenInt::new(0, 1); // φ
                                        // φ² = φ + 1
        assert_eq!(phi.mul(&phi), GoldenInt::new(1, 1));
        let x = GoldenInt::new(-1, 1); // φ − 1 = 1/φ ≈ 0.618 > 0
        assert!(x.is_nonneg());
        assert!(!x.neg().is_nonneg());
        assert!(GoldenInt::new(2, -1).is_nonneg()); // 2 − φ ≈ 0.382
        assert!(!GoldenInt::new(1, -1).is_nonneg()); // 1 − φ ≈ −0.618 < 0
        assert!(GoldenInt::new(0, 0).is_nonneg());
        assert!((GoldenInt::new(-1, 1).approx() - 0.618_033_988_749_895).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // GoldenInt: add / sub / from_int / is_zero
    // -----------------------------------------------------------------------

    #[test]
    fn golden_int_add_sub() {
        let a = GoldenInt::new(3, 2);
        let b = GoldenInt::new(1, -1);
        assert_eq!(a.add(&b), GoldenInt::new(4, 1));
        assert_eq!(a.sub(&b), GoldenInt::new(2, 3));
    }

    #[test]
    fn golden_int_from_int() {
        assert_eq!(GoldenInt::from_int(5), GoldenInt::new(5, 0));
        assert_eq!(GoldenInt::from_int(-3), GoldenInt::new(-3, 0));
    }

    #[test]
    fn golden_int_is_zero() {
        assert!(GoldenInt::zero().is_zero());
        assert!(!GoldenInt::new(1, 0).is_zero());
        assert!(!GoldenInt::new(0, 1).is_zero());
    }

    #[test]
    fn golden_int_mul_general() {
        assert_eq!(
            GoldenInt::new(2, 3).mul(&GoldenInt::new(1, -1)),
            GoldenInt::new(-1, -2)
        );
    }

    #[test]
    fn golden_int_neg() {
        let x = GoldenInt::new(3, -2);
        assert_eq!(x.neg(), GoldenInt::new(-3, 2));
        assert_eq!(x.neg().neg(), x);
    }

    // -----------------------------------------------------------------------
    // GoldenInt: boundary sign cases (Fibonacci) — verified by hand
    //
    // For v = a + bφ let x = 2a+b, y = b.
    // (−1597, 987): x = −3194+987 = −2207, y = 987
    //   x<0, y≥0: nonneg iff 5y² ≥ x²
    //   5·987² = 5·974169 = 4870845
    //   2207² = 4870849
    //   4870845 < 4870849 → false  (987φ − 1597 < 0)
    //
    // (−2584, 1597): x = −5168+1597 = −3571, y = 1597
    //   x<0, y≥0: nonneg iff 5y² ≥ x²
    //   5·1597² = 5·2550409 = 12752045
    //   3571² = 12751241
    //   12752045 ≥ 12751241 → true  (1597φ − 2584 > 0)
    // -----------------------------------------------------------------------

    #[test]
    fn golden_int_fibonacci_sign_negative() {
        // 987φ − 1597: tiny negative, f64 would give ≈ −0.000637
        assert!(!GoldenInt::new(-1597, 987).is_nonneg());
    }

    #[test]
    fn golden_int_fibonacci_sign_positive() {
        // 1597φ − 2584: tiny positive, f64 would give ≈ +0.000394
        assert!(GoldenInt::new(-2584, 1597).is_nonneg());
    }

    // Additional boundary: mixed sign x≥0, y<0
    // v = 3 − 2φ: x = 6−2 = 4, y = −2; x≥0, y<0: nonneg iff x²≥5y² → 16≥20 false
    #[test]
    fn golden_int_mixed_positive_x_negative_y() {
        assert!(!GoldenInt::new(3, -2).is_nonneg()); // 3 − 2φ ≈ 3 − 3.236 < 0
                                                     // v = 4 − φ: x = 8−1=7, y = −1; 49 ≥ 5 → true
        assert!(GoldenInt::new(4, -1).is_nonneg()); // 4 − φ ≈ 2.382 > 0
    }

    // -----------------------------------------------------------------------
    // Proptest: is_nonneg() agrees with f64 sign when |approx| > 1e-3
    // -----------------------------------------------------------------------

    proptest::proptest! {
        #[test]
        fn golden_int_is_nonneg_matches_approx(
            a in -10_000_i64..10_000,
            b in -10_000_i64..10_000,
        ) {
            let g = GoldenInt::new(a, b);
            let approx = g.approx();
            if approx.abs() > 1e-3 {
                assert_eq!(
                    g.is_nonneg(),
                    approx >= 0.0,
                    "is_nonneg mismatch for GoldenInt({a}, {b}): approx={approx}",
                );
            }
        }
    }
}
