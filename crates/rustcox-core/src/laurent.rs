//! Laurent polynomials in `v` over ℤ.
//!
//! Invariant: `coeffs` is empty (zero polynomial) **or** both `coeffs[0]` and
//! `coeffs[last]` are nonzero (no leading / trailing zero coefficients).
//! `val` is the exponent of `coeffs[0]`; for the zero polynomial `val == 0`.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Type
// ---------------------------------------------------------------------------

/// A Laurent polynomial in `v` with integer coefficients.
///
/// The polynomial is `Σ coeffs[i] · v^(val + i)` for `i` in `0..coeffs.len()`.
///
/// **Invariant**: `coeffs` is empty (zero) or both its first and last entries
/// are nonzero.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Laurent {
    val: i32,
    coeffs: Vec<i64>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Strip leading and trailing zeros from `coeffs`, adjusting `val` accordingly.
/// Returns `(val, coeffs)` satisfying the invariant.
fn normalize(mut val: i32, mut coeffs: Vec<i64>) -> (i32, Vec<i64>) {
    // strip trailing zeros
    while coeffs.last() == Some(&0) {
        coeffs.pop();
    }
    // count and drain leading zeros in one pass, advancing val
    let leading = coeffs.iter().take_while(|&&c| c == 0).count();
    if leading > 0 {
        coeffs.drain(..leading);
        val += leading as i32;
    }
    if coeffs.is_empty() {
        return (0, vec![]);
    }
    (val, coeffs)
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl Laurent {
    /// The zero polynomial.
    pub fn zero() -> Self {
        Self {
            val: 0,
            coeffs: vec![],
        }
    }

    /// The multiplicative identity `1`.
    pub fn one() -> Self {
        Self {
            val: 0,
            coeffs: vec![1],
        }
    }

    /// The monomial `c · v^exp`.
    pub fn monomial(c: i64, exp: i32) -> Self {
        if c == 0 {
            return Self::zero();
        }
        Self {
            val: exp,
            coeffs: vec![c],
        }
    }

    /// Construct from a lowest-exponent `val` and coefficient vector.
    ///
    /// Normalizes: strips leading/trailing zeros and adjusts `val`; an
    /// all-zero vector produces the zero polynomial.
    pub fn from_coeffs(val: i32, coeffs: Vec<i64>) -> Self {
        let (v, c) = normalize(val, coeffs);
        Self { val: v, coeffs: c }
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

impl Laurent {
    /// `true` iff this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// The lowest exponent (exponent of `coeffs[0]`).
    ///
    /// For the zero polynomial this returns `0` (consistent with PyCox).
    pub fn val(&self) -> i32 {
        self.val
    }

    /// The highest exponent, or `None` for the zero polynomial.
    pub fn degree(&self) -> Option<i32> {
        if self.is_zero() {
            None
        } else {
            Some(self.val + self.coeffs.len() as i32 - 1)
        }
    }

    /// Coefficient of the highest-degree term; `0` for the zero polynomial.
    pub fn leading_coeff(&self) -> i64 {
        self.coeffs.last().copied().unwrap_or(0)
    }

    /// The coefficient slice in ascending-exponent order, starting at [`val`].
    ///
    /// Together with [`val`](Self::val) this is the full lossless content of the
    /// polynomial: `Laurent::from_coeffs(p.val(), p.coeffs().to_vec()) == *p`.
    /// Empty for the zero polynomial.  Used by the relkl block-log binary
    /// serializer (Task Q4) to persist pool entries.
    pub fn coeffs(&self) -> &[i64] {
        &self.coeffs
    }

    /// Coefficient of `v^exp`; `0` if `exp` is out of range.
    pub fn coeff(&self, exp: i32) -> i64 {
        if self.is_zero() {
            return 0;
        }
        let idx = exp - self.val;
        if idx < 0 || idx as usize >= self.coeffs.len() {
            0
        } else {
            self.coeffs[idx as usize]
        }
    }
}

// ---------------------------------------------------------------------------
// Arithmetic helpers
// ---------------------------------------------------------------------------

impl Laurent {
    /// Return `v^d · self`.
    pub fn shifted(&self, d: i32) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        Self {
            val: self.val + d,
            coeffs: self.coeffs.clone(),
        }
    }

    /// Return `(self · v^d) · k` in a single allocation.
    ///
    /// Equivalent to `self.shifted(d).scaled(k)` but avoids the intermediate
    /// allocation.  `k == 0` or a zero polynomial returns the zero polynomial;
    /// `k == ±1` skips the per-coefficient multiply.
    pub fn shift_scaled(&self, d: i32, k: i64) -> Self {
        if k == 0 || self.is_zero() {
            return Self::zero();
        }
        let new_val = self.val + d;
        if k == 1 {
            return Self {
                val: new_val,
                coeffs: self.coeffs.clone(),
            };
        }
        if k == -1 {
            return Self {
                val: new_val,
                coeffs: self.coeffs.iter().map(|&c| -c).collect(),
            };
        }
        Self {
            val: new_val,
            coeffs: self.coeffs.iter().map(|&c| c * k).collect(),
        }
    }

    /// Return `k · self`.
    pub fn scaled(&self, k: i64) -> Self {
        if k == 0 || self.is_zero() {
            return Self::zero();
        }
        Self {
            val: self.val,
            coeffs: self.coeffs.iter().map(|&c| c * k).collect(),
        }
    }

    /// Return the part consisting of terms with exponent **> 0**.
    pub fn pos_part(&self) -> Self {
        self.part_above(1)
    }

    /// Return the part consisting of terms with exponent **≥ 0**.
    pub fn nonneg_part(&self) -> Self {
        self.part_above(0)
    }

    /// The coefficient of `v^0`.
    pub fn zero_part(&self) -> i64 {
        self.coeff(0)
    }

    /// The image of `self` under `v ↦ v⁻¹` (i.e. reverse the coefficient
    /// array and negate `val` → `−degree`).
    pub fn bar(&self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let mut coeffs = self.coeffs.clone();
        coeffs.reverse();
        let new_val = -(self.val + self.coeffs.len() as i32 - 1);
        Self {
            val: new_val,
            coeffs,
        }
    }

    /// Evaluate at the integer `x`.
    ///
    /// **Overflow is unchecked; intended for small |x| (KL uses x ∈ {−1, 1, 2}).**
    ///
    /// For `val ≥ 0` any `x` is accepted.  For `val < 0` only `x ∈ {−1, 1}`
    /// are allowed (assert); for those values `v^(val+i)` = `x^(val+i)` is
    /// well-defined as an integer because `x^k = ±1` for all `k`.
    pub fn eval_i64(&self, x: i64) -> i64 {
        if self.is_zero() {
            return 0;
        }
        assert!(
            self.val >= 0 || x == 1 || x == -1,
            "eval_i64: negative val requires x ∈ {{-1, 1}}, got x={x}"
        );
        if self.val >= 0 {
            // Horner's method: evaluate as a polynomial in x starting at
            // val = 0 and multiply by x^val afterwards.
            let mut acc: i64 = 0;
            for &c in self.coeffs.iter().rev() {
                acc = acc * x + c;
            }
            // multiply by x^val
            let xval = x.pow(self.val as u32);
            acc * xval
        } else {
            // val < 0, x ∈ {-1, 1}: compute Σ c_i · x^(val+i)
            // Since x^k = x^(k mod 2) for x = ±1:
            self.coeffs
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    let exp = self.val + i as i32; // may be negative
                    let xpow = if x == 1 {
                        1i64
                    } else {
                        // x = -1: (-1)^exp = 1 if exp even, -1 if exp odd
                        if exp % 2 == 0 {
                            1
                        } else {
                            -1
                        }
                    };
                    c * xpow
                })
                .sum()
        }
    }

    // Internal: terms with exponent ≥ `low`.
    fn part_above(&self, low: i32) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let start_idx = (low - self.val).max(0) as usize;
        if start_idx >= self.coeffs.len() {
            return Self::zero();
        }
        Self::from_coeffs(
            self.val + start_idx as i32,
            self.coeffs[start_idx..].to_vec(),
        )
    }
}

// ---------------------------------------------------------------------------
// Operator overloads (on references, returning owned values)
// ---------------------------------------------------------------------------

impl std::ops::Neg for &Laurent {
    type Output = Laurent;
    fn neg(self) -> Laurent {
        if self.is_zero() {
            return Laurent::zero();
        }
        Laurent {
            val: self.val,
            coeffs: self.coeffs.iter().map(|&c| -c).collect(),
        }
    }
}

impl std::ops::Add for &Laurent {
    type Output = Laurent;
    fn add(self, rhs: &Laurent) -> Laurent {
        if self.is_zero() {
            return rhs.clone();
        }
        if rhs.is_zero() {
            return self.clone();
        }
        let lo = self.val.min(rhs.val);
        let hi_self = self.val + self.coeffs.len() as i32 - 1;
        let hi_rhs = rhs.val + rhs.coeffs.len() as i32 - 1;
        let hi = hi_self.max(hi_rhs);
        let len = (hi - lo + 1) as usize;
        let mut result = vec![0i64; len];
        let off_l = (self.val - lo) as usize;
        for (i, &c) in self.coeffs.iter().enumerate() {
            result[off_l + i] += c;
        }
        let off_r = (rhs.val - lo) as usize;
        for (i, &c) in rhs.coeffs.iter().enumerate() {
            result[off_r + i] += c;
        }
        Laurent::from_coeffs(lo, result)
    }
}

impl std::ops::Sub for &Laurent {
    type Output = Laurent;
    fn sub(self, rhs: &Laurent) -> Laurent {
        if self.is_zero() {
            return -rhs;
        }
        if rhs.is_zero() {
            return self.clone();
        }
        let lo = self.val.min(rhs.val);
        let hi_self = self.val + self.coeffs.len() as i32 - 1;
        let hi_rhs = rhs.val + rhs.coeffs.len() as i32 - 1;
        let hi = hi_self.max(hi_rhs);
        let len = (hi - lo + 1) as usize;
        let mut result = vec![0i64; len];
        let off_l = (self.val - lo) as usize;
        for (i, &c) in self.coeffs.iter().enumerate() {
            result[off_l + i] += c;
        }
        let off_r = (rhs.val - lo) as usize;
        for (i, &c) in rhs.coeffs.iter().enumerate() {
            result[off_r + i] -= c;
        }
        Laurent::from_coeffs(lo, result)
    }
}

impl std::ops::AddAssign<&Laurent> for Laurent {
    fn add_assign(&mut self, rhs: &Laurent) {
        if rhs.is_zero() {
            return;
        }
        if self.is_zero() {
            *self = rhs.clone();
            return;
        }
        let rhs_end = rhs.val + rhs.coeffs.len() as i32 - 1;
        let self_end = self.val + self.coeffs.len() as i32 - 1;
        if rhs.val >= self.val && rhs_end <= self_end {
            // rhs range fits entirely within self — mutate in place
            let off = (rhs.val - self.val) as usize;
            for (i, &c) in rhs.coeffs.iter().enumerate() {
                self.coeffs[off + i] += c;
            }
            // re-normalize: strip zero ends
            let (v, c) = normalize(self.val, std::mem::take(&mut self.coeffs));
            self.val = v;
            self.coeffs = c;
        } else {
            *self = &*self + rhs;
        }
    }
}

impl std::ops::SubAssign<&Laurent> for Laurent {
    fn sub_assign(&mut self, rhs: &Laurent) {
        if rhs.is_zero() {
            return;
        }
        if self.is_zero() {
            *self = -rhs;
            return;
        }
        let rhs_end = rhs.val + rhs.coeffs.len() as i32 - 1;
        let self_end = self.val + self.coeffs.len() as i32 - 1;
        if rhs.val >= self.val && rhs_end <= self_end {
            // rhs range fits entirely within self — mutate in place
            let off = (rhs.val - self.val) as usize;
            for (i, &c) in rhs.coeffs.iter().enumerate() {
                self.coeffs[off + i] -= c;
            }
            // re-normalize: strip zero ends
            let (v, c) = normalize(self.val, std::mem::take(&mut self.coeffs));
            self.val = v;
            self.coeffs = c;
        } else {
            *self = &*self - rhs;
        }
    }
}

impl std::ops::Mul for &Laurent {
    type Output = Laurent;
    fn mul(self, rhs: &Laurent) -> Laurent {
        if self.is_zero() || rhs.is_zero() {
            return Laurent::zero();
        }
        let new_val = self.val + rhs.val;
        let new_len = self.coeffs.len() + rhs.coeffs.len() - 1;
        let mut result = vec![0i64; new_len];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in rhs.coeffs.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        Laurent::from_coeffs(new_val, result)
    }
}

// ---------------------------------------------------------------------------
// Serde
// ---------------------------------------------------------------------------

/// Wire format: `{"v": <val>, "c": [<coefficients>]}`.
#[derive(Serialize, Deserialize)]
struct LaurentWire {
    v: i32,
    c: Vec<i64>,
}

impl Serialize for Laurent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        LaurentWire {
            v: self.val,
            c: self.coeffs.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Laurent {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = LaurentWire::deserialize(deserializer)?;
        Ok(Laurent::from_coeffs(wire.v, wire.c))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- exact tests from the task spec ---

    #[test]
    fn arithmetic_and_normalization() {
        let p = Laurent::from_coeffs(-2, vec![-1, 6, -12, 8]); // -v⁻² + 6v⁻¹ - 12 + 8v
        assert_eq!(p.val(), -2);
        assert_eq!(p.degree(), Some(1));
        assert!(Laurent::zero().is_zero());
        assert_eq!(Laurent::from_coeffs(0, vec![0, 0]), Laurent::zero()); // strips to zero
        let q = Laurent::monomial(1, 2); // v²
        assert_eq!(&q + &Laurent::one(), Laurent::from_coeffs(0, vec![1, 0, 1]));
        assert_eq!(&q - &q, Laurent::zero());
        // (1+v²)·(1−v²) = 1−v⁴
        let a = Laurent::from_coeffs(0, vec![1, 0, 1]);
        let b = Laurent::from_coeffs(0, vec![1, 0, -1]);
        assert_eq!(&a * &b, Laurent::from_coeffs(0, vec![1, 0, 0, 0, -1]));
        // cancellation inside add: (1 + v) + (-v) = 1
        let c = Laurent::from_coeffs(0, vec![1, 1]);
        assert_eq!(&c + &Laurent::monomial(-1, 1), Laurent::one());
    }

    #[test]
    fn parts_and_bar() {
        // f = v⁻¹ + 2 + 3v
        let f = Laurent::from_coeffs(-1, vec![1, 2, 3]);
        assert_eq!(f.pos_part(), Laurent::monomial(3, 1));
        assert_eq!(f.nonneg_part(), Laurent::from_coeffs(0, vec![2, 3]));
        assert_eq!(f.zero_part(), 2);
        assert_eq!(f.bar(), Laurent::from_coeffs(-1, vec![3, 2, 1]));
        assert_eq!(Laurent::zero().zero_part(), 0);
    }

    #[test]
    fn shift_scaled_single_pass() {
        // from_coeffs(0,[1,0,1]).shift_scaled(2,-3) == from_coeffs(2,[-3,0,-3])
        let p = Laurent::from_coeffs(0, vec![1, 0, 1]);
        assert_eq!(
            p.shift_scaled(2, -3),
            Laurent::from_coeffs(2, vec![-3, 0, -3])
        );

        // shift_scaled(_,0) is zero
        assert_eq!(p.shift_scaled(5, 0), Laurent::zero());
        assert_eq!(Laurent::zero().shift_scaled(2, -3), Laurent::zero());

        // k==1 and k==-1 trivial paths
        assert_eq!(p.shift_scaled(1, 1), Laurent::from_coeffs(1, vec![1, 0, 1]));
        assert_eq!(
            p.shift_scaled(1, -1),
            Laurent::from_coeffs(1, vec![-1, 0, -1])
        );

        // consistency: shift_scaled == shifted().scaled()
        let q = Laurent::from_coeffs(-2, vec![3, 1, 4, 1, 5]);
        assert_eq!(q.shift_scaled(3, 7), q.shifted(3).scaled(7));
    }

    #[test]
    fn shift_and_eval() {
        let p = Laurent::from_coeffs(0, vec![1, 0, 1]); // 1+v²
        assert_eq!(p.shifted(-3), Laurent::from_coeffs(-3, vec![1, 0, 1]));
        assert_eq!(p.eval_i64(2), 5);
        assert_eq!(p.scaled(-2), Laurent::from_coeffs(0, vec![-2, 0, -2]));
        // negative val: -v⁻² + 6v⁻¹ - 12 + 8v  at x = -1
        // = -(-1)⁻² + 6(-1)⁻¹ - 12 + 8(-1) = -1 - 6 - 12 - 8 = -27
        assert_eq!(
            Laurent::from_coeffs(-2, vec![-1, 6, -12, 8]).eval_i64(-1),
            -27
        );
    }

    #[test]
    fn add_assign_and_sub_assign() {
        // accumulate three polys in-place; result should equal fold with `+`
        let a = Laurent::from_coeffs(0, vec![1, 2, 3]);
        let b = Laurent::from_coeffs(1, vec![10, 20]);
        let c = Laurent::from_coeffs(0, vec![100]);
        let expected = &(&a + &b) + &c;

        let mut acc = Laurent::zero();
        acc += &a;
        acc += &b;
        acc += &c;
        assert_eq!(acc, expected);

        // in-place with cancellation: (1+v²) - (1+v²) == 0
        let p = Laurent::from_coeffs(0, vec![1, 0, 1]);
        let mut q = p.clone();
        q -= &p;
        assert_eq!(q, Laurent::zero());

        // in-place subtraction producing zero for a monomial subset
        let mut r = Laurent::from_coeffs(0, vec![3, 5, 7]);
        r -= &Laurent::from_coeffs(0, vec![3, 5, 7]);
        assert_eq!(r, Laurent::zero());

        // fallback path: rhs range extends beyond self
        let mut s = Laurent::from_coeffs(1, vec![1, 1]); // v + v²
        let extra = Laurent::from_coeffs(0, vec![5, 5, 5]); // 5 + 5v + 5v²
        s += &extra;
        assert_eq!(s, Laurent::from_coeffs(0, vec![5, 6, 6]));
    }

    #[test]
    fn serde_round_trip() {
        let poly = Laurent::from_coeffs(0, vec![1, 0, -1]);
        let v = serde_json::to_value(&poly).unwrap();
        assert_eq!(v, serde_json::json!({"v": 0, "c": [1, 0, -1]}));

        let zero_v = serde_json::to_value(Laurent::zero()).unwrap();
        assert_eq!(zero_v, serde_json::json!({"v": 0, "c": []}));

        // deserialize back
        let back: Laurent = serde_json::from_value(v).unwrap();
        assert_eq!(back, poly);

        let zero_back: Laurent = serde_json::from_value(zero_v).unwrap();
        assert_eq!(zero_back, Laurent::zero());
    }

    // --- proptest: ring axioms ---

    proptest::proptest! {
        #[test]
        fn ring_axioms(
            a_val in -6i32..6,
            a_coeffs in proptest::collection::vec(-9i64..9, 0..=5usize),
            b_val in -6i32..6,
            b_coeffs in proptest::collection::vec(-9i64..9, 0..=5usize),
            c_val in -6i32..6,
            c_coeffs in proptest::collection::vec(-9i64..9, 0..=5usize),
        ) {
            let a = Laurent::from_coeffs(a_val, a_coeffs);
            let b = Laurent::from_coeffs(b_val, b_coeffs);
            let c = Laurent::from_coeffs(c_val, c_coeffs);

            // add commutativity
            proptest::prop_assert_eq!(&a + &b, &b + &a);
            // mul commutativity
            proptest::prop_assert_eq!(&a * &b, &b * &a);
            // distributivity: (a+b)·c == a·c + b·c
            proptest::prop_assert_eq!(&(&a + &b) * &c, &(&a * &c) + &(&b * &c));
            // bar is an involution
            proptest::prop_assert_eq!(a.bar().bar(), a);
        }
    }
}
