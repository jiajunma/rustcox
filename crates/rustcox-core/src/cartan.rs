//! Cartan matrices, Coxeter matrices, and degree data for all finite Coxeter
//! groups.
//!
//! Conventions follow PyCox (`pycox_ref.py`, function `cartanmat`):
//!
//! - The (i,j) entry of the Cartan matrix is 2·(eᵢ,eⱼ)/(eᵢ,eᵢ) where
//!   e₀,e₁,… are the simple roots.
//! - Diagonal entries are 2.
//! - Type B_n: node 0 is the short root end; entry c[0][1] = −2, c[1][0] = −1.
//!   Diagram: 0 ≤= 1 — 2 — … — n−1.
//! - Type C_n: node 0 is the long root end; entry c[1][0] = −2, c[0][1] = −1.
//!   Diagram: 0 => 1 — 2 — … — n−1.
//! - Type D_n (n ≥ 3): nodes 0,1 are the fork tips, node 2 is the fork centre.
//! - Type G₂: [[2,−1],[−3,2]] (node 0 short, node 1 long).
//! - Type F₄: nodes 0–3 with double bond between 1–2 (c[2][1]=−2, c[1][2]=−1).
//! - Types H₃, H₄: off-diagonal entries involving node 0 are ±φ where
//!   φ = (1+√5)/2, represented as [`GoldenInt`]`{a:0, b:1}`.
//! - Type I₂(m): integer Cartan for m∈{3,4,6}; golden (φ) for m=5; all other
//!   m use cyclotomic integers [`CycInt`] (the `Cyc` variant) following PyCox's
//!   `cartanmat` exactly (even/odd m differ — see [`cartan_i`]).

use crate::ring::{CycInt, GoldenInt, RootCoeff};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Series label for finite Coxeter groups.
///
/// `I(m)` represents the dihedral group I₂(m) of order 2m.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Series {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    /// Dihedral group I₂(m).  `rank` must be 2 when used with [`cartan_mat`].
    I(u32),
}

impl std::fmt::Display for Series {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Series::A => write!(f, "A"),
            Series::B => write!(f, "B"),
            Series::C => write!(f, "C"),
            Series::D => write!(f, "D"),
            Series::E => write!(f, "E"),
            Series::F => write!(f, "F"),
            Series::G => write!(f, "G"),
            Series::H => write!(f, "H"),
            Series::I(m) => write!(f, "I{m}"),
        }
    }
}

/// A Cartan matrix with integer, golden-ratio, or cyclotomic entries.
#[derive(Clone, Debug)]
pub enum CartanMat {
    /// All entries are integers (crystallographic types and most dihedral).
    Int(Vec<Vec<i64>>),
    /// Entries live in ℤ[φ] (types H₃, H₄, I₂(5)).
    Golden(Vec<Vec<GoldenInt>>),
    /// Entries live in ℤ[ζ_m]/(Φ_m) (dihedral I₂(m), m ∉ {3,4,5,6}).
    Cyc(Vec<Vec<CycInt>>),
}

/// Errors produced by Cartan-data functions.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    /// The series letter is not one of A–H or I.
    #[error("unknown Coxeter series '{0}'")]
    UnknownSeries(String),

    /// The rank is outside the valid range for this series.
    #[error("rank {rank} is out of range for series {series}")]
    RankOutOfRange { series: String, rank: usize },

    /// The dihedral parameter m is below the minimum of 3.
    #[error("I2(m) requires m ≥ 3, got {0}")]
    InvalidDihedralParam(u32),

    /// A parse error while reading a type string.
    #[error("failed to parse type string '{0}': {1}")]
    ParseError(String, String),
}

// ---------------------------------------------------------------------------
// cartan_mat
// ---------------------------------------------------------------------------

/// Return the Cartan matrix for the given series and rank.
///
/// # Rank constraints (matching PyCox)
///
/// | Series | Valid ranks |
/// |--------|-------------|
/// | A | n ≥ 1 |
/// | B | n ≥ 2 |
/// | C | n ≥ 2 |
/// | D | n ≥ 3 (D₃ = A₃ in PyCox; golden files start at D₄) |
/// | E | n ∈ {6, 7, 8} |
/// | F | n = 4 |
/// | G | n = 2 |
/// | H | n ∈ {3, 4} |
/// | I(m) | rank must be 2; m ≥ 3 |
pub fn cartan_mat(series: Series, rank: usize) -> Result<CartanMat, Error> {
    match series {
        Series::A => cartan_a(rank),
        Series::B => cartan_b(rank),
        Series::C => cartan_c(rank),
        Series::D => cartan_d(rank),
        Series::E => cartan_e(rank),
        Series::F => cartan_f(rank),
        Series::G => cartan_g(rank),
        Series::H => cartan_h(rank),
        Series::I(m) => cartan_i(m, rank),
    }
}

// ---------------------------------------------------------------------------
// Per-series builders
// ---------------------------------------------------------------------------

fn rank_err(series: &str, rank: usize) -> Error {
    Error::RankOutOfRange {
        series: series.to_string(),
        rank,
    }
}

/// Build an n×n integer identity-2 tridiagonal matrix (type A_n).
fn cartan_mat_a_int(n: usize) -> Vec<Vec<i64>> {
    let mut a = vec![vec![0_i64; n]; n];
    for i in 0..n {
        a[i][i] = 2;
        if i + 1 < n {
            a[i][i + 1] = -1;
            a[i + 1][i] = -1;
        }
    }
    a
}

fn cartan_a(rank: usize) -> Result<CartanMat, Error> {
    if rank < 1 {
        return Err(rank_err("A", rank));
    }
    Ok(CartanMat::Int(cartan_mat_a_int(rank)))
}

fn cartan_b(rank: usize) -> Result<CartanMat, Error> {
    // B_n, n ≥ 2. Diagram: 0 ≤= 1 — 2 — … — n−1
    // c[0][1] = −2, c[1][0] = −1 (PyCox convention: a[0][1]=-2)
    if rank < 2 {
        return Err(rank_err("B", rank));
    }
    let mut a = cartan_mat_a_int(rank);
    a[0][1] = -2;
    // a[1][0] is already -1 from A_n template
    Ok(CartanMat::Int(a))
}

fn cartan_c(rank: usize) -> Result<CartanMat, Error> {
    // C_n, n ≥ 2. Diagram: 0 => 1 — 2 — … — n−1
    // c[1][0] = −2, c[0][1] = −1 (PyCox convention: a[1][0]=-2)
    if rank < 2 {
        return Err(rank_err("C", rank));
    }
    let mut a = cartan_mat_a_int(rank);
    a[1][0] = -2;
    // a[0][1] is already -1 from A_n template
    Ok(CartanMat::Int(a))
}

fn cartan_d(rank: usize) -> Result<CartanMat, Error> {
    // D_n, n ≥ 3. Nodes 0,1 are fork tips; node 2 is the fork centre.
    // Fork structure: 0—2, 1—2, 2—3—…—n−1 chain.
    // The rank < 3 guard below already rejects rank 2 and below.
    if rank < 3 {
        return Err(rank_err("D", rank));
    }
    // Start from A_n, then rewire the first two rows/cols
    let mut a = cartan_mat_a_int(rank);
    // Remove the 0–1 edge
    a[0][1] = 0;
    a[1][0] = 0;
    // Add 0–2 and 1–2 edges
    a[0][2] = -1;
    a[2][0] = -1;
    a[1][2] = -1;
    a[2][1] = -1;
    Ok(CartanMat::Int(a))
}

fn cartan_e(rank: usize) -> Result<CartanMat, Error> {
    // E_6, E_7, E_8.  Node 1 is the branch node (attached to node 3).
    // Numbering: 0—2—3—4—5(—6—7), with 1 attached to 3.
    match rank {
        6 => Ok(CartanMat::Int(vec![
            vec![2, 0, -1, 0, 0, 0],
            vec![0, 2, 0, -1, 0, 0],
            vec![-1, 0, 2, -1, 0, 0],
            vec![0, -1, -1, 2, -1, 0],
            vec![0, 0, 0, -1, 2, -1],
            vec![0, 0, 0, 0, -1, 2],
        ])),
        7 => Ok(CartanMat::Int(vec![
            vec![2, 0, -1, 0, 0, 0, 0],
            vec![0, 2, 0, -1, 0, 0, 0],
            vec![-1, 0, 2, -1, 0, 0, 0],
            vec![0, -1, -1, 2, -1, 0, 0],
            vec![0, 0, 0, -1, 2, -1, 0],
            vec![0, 0, 0, 0, -1, 2, -1],
            vec![0, 0, 0, 0, 0, -1, 2],
        ])),
        8 => Ok(CartanMat::Int(vec![
            vec![2, 0, -1, 0, 0, 0, 0, 0],
            vec![0, 2, 0, -1, 0, 0, 0, 0],
            vec![-1, 0, 2, -1, 0, 0, 0, 0],
            vec![0, -1, -1, 2, -1, 0, 0, 0],
            vec![0, 0, 0, -1, 2, -1, 0, 0],
            vec![0, 0, 0, 0, -1, 2, -1, 0],
            vec![0, 0, 0, 0, 0, -1, 2, -1],
            vec![0, 0, 0, 0, 0, 0, -1, 2],
        ])),
        _ => Err(rank_err("E", rank)),
    }
}

fn cartan_f(rank: usize) -> Result<CartanMat, Error> {
    // F_4 only.  Diagram: 0—1 =>= 2—3 (double bond between 1 and 2).
    // c[2][1] = −2, c[1][2] = −1 (PyCox: a[2][1]=-2)
    if rank != 4 {
        return Err(rank_err("F", rank));
    }
    Ok(CartanMat::Int(vec![
        vec![2, -1, 0, 0],
        vec![-1, 2, -1, 0],
        vec![0, -2, 2, -1],
        vec![0, 0, -1, 2],
    ]))
}

fn cartan_g(rank: usize) -> Result<CartanMat, Error> {
    // G_2 only.  Diagram: 0 ->- 1 (label 6; triple bond).
    // c[0][1] = −1, c[1][0] = −3 (PyCox: [[2,−1],[−3,2]])
    if rank != 2 {
        return Err(rank_err("G", rank));
    }
    Ok(CartanMat::Int(vec![vec![2, -1], vec![-3, 2]]))
}

fn cartan_h(rank: usize) -> Result<CartanMat, Error> {
    // H_3 and H_4.  The edge between nodes 0 and 1 has bond label 5.
    // Off-diagonal entries: c[0][1] = c[1][0] = −φ where φ = GoldenInt{a:0,b:1}.
    //
    // PyCox: `ir(5) = zeta5(0,1)` = φ; entries are -ir(5) = GoldenInt{a:0,b:-1}.
    let neg_phi = GoldenInt::new(0, -1); // −φ
    let zero = GoldenInt::new(0, 0);
    let two = GoldenInt::new(2, 0);
    let neg_one = GoldenInt::new(-1, 0);
    match rank {
        3 => Ok(CartanMat::Golden(vec![
            vec![two, neg_phi, zero],
            vec![neg_phi, two, neg_one],
            vec![zero, neg_one, two],
        ])),
        4 => Ok(CartanMat::Golden(vec![
            vec![two, neg_phi, zero, zero],
            vec![neg_phi, two, neg_one, zero],
            vec![zero, neg_one, two, neg_one],
            vec![zero, zero, neg_one, two],
        ])),
        _ => Err(rank_err("H", rank)),
    }
}

/// Build the I₂(m) Cartan matrix.
///
/// This transcribes PyCox `cartanmat` for type `"I?"` exactly
/// (`pycox_ref.py` ≈2262–2290). The special small cases use ℤ or ℤ[φ]:
///
/// - m = 3 → `[[2,−1],[−1,2]]`
/// - m = 4 → `[[2,−1],[−2,2]]`
/// - m = 6 → `[[2,−1],[−3,2]]`
/// - m = 5 → `[[2,−φ],[−φ,2]]` (golden, identical ordering to the H-type edge)
///
/// All other m use cyclotomic integers ([`CartanMat::Cyc`]) in the field of
/// order m (ζ = ζ_m), and **even and odd m produce structurally different
/// matrices**, matching PyCox:
///
/// - **even m** (m ∉ {4,6}): asymmetric `[[2,−1],[−2−ir(m/2), 2]]` where
///   `ir(m/2) = ζ_m + ζ_m^{-1} = 2cos(2π/m)`;
/// - **odd m** (m ∉ {3,5}): symmetric `[[2, z1],[z1, 2]]` where
///   `d = bezout1(2+m, 2m)`, `z = ζ_m`, and
///   `z1 = z^d + z^{-d}` (or `−z^d − z^{-d}` if `d` is even).
fn cartan_i(m: u32, rank: usize) -> Result<CartanMat, Error> {
    // I_2(m), always rank 2.
    if rank != 2 {
        return Err(Error::RankOutOfRange {
            series: format!("I{m}"),
            rank,
        });
    }
    if m < 3 {
        return Err(Error::InvalidDihedralParam(m));
    }
    match m {
        3 => Ok(CartanMat::Int(vec![vec![2, -1], vec![-1, 2]])),
        4 => Ok(CartanMat::Int(vec![vec![2, -1], vec![-2, 2]])),
        6 => Ok(CartanMat::Int(vec![vec![2, -1], vec![-3, 2]])),
        5 => {
            // I_2(5): entries are ±φ (golden ratio), same as H-type edge.
            let neg_phi = GoldenInt::new(0, -1); // −φ
            let two = GoldenInt::new(2, 0);
            Ok(CartanMat::Golden(vec![
                vec![two, neg_phi],
                vec![neg_phi, two],
            ]))
        }
        _ => Ok(CartanMat::Cyc(cartan_i_cyc(m))),
    }
}

/// First Bézout coefficient `s` in `gcd(a, b) = s·a + t·b`, matching PyCox's
/// `gcdex(a, b)['coeff1']` (extended Euclid, GAP-style sign conventions).
fn bezout_coeff1(a: i64, b: i64) -> i64 {
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

/// Build the cyclotomic I₂(m) Cartan matrix for m ∉ {3,4,5,6}.
///
/// See [`cartan_i`] for the even/odd construction rules.
fn cartan_i_cyc(m: u32) -> Vec<Vec<CycInt>> {
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
        // [[2, -1], [-2 - ir(m/2), 2]], ir(m/2) = ζ_m + ζ_m^{-1}.
        let ir = zeta_pow(1).add(&zeta_pow(-1));
        let c10 = CycInt::from_int(-2).sub(&ir);
        vec![vec![two.clone(), neg_one], vec![c10, two]]
    } else {
        // d = gcdex(2+m, 2m)['coeff1']; z1 = ζ^d + ζ^{-d} (negate if d even).
        let d = bezout_coeff1(2 + m as i64, 2 * m as i64);
        let base = zeta_pow(d).add(&zeta_pow(-d));
        let z1 = if d % 2 == 0 { base.neg() } else { base };
        vec![vec![two.clone(), z1.clone()], vec![z1, two]]
    }
}

// ---------------------------------------------------------------------------
// degrees_of
// ---------------------------------------------------------------------------

/// Return the reflection degrees of the finite Coxeter group.
///
/// Degrees are given in the same order as PyCox's `degreesdata` function.
/// They are not necessarily sorted.
pub fn degrees_of(series: Series, rank: usize) -> Result<Vec<u32>, Error> {
    // Validate rank first (reuse cartan_mat for the check, but only parse it).
    match series {
        Series::A => {
            if rank < 1 {
                return Err(rank_err("A", rank));
            }
            // 2, 3, …, n+1
            Ok((2..=(rank as u32 + 1)).collect())
        }
        Series::B | Series::C => {
            if rank < 2 {
                return Err(rank_err(
                    match series {
                        Series::B => "B",
                        Series::C => "C",
                        _ => unreachable!("matched B|C arm"),
                    },
                    rank,
                ));
            }
            // 2, 4, 6, …, 2n
            Ok((1..=(rank as u32)).map(|i| 2 * i).collect())
        }
        Series::D => {
            if rank < 3 {
                return Err(rank_err("D", rank));
            }
            // 2, 4, …, 2(n−1), n
            let mut degs: Vec<u32> = (1..=(rank as u32 - 1)).map(|i| 2 * i).collect();
            degs.push(rank as u32);
            Ok(degs)
        }
        Series::E => match rank {
            6 => Ok(vec![2, 5, 6, 8, 9, 12]),
            7 => Ok(vec![2, 6, 8, 10, 12, 14, 18]),
            8 => Ok(vec![2, 8, 12, 14, 18, 20, 24, 30]),
            _ => Err(rank_err("E", rank)),
        },
        Series::F => {
            if rank != 4 {
                return Err(rank_err("F", rank));
            }
            Ok(vec![2, 6, 8, 12])
        }
        Series::G => {
            if rank != 2 {
                return Err(rank_err("G", rank));
            }
            Ok(vec![2, 6])
        }
        Series::H => match rank {
            3 => Ok(vec![2, 6, 10]),
            4 => Ok(vec![2, 12, 20, 30]),
            _ => Err(rank_err("H", rank)),
        },
        Series::I(m) => {
            if rank != 2 {
                return Err(Error::RankOutOfRange {
                    series: format!("I{m}"),
                    rank,
                });
            }
            if m < 3 {
                return Err(Error::InvalidDihedralParam(m));
            }
            // Degrees [2, m] for all I_2(m); NeedsCyc is only for cartan_mat.
            Ok(vec![2, m])
        }
    }
}

// ---------------------------------------------------------------------------
// coxeter_mat_from_cartan
// ---------------------------------------------------------------------------

/// Compute the Coxeter matrix from a Cartan matrix.
///
/// Convention:
/// - Diagonal entries are 1.
/// - For off-diagonal (s,t): the order mₛₜ is determined by cₛₜ · cₜₛ:
///   - 0 → 2
///   - 1 → 3  (A-type edge)
///   - 2 → 4  (B/C-type edge)
///   - 3 → 6  (G₂/F₄ triple edge)
///   - golden: c·c = 1 − c → 5  (H-type and I₂(5) edge)
///
/// This matches PyCox's `coxetermat` computation in the `coxeter` class.
///
/// # Panics
///
/// Panics if the matrix was not produced by [`cartan_mat`].  Specifically:
/// - Integer matrices: panics if any off-diagonal product cₛₜ·cₜₛ is outside
///   {0, 1, 2, 3}.
/// - Golden matrices: panics if a golden off-diagonal pair does not satisfy the
///   golden identity (cₛₜ·cₜₛ == 1 − cₛₜ) and is not a plain integer pair.
pub fn coxeter_mat_from_cartan(c: &CartanMat) -> Vec<Vec<u32>> {
    match c {
        CartanMat::Int(mat) => coxeter_from_int(mat),
        CartanMat::Golden(mat) => coxeter_from_golden(mat),
        CartanMat::Cyc(mat) => coxeter_from_cyc(mat),
    }
}

fn coxeter_from_int(mat: &[Vec<i64>]) -> Vec<Vec<u32>> {
    let n = mat.len();
    let mut result = vec![vec![0u32; n]; n];
    for s in 0..n {
        for t in 0..n {
            result[s][t] = if s == t {
                1
            } else {
                int_off_diag_order(mat[s][t], mat[t][s])
            };
        }
    }
    result
}

/// Map the product c_{st}·c_{ts} to the Coxeter order.
///
/// For integer Cartan matrices the product is always in {0,1,2,3}.
fn int_off_diag_order(c_st: i64, c_ts: i64) -> u32 {
    match c_st * c_ts {
        0 => 2,
        1 => 3,
        2 => 4,
        3 => 6,
        p => panic!("unexpected Cartan product {p} for entries ({c_st}, {c_ts})"),
    }
}

fn coxeter_from_golden(mat: &[Vec<GoldenInt>]) -> Vec<Vec<u32>> {
    let n = mat.len();
    let mut result = vec![vec![0u32; n]; n];
    for s in 0..n {
        for t in 0..n {
            result[s][t] = if s == t {
                1
            } else {
                golden_off_diag_order(&mat[s][t], &mat[t][s])
            };
        }
    }
    result
}

/// Determine the Coxeter order for a pair of GoldenInt Cartan entries.
///
/// For H-types and I₂(5), the off-diagonal entries on the special edge are
/// both −φ.  The product is φ² = φ+1 = GoldenInt{a:1,b:1}.
///
/// PyCox condition for order-5: `c[s][t]*c[t][s] == 1 - c[s][t]`
/// i.e.  φ² == 1 − (−φ) = 1 + φ  ✓  (golden ratio identity).
///
/// All other golden entries on non-special edges are plain integers (0 or −1).
fn golden_off_diag_order(c_st: &GoldenInt, c_ts: &GoldenInt) -> u32 {
    let zero = GoldenInt::new(0, 0);
    if c_st == &zero || c_ts == &zero {
        return 2;
    }
    // Check if both are integer (b == 0); if so use integer logic.
    if c_st.b == 0 && c_ts.b == 0 {
        return int_off_diag_order(c_st.a, c_ts.a);
    }
    // Check for the golden order-5 condition:
    // c_st * c_ts == 1 − c_st
    let product = c_st.mul(c_ts);
    let one_minus_c_st = GoldenInt::new(1 - c_st.a, -c_st.b);
    if product == one_minus_c_st {
        return 5;
    }
    panic!(
        "unexpected golden Cartan pair ({c_st:?}, {c_ts:?}): product={product:?}, 1-c_st={one_minus_c_st:?}"
    );
}

/// Coxeter matrix for a cyclotomic ([`CartanMat::Cyc`]) Cartan matrix.
///
/// The `Cyc` variant is produced solely by [`cartan_i`] for the dihedral group
/// I₂(m) with m ∉ {3,4,5,6}, so it is always a 2×2 matrix whose single edge has
/// Coxeter order m. The order m equals the cyclotomic field order carried by any
/// off-diagonal entry (`CycInt::order`); diagonal `2`s are sentinel constants.
fn coxeter_from_cyc(mat: &[Vec<CycInt>]) -> Vec<Vec<u32>> {
    assert_eq!(
        mat.len(),
        2,
        "Cyc Cartan matrices are dihedral (2×2); got {}×{}",
        mat.len(),
        mat.len()
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
// order_from_degrees
// ---------------------------------------------------------------------------

/// Return the group order as the product of the reflection degrees.
pub fn order_from_degrees(degrees: &[u32]) -> u128 {
    degrees.iter().map(|&d| d as u128).product()
}

// ---------------------------------------------------------------------------
// parse_type
// ---------------------------------------------------------------------------

/// Parse a type string like `"A5"`, `"I7"`, `"A2xA1"` into a list of
/// `(Series, rank)` pairs.
///
/// - Simple types: letter followed by a rank digit (e.g. `"B4"`, `"E6"`).
/// - Dihedral I-types: `"I7"` means I₂(7) with rank 2; the number after `I`
///   is the dihedral parameter m, not the rank.
/// - Product types: separated by `x` (lowercase, e.g. `"A2xA1"`).
///
/// Note: rank ranges (e.g. B requires rank ≥ 2, E requires rank ∈ {6,7,8}) are
/// NOT validated here; they are validated by [`cartan_mat`] and [`degrees_of`].
pub fn parse_type(s: &str) -> Result<Vec<(Series, usize)>, Error> {
    s.split('x').map(|part| parse_single(part, s)).collect()
}

fn parse_single(part: &str, full: &str) -> Result<(Series, usize), Error> {
    let part = part.trim();
    if part.is_empty() {
        return Err(Error::ParseError(
            full.to_string(),
            "empty component".to_string(),
        ));
    }

    // First character is the series letter.
    let mut chars = part.chars();
    let letter = chars.next().unwrap();
    let rest: String = chars.collect();

    // Parse the numeric suffix.
    let n: usize = rest.parse().map_err(|_| {
        Error::ParseError(
            full.to_string(),
            format!("expected integer after '{letter}', got '{rest}'"),
        )
    })?;

    let series = match letter {
        'A' => Series::A,
        'B' => Series::B,
        'C' => Series::C,
        'D' => Series::D,
        'E' => Series::E,
        'F' => Series::F,
        'G' => Series::G,
        'H' => Series::H,
        'I' => {
            // "I7" means I₂(7): m = n, rank = 2.
            let m = n as u32;
            return Ok((Series::I(m), 2));
        }
        other => {
            return Err(Error::ParseError(
                full.to_string(),
                format!("unknown series '{other}'"),
            ));
        }
    };

    Ok((series, n))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- parse_type ----------------------------------------------------------

    #[test]
    fn parse_a5() {
        assert_eq!(parse_type("A5").unwrap(), vec![(Series::A, 5)]);
    }

    #[test]
    fn parse_i7() {
        // "I7" = I₂(7), rank 2
        assert_eq!(parse_type("I7").unwrap(), vec![(Series::I(7), 2)]);
    }

    #[test]
    fn parse_a2xa1() {
        assert_eq!(
            parse_type("A2xA1").unwrap(),
            vec![(Series::A, 2), (Series::A, 1)]
        );
    }

    #[test]
    fn parse_unknown_series() {
        assert!(parse_type("Z9").is_err());
    }

    #[test]
    fn parse_empty_string_is_err() {
        assert!(parse_type("").is_err());
    }

    #[test]
    fn parse_letter_only_is_err() {
        // "A" has no numeric suffix — should fail
        assert!(parse_type("A").is_err());
    }

    #[test]
    fn parse_a0_is_ok() {
        // parse_type does NOT validate rank ranges; "A0" parses successfully.
        // The caller (cartan_mat / degrees_of) is responsible for rejecting rank 0.
        assert_eq!(parse_type("A0").unwrap(), vec![(Series::A, 0)]);
    }

    #[test]
    fn parse_a2x_is_err() {
        // Trailing "x" produces an empty component which should be rejected.
        assert!(parse_type("A2x").is_err());
    }

    // -- B2 concrete matrix (from PyCox) ------------------------------------

    #[test]
    fn b2_cartan_matrix() {
        // PyCox: [[2,-2],[-1,2]]  (c[0][1]=-2, c[1][0]=-1)
        let expected = CartanMat::Int(vec![vec![2, -2], vec![-1, 2]]);
        let got = cartan_mat(Series::B, 2).unwrap();
        assert_int_mat_eq(&got, &expected);
    }

    #[test]
    fn c3_cartan_off_diag() {
        // PyCox: [[2,-1,0],[-2,2,-1],[0,-1,2]]  (c[1][0]=-2)
        let got = cartan_mat(Series::C, 3).unwrap();
        if let CartanMat::Int(m) = got {
            assert_eq!(m[1][0], -2, "C3: c[1][0] should be -2");
            assert_eq!(m[0][1], -1, "C3: c[0][1] should be -1");
        } else {
            panic!("C3 should be Int");
        }
    }

    // -- H3 contains GoldenInt entries --------------------------------------

    #[test]
    fn h3_cartan_has_golden_entries() {
        let got = cartan_mat(Series::H, 3).unwrap();
        assert!(
            matches!(got, CartanMat::Golden(_)),
            "H3 should use Golden variant"
        );
        if let CartanMat::Golden(m) = got {
            let neg_phi = GoldenInt::new(0, -1);
            assert_eq!(m[0][1], neg_phi, "H3 c[0][1] should be -phi");
            assert_eq!(m[1][0], neg_phi, "H3 c[1][0] should be -phi");
            assert_eq!(m[1][2], GoldenInt::new(-1, 0), "H3 c[1][2] should be -1");
        }
    }

    // -- G2 Coxeter matrix --------------------------------------------------

    #[test]
    fn g2_coxeter_mat() {
        let cartan = cartan_mat(Series::G, 2).unwrap();
        let cox = coxeter_mat_from_cartan(&cartan);
        assert_eq!(cox, vec![vec![1, 6], vec![6, 1]]);
    }

    // -- I5 Coxeter matrix --------------------------------------------------

    #[test]
    fn i5_coxeter_mat() {
        let cartan = cartan_mat(Series::I(5), 2).unwrap();
        let cox = coxeter_mat_from_cartan(&cartan);
        assert_eq!(cox, vec![vec![1, 5], vec![5, 1]]);
    }

    // -- I2(m) cyclotomic Cartan matrices (transcribed from PyCox) ----------

    /// Dense (low-degree-first) coefficient vectors of the two off-diagonal
    /// entries of the I₂(m) cyclotomic Cartan matrix, for cross-checking
    /// against PyCox `cartanmat("I?", 2)`.
    fn cyc_off_diags(m: u32) -> (Vec<i64>, Vec<i64>) {
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

    // -- degrees_of ---------------------------------------------------------

    #[test]
    fn degrees_a4() {
        let mut d = degrees_of(Series::A, 4).unwrap();
        d.sort_unstable();
        assert_eq!(d, vec![2, 3, 4, 5]);
    }

    #[test]
    fn degrees_d5() {
        let mut d = degrees_of(Series::D, 5).unwrap();
        d.sort_unstable();
        assert_eq!(d, vec![2, 4, 5, 6, 8]);
    }

    #[test]
    fn degrees_i7() {
        // I2(m) degrees [2, m] work even without CycInt
        assert_eq!(degrees_of(Series::I(7), 2).unwrap(), vec![2, 7]);
    }

    // -- order_from_degrees -------------------------------------------------

    #[test]
    fn order_h3() {
        let degs = degrees_of(Series::H, 3).unwrap();
        assert_eq!(order_from_degrees(&degs), 120);
    }

    #[test]
    fn order_h4() {
        let degs = degrees_of(Series::H, 4).unwrap();
        assert_eq!(order_from_degrees(&degs), 14400);
    }

    // -- rank validation errors ---------------------------------------------

    #[test]
    fn b1_is_error() {
        assert!(matches!(
            cartan_mat(Series::B, 1),
            Err(Error::RankOutOfRange { .. })
        ));
    }

    #[test]
    fn d2_is_error() {
        assert!(matches!(
            cartan_mat(Series::D, 2),
            Err(Error::RankOutOfRange { .. })
        ));
    }

    #[test]
    fn e5_is_error() {
        assert!(matches!(
            cartan_mat(Series::E, 5),
            Err(Error::RankOutOfRange { .. })
        ));
    }

    // -- helper -------------------------------------------------------------

    fn assert_int_mat_eq(got: &CartanMat, expected: &CartanMat) {
        match (got, expected) {
            (CartanMat::Int(g), CartanMat::Int(e)) => {
                assert_eq!(g, e, "Cartan matrix mismatch");
            }
            _ => panic!("CartanMat variant mismatch: got {got:?}, expected {expected:?}"),
        }
    }
}
