//! Root system generation for finite Coxeter groups.
//!
//! Implements the BFS closure algorithm from PyCox (`roots1`/`roots` in
//! `pycox_ref.py`, lines 2758–2788) and the permutation construction
//! (`permroots`, lines 2779–2788).
//!
//! # Ordering convention
//!
//! Positive roots are sorted by height ascending; ties broken by coordinate
//! vector lex **descending** (matching PyCox: `l.sort(reverse=True)` then
//! stable `l.sort(key=sum)`).
//!
//! Roots are indexed 0..2N: positive 0..N, negative N..2N with
//! `roots[N+i] = −roots[i]`.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::cartan::CartanMat;
use crate::element::Perm;
use crate::ring::RootCoeff;

// Maximum positive roots before we panic (sanity cap for non-finite types).
const MAX_POS_ROOTS: usize = 10_000;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A root system for a finite Coxeter group.
pub struct RootSystem {
    /// Number of positive roots N.
    pub n_pos: u32,
    /// All 2N coordinate vectors, for `CartanMat::Int` only.
    /// Each inner vector has length `rank`.
    pub roots_int: Option<Vec<Vec<i64>>>,
    /// Generator permutations.  `permgens[s].0[i]` is the index of s(roots[i]).
    /// Length is `rank`; each `Perm` has length 2N.
    pub permgens: Vec<Perm>,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Build the root system for the given Cartan matrix.
///
/// Dispatches on the variant of `cmat`:
/// - `CartanMat::Int` → uses `i64` arithmetic, stores `roots_int`.
/// - `CartanMat::Golden` → uses `GoldenInt` arithmetic, `roots_int` is `None`.
/// - `CartanMat::Cyc` → uses `CycInt` arithmetic, `roots_int` is `None`.
pub fn build(cmat: &CartanMat) -> RootSystem {
    match cmat {
        CartanMat::Int(mat) => {
            let (roots, permgens) = build_generic(mat);
            let n_pos = roots.len() as u32 / 2;
            let roots_int = Some(roots);
            RootSystem {
                n_pos,
                roots_int,
                permgens,
            }
        }
        // Golden (ℤ[φ]) and Cyc (ℤ[ζ]) both use exact non-integer arithmetic
        // and store no integer root coordinates.
        CartanMat::Golden(mat) => build_no_int_roots(mat),
        CartanMat::Cyc(mat) => build_no_int_roots(mat),
    }
}

/// Build a root system over a non-integer coefficient ring, discarding the
/// (non-`i64`) root coordinates and keeping only `n_pos` + permgens.
fn build_no_int_roots<R: RootCoeff>(mat: &[Vec<R>]) -> RootSystem {
    let (roots, permgens) = build_generic(mat);
    let n_pos = roots.len() as u32 / 2;
    RootSystem {
        n_pos,
        roots_int: None,
        permgens,
    }
}

// ---------------------------------------------------------------------------
// Private generic implementation
// ---------------------------------------------------------------------------

/// Reflect coordinate vector `root` by generator `s` using the Cartan matrix.
///
/// Formula: `r'[s] = r[s] − Σ_t cmat[s][t]·r[t]`  (only where cmat[s][t] ≠ 0).
fn reflect<R: RootCoeff>(root: &[R], s: usize, cmat: &[Vec<R>]) -> Vec<R> {
    let mut result = root.to_vec();
    // Compute the sum Σ_t cmat[s][t]·r[t]
    let mut sum = R::zero();
    for t in 0..cmat[s].len() {
        if !cmat[s][t].is_zero() {
            sum = sum.add(&cmat[s][t].mul(&root[t]));
        }
    }
    result[s] = root[s].sub(&sum);
    result
}

/// Compare two `RootCoeff` values exactly.
///
/// Uses `a.sub(b).is_nonneg()` to avoid floating-point errors.
fn cmp_coeff<R: RootCoeff>(a: &R, b: &R) -> Ordering {
    if a == b {
        Ordering::Equal
    } else if a.sub(b).is_nonneg() {
        Ordering::Greater
    } else {
        Ordering::Less
    }
}

/// Compare two root vectors by height, then lex descending.
///
/// Height = sum of all coordinates.  Comparison of heights uses exact
/// arithmetic via `cmp_coeff`.  Tie-break: lex descending (coordinate 0 first).
fn cmp_root_sort_key<R: RootCoeff>(a: &[R], b: &[R]) -> Ordering {
    // Compare heights (sums)
    let ha: R = a.iter().skip(1).fold(a[0].clone(), |acc, x| acc.add(x));
    let hb: R = b.iter().skip(1).fold(b[0].clone(), |acc, x| acc.add(x));
    let height_ord = cmp_coeff(&ha, &hb);
    if height_ord != Ordering::Equal {
        return height_ord;
    }
    // Tie-break: lex descending
    for (ca, cb) in a.iter().zip(b.iter()) {
        let c = cmp_coeff(ca, cb);
        if c != Ordering::Equal {
            return c.reverse(); // descending
        }
    }
    Ordering::Equal
}

/// The generic BFS root-system builder.
///
/// Returns `(all_roots, permgens)` where `all_roots` has length `2N` and
/// `permgens` has length `rank`.
fn build_generic<R: RootCoeff>(cmat: &[Vec<R>]) -> (Vec<Vec<R>>, Vec<Perm>) {
    let rank = cmat.len();

    // --- Step 1: BFS closure to collect all positive roots ---
    //
    // Start with the rank simple-root vectors (identity rows).
    let mut pos_roots: Vec<Vec<R>> = (0..rank)
        .map(|s| {
            let mut v = vec![R::zero(); rank];
            v[s] = R::from_int(1);
            v
        })
        .collect();

    // Use a set for fast membership testing (O(1) duplicate check).
    let mut pos_set: HashMap<Vec<R>, ()> = pos_roots.iter().map(|r| (r.clone(), ())).collect();

    let mut i = 0;
    while i < pos_roots.len() {
        for s in 0..rank {
            let nr = reflect(&pos_roots[i], s, cmat);
            // The reflected root is positive iff its s-coordinate is non-negative.
            if nr[s].is_nonneg() && !pos_set.contains_key(&nr) {
                pos_set.insert(nr.clone(), ());
                pos_roots.push(nr);
                if pos_roots.len() > MAX_POS_ROOTS {
                    panic!(
                        "BFS exceeded {MAX_POS_ROOTS} positive roots — \
                         input Cartan matrix is not of finite type"
                    );
                }
            }
        }
        i += 1;
    }

    // --- Step 2: Sort positive roots ---
    //
    // PyCox: `l.sort(reverse=True)` (lex desc) then stable `l.sort(key=sum)`
    // (height asc).  Equivalent to a single sort by (height asc, lex desc).
    pos_roots.sort_unstable_by(|a, b| cmp_root_sort_key(a, b));

    let n_pos = pos_roots.len();

    // --- Step 3: Append negatives ---
    let mut all_roots: Vec<Vec<R>> = Vec::with_capacity(2 * n_pos);
    for r in &pos_roots {
        all_roots.push(r.clone());
    }
    for r in &pos_roots {
        all_roots.push(r.iter().map(|c| c.neg()).collect());
    }

    // --- Step 4: Build index map and permgens ---
    let root_to_idx: HashMap<Vec<R>, usize> = all_roots
        .iter()
        .enumerate()
        .map(|(i, r)| (r.clone(), i))
        .collect();

    let mut permgens: Vec<Perm> = Vec::with_capacity(rank);
    for s in 0..rank {
        let perm: Box<[u32]> = all_roots
            .iter()
            .enumerate()
            .map(|(i, root)| {
                let image = reflect(root, s, cmat);
                *root_to_idx.get(&image).unwrap_or_else(|| {
                    panic!(
                        "reflection of root not found in root system \
                             (generator s={s}, root index i={i})"
                    )
                }) as u32
            })
            .collect();
        permgens.push(Perm(perm));
    }

    (all_roots, permgens)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartan::{cartan_mat, Series};

    #[test]
    fn a2_root_system() {
        let cm = cartan_mat(Series::A, 2).unwrap();
        let rs = build(&cm);
        assert_eq!(rs.n_pos, 3);
        let roots = rs.roots_int.as_ref().unwrap();
        assert_eq!(roots[..3], [vec![1, 0], vec![0, 1], vec![1, 1]]);
        assert_eq!(roots[3..], [vec![-1, 0], vec![0, -1], vec![-1, -1]]);
        // s0: α0→−α0 (idx 3), α1→α0+α1 (idx 2), α0+α1→α1 (idx 1), and negatives mirrored
        assert_eq!(&*rs.permgens[0].0, &[3, 2, 1, 0, 5, 4]);
    }

    #[test]
    fn a1_root_system() {
        let cm = cartan_mat(Series::A, 1).unwrap();
        let rs = build(&cm);
        assert_eq!(rs.n_pos, 1);
        let roots = rs.roots_int.as_ref().unwrap();
        assert_eq!(roots[0], vec![1]);
        assert_eq!(roots[1], vec![-1]);
        // s0: α0→−α0
        assert_eq!(&*rs.permgens[0].0, &[1, 0]);
    }

    #[test]
    fn g2_root_system_n_pos() {
        let cm = cartan_mat(Series::G, 2).unwrap();
        let rs = build(&cm);
        assert_eq!(rs.n_pos, 6);
    }

    #[test]
    fn h3_root_system_n_pos() {
        let cm = cartan_mat(Series::H, 3).unwrap();
        let rs = build(&cm);
        assert_eq!(rs.n_pos, 15);
        assert!(rs.roots_int.is_none());
    }

    #[test]
    fn i5_root_system_n_pos() {
        let cm = cartan_mat(Series::I(5), 2).unwrap();
        let rs = build(&cm);
        assert_eq!(rs.n_pos, 5);
        assert!(rs.roots_int.is_none());
    }

    #[test]
    fn i7_cyclotomic_root_system() {
        let cm = cartan_mat(Series::I(7), 2).unwrap();
        let rs = build(&cm);
        assert_eq!(rs.n_pos, 7);
        assert!(rs.roots_int.is_none());
        // Each generator is an involution on the 14 roots.
        for s in 0..2 {
            let p = &rs.permgens[s].0;
            assert_eq!(p.len(), 14);
            for i in 0..14 {
                assert_eq!(p[p[i] as usize] as usize, i, "permgens[{s}] not involution");
            }
        }
    }

    #[test]
    fn i8_cyclotomic_root_system() {
        let cm = cartan_mat(Series::I(8), 2).unwrap();
        let rs = build(&cm);
        assert_eq!(rs.n_pos, 8);
        assert!(rs.roots_int.is_none());
    }

    #[test]
    fn i12_cyclotomic_root_system_n_pos() {
        // Even m with a longer cyclotomic entry exercises reduction in the BFS.
        let cm = cartan_mat(Series::I(12), 2).unwrap();
        let rs = build(&cm);
        assert_eq!(rs.n_pos, 12);
    }

    /// Routing decision (m = 5): the production path keeps I₂(5) on `GoldenInt`.
    /// This test documents *why* — building I₂(5) through the cyclotomic ring
    /// (the odd-m rule gives `z1 = ζ_5^2 + ζ_5^3 = −φ`) yields **byte-identical
    /// permgens**, so either ring is safe; we keep `GoldenInt` because it is the
    /// exact representation already covered by the golden files.
    #[test]
    fn i5_golden_and_cyc_permgens_agree() {
        use crate::cartan::CartanMat;
        use crate::ring::CycInt;

        let golden = cartan_mat(Series::I(5), 2).unwrap();
        let rs_g = build(&golden);

        // Cyclotomic I5 (odd-m rule, m=5): z1 = ζ^2 + ζ^3.
        let two = CycInt::from_int(2);
        let z1 = CycInt::new(5, vec![0, 0, 1, 1]);
        let cyc = CartanMat::Cyc(vec![vec![two.clone(), z1.clone()], vec![z1, two]]);
        let rs_c = build(&cyc);

        assert_eq!(rs_g.n_pos, rs_c.n_pos, "I5 n_pos differs across rings");
        let pg: Vec<&[u32]> = rs_g.permgens.iter().map(|p| &*p.0).collect();
        let pc: Vec<&[u32]> = rs_c.permgens.iter().map(|p| &*p.0).collect();
        assert_eq!(pg, pc, "I5 permgens differ between Golden and Cyc rings");
    }
}
