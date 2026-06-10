//! Kazhdan–Lusztig polynomials and cells.
//!
//! This module is the entry point for the KL machinery.
//! - `table`: storage layer (polynomials, mu values, rows).
//! - Options and validation live here.

pub mod table;
pub use table::{KlRow, KlTable, MuMode};

use crate::group::CoxeterGroup;

// ---------------------------------------------------------------------------
// KlOpts
// ---------------------------------------------------------------------------

/// Options controlling KL computation.
#[derive(Clone, Debug)]
pub struct KlOpts {
    /// Generator weights `L(s)`.  Must have `len() == rank`.
    pub weights: Vec<u32>,
    /// Number of threads for parallel computation.  `None` = use Rayon default.
    pub threads: Option<usize>,
    /// Layer chunk size for parallel computation.  `None` = automatic.
    pub layer_chunk: Option<usize>,
}

impl KlOpts {
    /// Construct equal-parameter options: `weights = vec![1; rank]`.
    pub fn equal(rank: usize) -> Self {
        KlOpts {
            weights: vec![1; rank],
            threads: None,
            layer_chunk: None,
        }
    }

    /// Validate options against a group.
    ///
    /// Checks:
    /// 1. `weights.len() == rank`.
    /// 2. Generators that are conjugate (i.e. connected by a path of edges
    ///    with odd Coxeter label `m`) must have equal weights.  Conjugacy is
    ///    propagated transitively via union-find over all `(s, t)` pairs with
    ///    `coxmat[s][t]` odd (≥ 3 and odd, since `m=2` means no bond and `m=1`
    ///    is the diagonal).
    pub fn validate(&self, group: &CoxeterGroup) -> Result<(), KlError> {
        let rank = group.rank;

        if self.weights.len() != rank {
            return Err(KlError::WeightsLen(self.weights.len(), rank));
        }

        // Build conjugacy classes via union-find.
        // Two generators s, t are conjugate if coxmat[s][t] is odd and ≥ 3.
        let mut parent: Vec<usize> = (0..rank).collect();

        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]]; // path compression
                x = parent[x];
            }
            x
        }

        fn union(parent: &mut [usize], x: usize, y: usize) {
            let rx = find(parent, x);
            let ry = find(parent, y);
            if rx != ry {
                parent[rx] = ry;
            }
        }

        for s in 0..rank {
            for t in (s + 1)..rank {
                let m = group.coxmat[s][t];
                // Odd m ≥ 3 means s and t are conjugate in W.
                if m >= 3 && m % 2 == 1 {
                    union(&mut parent, s, t);
                }
            }
        }

        // Check that all generators in the same class have equal weights.
        for s in 0..rank {
            for t in (s + 1)..rank {
                let rs = find(&mut parent, s);
                let rt = find(&mut parent, t);
                if rs == rt && self.weights[s] != self.weights[t] {
                    return Err(KlError::ConjugateWeights(s, t));
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// KlError
// ---------------------------------------------------------------------------

/// Errors from KL option validation.
#[derive(Debug, thiserror::Error)]
pub enum KlError {
    #[error("weights length {0} != rank {1}")]
    WeightsLen(usize, usize),
    #[error("generators {0} and {1} are conjugate (odd m) but have different weights")]
    ConjugateWeights(usize, usize),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::CoxeterGroup;

    // -----------------------------------------------------------------------
    // Test: klopts_validate
    // -----------------------------------------------------------------------

    /// A3 (generators 0-1-2 all connected by m=3 edges, i.e. all conjugate).
    /// Equal weights [1,1,1] → OK.
    /// [1,2,1] → generators 0 and 1 are conjugate with different weights → Err.
    /// [1,1,2] → generators 1 and 2 are conjugate with different weights → Err.
    #[test]
    fn klopts_validate_a3() {
        let group = CoxeterGroup::from_type("A3").unwrap();

        // All-equal weights OK
        let ok = KlOpts {
            weights: vec![1, 1, 1],
            threads: None,
            layer_chunk: None,
        };
        assert!(ok.validate(&group).is_ok(), "A3 [1,1,1] should be OK");

        // [1,2,1]: generators 0 and 1 connected by m=3 (odd) → different weights
        let bad1 = KlOpts {
            weights: vec![1, 2, 1],
            threads: None,
            layer_chunk: None,
        };
        assert!(
            matches!(bad1.validate(&group), Err(KlError::ConjugateWeights(..))),
            "A3 [1,2,1] should fail ConjugateWeights"
        );

        // [1,1,2]: generators 1 and 2 connected by m=3 → different weights
        let bad2 = KlOpts {
            weights: vec![1, 1, 2],
            threads: None,
            layer_chunk: None,
        };
        assert!(
            matches!(bad2.validate(&group), Err(KlError::ConjugateWeights(..))),
            "A3 [1,1,2] should fail ConjugateWeights"
        );
    }

    /// B2: coxmat[0][1] = 4 (even) → generators 0 and 1 are NOT conjugate.
    /// So [2, 1] is valid (unequal but not conjugate).
    /// Wrong length → WeightsLen.
    #[test]
    fn klopts_validate_b2() {
        let group = CoxeterGroup::from_type("B2").unwrap();

        // B2 has m=4 (even), so generators are not conjugate → [2,1] is OK
        let ok = KlOpts {
            weights: vec![2, 1],
            threads: None,
            layer_chunk: None,
        };
        assert!(ok.validate(&group).is_ok(), "B2 [2,1] should be OK");

        // Wrong length
        let bad_len = KlOpts {
            weights: vec![1],
            threads: None,
            layer_chunk: None,
        };
        assert!(
            matches!(bad_len.validate(&group), Err(KlError::WeightsLen(1, 2))),
            "B2 len=1 should fail WeightsLen(1, 2)"
        );
    }

    /// G2: coxmat[0][1] = 6 (even) → generators 0 and 1 are NOT conjugate.
    /// So [1, 3] is valid.
    #[test]
    fn klopts_validate_g2() {
        let group = CoxeterGroup::from_type("G2").unwrap();

        let ok = KlOpts {
            weights: vec![1, 3],
            threads: None,
            layer_chunk: None,
        };
        assert!(ok.validate(&group).is_ok(), "G2 [1,3] should be OK");
    }

    /// Conjugacy transitivity: A3 generators 0-1 (m=3, odd), 1-2 (m=3, odd)
    /// → all three are in the same conjugacy class transitively.
    /// weights [2,1,1]: 0-1 edge odd → 0 and 1 conjugate → weights[0]=2 ≠
    /// weights[1]=1, but also 0 is in the same class as 2 via transitivity
    /// through 1.  The error must name a conjugate pair.
    #[test]
    fn klopts_validate_a3_transitivity() {
        let group = CoxeterGroup::from_type("A3").unwrap();

        // [2,1,1]: generators 0 and 1 are directly connected by odd m=3.
        // Transitivity via generator 1 also puts 0 and 2 in the same class.
        // The validator must return Err(ConjugateWeights(s, t)) for some pair.
        let bad = KlOpts {
            weights: vec![2, 1, 1],
            threads: None,
            layer_chunk: None,
        };
        let result = bad.validate(&group);
        assert!(
            matches!(result, Err(KlError::ConjugateWeights(..))),
            "A3 [2,1,1] transitivity should fail with ConjugateWeights, got {result:?}"
        );
        // Verify the error message mentions a conjugate pair.
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("conjugate"),
            "error message should mention conjugate pair, got: {err_msg}"
        );
    }

    /// KlOpts::equal() produces all-1 weights and None options.
    #[test]
    fn klopts_equal_constructor() {
        let opts = KlOpts::equal(3);
        assert_eq!(opts.weights, vec![1u32, 1, 1]);
        assert!(opts.threads.is_none());
        assert!(opts.layer_chunk.is_none());
    }
}
