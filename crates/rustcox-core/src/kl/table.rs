//! KL table storage: polynomial pool, mu pool, and per-row accessors.
//!
//! This module is purely storage + simple accessors.  No KL recursion lives
//! here; that belongs to `kl/compute.rs` (Task 9).
//!
//! Direct field mutation of [`KlRow`] by the compute pass is intentional:
//! the row is owned by the table and is only written by one writer at a time.

use crate::{element::ElmIdx, enumerate::ElementTable, laurent::Laurent};

// ---------------------------------------------------------------------------
// Sentinel constants
// ---------------------------------------------------------------------------

/// Sentinel stored in `KlRow::pol[y]` when `y` is not Bruhat-below `w`.
pub const NOT_LEQ: u32 = u32::MAX;

/// Sentinel stored in the mu flat array when no mu slot exists for `(y, s, w)`.
pub const NO_MU: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// MuMode
// ---------------------------------------------------------------------------

/// How mu values are stored/retrieved.
///
/// - `Implicit`: mu is derived on the fly from the KL polynomial via
///   `zero_part(v^shift · P_{y,w})`.  Only slot-presence flags are stored.
/// - `Stored`: mu values are interned into per-generator pools (`mues`).
#[derive(Clone, Debug, PartialEq)]
pub enum MuMode {
    Implicit,
    Stored,
}

// ---------------------------------------------------------------------------
// KlRow
// ---------------------------------------------------------------------------

/// Storage for one row `w` of the KL table.
///
/// Row `w` covers indices `y` in `0..=w` (by canonical element order).
///
/// ## pol
///
/// `pol[y]` is an index into `KlTable::pols`, or `NOT_LEQ` when `y ≰ w`.
///
/// ## mu (Stored mode)
///
/// `mu` holds a flat array of length `(w+1) * rank`.
/// `mu[y * rank + s]` is an index into `KlTable::mues[s]`, or `NO_MU`.
///
/// ## mu_present (Implicit mode)
///
/// `mu_present` holds a flat bool array of length `(w+1) * rank`.
/// `mu_present[y * rank + s]` is `true` when the mu slot is non-zero.
/// The actual value is derived from the KL polynomial at query time.
///
/// ## Direct field mutation
///
/// The compute pass writes directly into `pol`, `mu`, and `mu_present`
/// after pushing a fresh row.  This is intentional: the row is owned
/// exclusively by the table during the write phase.
#[derive(Clone, Debug, PartialEq)]
pub struct KlRow {
    /// `pol[y]` = index into `KlTable::pols`, or `NOT_LEQ`.
    pub pol: Vec<u32>,
    /// Stored mode only: flat `(w+1) * rank` array of mu indices.
    pub mu: Option<Vec<u32>>,
    /// Implicit mode only: flat `(w+1) * rank` array of presence flags.
    pub mu_present: Option<Vec<bool>>,
}

// ---------------------------------------------------------------------------
// KlTable
// ---------------------------------------------------------------------------

/// The full KL table for a finite Coxeter group.
///
/// Polynomials and mu-values are interned into pools so identical objects are
/// stored only once.  `pols[0]` is always `Laurent::one()` (the KL polynomial
/// for `P_{w,w}`).  In Stored mode, `mues[s][0]` is always `Laurent::zero()`.
#[derive(Debug, PartialEq)]
pub struct KlTable {
    /// The underlying element table (canonical ordering).
    pub elms: ElementTable,
    /// Generator weights `L(s)` for each generator `s`.
    pub weights: Vec<u32>,
    /// `lweights[i]` = `L(w_i)` = sum of weights along the canonical word.
    pub lweights: Vec<u32>,
    /// Polynomial pool.  `pols[0]` is always `Laurent::one()`.
    pub pols: Vec<Laurent>,
    /// Mu pools, one per generator.  Non-empty only in `Stored` mode.
    ///
    /// **Pool invariant (Stored mode):** index `0` is the canonical zero
    /// polynomial.  The zero polynomial must **never** be interned at index ≥ 1;
    /// Task 9 must dedup zero values to index `0`.  All other entries
    /// (index ≥ 1) are guaranteed non-zero by the dedup invariant.
    /// `mues[s][0]` is always `Laurent::zero()`.
    pub mues: Vec<Vec<Laurent>>,
    /// Whether mu values are stored explicitly or derived implicitly.
    pub mu_mode: MuMode,
    /// Per-element rows.  `rows[w]` covers `y` in `0..=w`.
    pub rows: Vec<KlRow>,
}

// ---------------------------------------------------------------------------
// impl KlTable
// ---------------------------------------------------------------------------

impl KlTable {
    /// Create an empty table with seeded pools.
    ///
    /// - `pols` is seeded with `[Laurent::one()]`.
    /// - In `Stored` mode, `mues[s]` is seeded with `[Laurent::zero()]` for
    ///   each generator `s`.  In `Implicit` mode, `mues` is empty.
    /// - `rows` is empty; rows are pushed by the compute pass.
    pub fn new_empty(elms: ElementTable, weights: Vec<u32>, mu_mode: MuMode) -> Self {
        let rank = elms.rank;
        let lweights = elms.lweights(&weights);

        let mues = if mu_mode == MuMode::Stored {
            vec![vec![Laurent::zero()]; rank]
        } else {
            vec![]
        };

        KlTable {
            elms,
            weights,
            lweights,
            pols: vec![Laurent::one()],
            mues,
            mu_mode,
            rows: vec![],
        }
    }

    /// Number of elements for which rows have been computed so far.
    #[inline]
    pub fn n(&self) -> usize {
        self.rows.len()
    }

    /// Number of generators.
    #[inline]
    pub fn rank(&self) -> usize {
        self.elms.rank
    }

    /// Return `true` iff `y ≤ w` in Bruhat order.
    ///
    /// Requires `y <= w` (by canonical index); panics in debug mode otherwise.
    /// If `y == w` always returns `true`.
    #[inline]
    pub fn bruhat_leq(&self, y: ElmIdx, w: ElmIdx) -> bool {
        debug_assert!(y <= w, "bruhat_leq: y={y} > w={w}");
        if y == w {
            return true;
        }
        let row = &self.rows[w as usize];
        row.pol[y as usize] != NOT_LEQ
    }

    /// Return the KL polynomial `P_{y,w}`, or `None` if `y ≰ w`.
    ///
    /// Requires `y <= w` (by canonical index).
    #[inline]
    pub fn pol(&self, y: ElmIdx, w: ElmIdx) -> Option<&Laurent> {
        debug_assert!(y <= w, "pol: y={y} > w={w}");
        if y == w {
            // P_{w,w} = 1 = pols[0]
            return Some(&self.pols[0]);
        }
        let row = &self.rows[w as usize];
        let idx = row.pol[y as usize];
        if idx == NOT_LEQ {
            None
        } else {
            Some(&self.pols[idx as usize])
        }
    }

    /// Return the mu coefficient `μ^s_{y,w}`.
    ///
    /// In `Implicit` mode: computed from the KL polynomial as the coefficient
    /// of `v^{L(w) − L(y) − 1}` in `P_{y,w}` (equivalently, the constant term
    /// of `v^{1 + L(y) − L(w)} · P_{y,w}`).  If the slot is not present
    /// (flag false) or `y ≰ w`, returns zero.
    ///
    /// In `Stored` mode: pool lookup.  If `NO_MU` sentinel, returns zero.
    ///
    /// For the Stored-mode hot path prefer [`mu_ref`] to avoid cloning.
    #[inline]
    pub fn mu(&self, s: usize, y: ElmIdx, w: ElmIdx) -> Laurent {
        debug_assert!(y <= w, "mu: y={y} > w={w}");
        let rank = self.rank();

        match self.mu_mode {
            MuMode::Implicit => {
                // Check presence flag
                if y == w {
                    return Laurent::zero();
                }
                let row = &self.rows[w as usize];
                let present = row
                    .mu_present
                    .as_ref()
                    .expect("Implicit-mode row missing mu_present")[y as usize * rank + s];
                if !present {
                    return Laurent::zero();
                }
                // Derive: coefficient of v^{L(w)-L(y)-1} in P_{y,w}.
                // This equals zero_part(v^shift · P_{y,w}) with
                // shift = 1 + L(y) - L(w), but h.coeff(-shift) avoids
                // allocating a shifted copy.
                let Some(h) = self.pol(y, w) else {
                    return Laurent::zero();
                };
                let shift =
                    1i32 + self.lweights[y as usize] as i32 - self.lweights[w as usize] as i32;
                let c = h.coeff(-shift);
                Laurent::monomial(c, 0)
            }
            MuMode::Stored => {
                if y == w {
                    return Laurent::zero();
                }
                let row = &self.rows[w as usize];
                let mu_vec = row.mu.as_ref().expect("Stored mode: mu vec missing");
                let idx = mu_vec[y as usize * rank + s];
                if idx == NO_MU {
                    Laurent::zero()
                } else {
                    self.mues[s][idx as usize].clone()
                }
            }
        }
    }

    /// Return a reference to `μ^s_{y,w}` without cloning (Stored mode only).
    ///
    /// Returns `None` in Implicit mode or when no mu slot exists for `(y, s, w)`.
    /// Task 9's stored-mode hot path should prefer this over [`mu`] to avoid
    /// pool clones.
    pub fn mu_ref(&self, s: usize, y: ElmIdx, w: ElmIdx) -> Option<&Laurent> {
        debug_assert!(y <= w, "mu_ref: y={y} > w={w}");
        if self.mu_mode != MuMode::Stored {
            return None;
        }
        if y == w {
            return None;
        }
        let rank = self.rank();
        let row = &self.rows[w as usize];
        let mu_vec = row.mu.as_ref().expect("Stored mode: mu vec missing");
        let idx = mu_vec[y as usize * rank + s];
        if idx == NO_MU {
            None
        } else {
            Some(&self.mues[s][idx as usize])
        }
    }

    /// Return `true` iff `μ^s_{y,w} ≠ 0`.
    ///
    /// In `Implicit` mode this is cheaper than `mu()` because it reads a single
    /// coefficient without allocating a shifted copy.
    #[inline]
    pub fn mu_is_nonzero(&self, s: usize, y: ElmIdx, w: ElmIdx) -> bool {
        debug_assert!(y <= w, "mu_is_nonzero: y={y} > w={w}");
        if y == w {
            return false;
        }
        let rank = self.rank();

        match self.mu_mode {
            MuMode::Implicit => {
                let row = &self.rows[w as usize];
                let present = row
                    .mu_present
                    .as_ref()
                    .expect("Implicit-mode row missing mu_present")[y as usize * rank + s];
                if !present {
                    return false;
                }
                let Some(h) = self.pol(y, w) else {
                    return false;
                };
                let shift =
                    1i32 + self.lweights[y as usize] as i32 - self.lweights[w as usize] as i32;
                h.coeff(-shift) != 0
            }
            MuMode::Stored => {
                let row = &self.rows[w as usize];
                let mu_vec = row.mu.as_ref().expect("Stored mode: mu vec missing");
                let idx = mu_vec[y as usize * rank + s];
                // Pool invariant: index 0 is the canonical zero; non-zero values
                // are interned at index ≥ 1.  So nonzero iff idx is a valid
                // non-zero slot.
                if idx == NO_MU || idx == 0 {
                    return false;
                }
                debug_assert!(
                    !self.mues[s][idx as usize].is_zero(),
                    "mu pool invariant violated: zero interned at index {idx} (must be 0)"
                );
                true
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{enumerate::ElementTable, group::CoxeterGroup, laurent::Laurent};

    // -----------------------------------------------------------------------
    // Helper: build A1 ElementTable
    // -----------------------------------------------------------------------
    fn a1_table() -> ElementTable {
        let group = CoxeterGroup::from_type("A1").unwrap();
        ElementTable::build(&group)
    }

    // -----------------------------------------------------------------------
    // Test 1: a1_hand_table
    // -----------------------------------------------------------------------
    /// Hand-build a KL table for A1 in Implicit mode and verify all accessors.
    ///
    /// A1 has 2 elements: e (idx 0, length 0) and s0 (idx 1, length 1).
    /// Equal weights [1].  P_{e,e} = P_{s0,s0} = 1, P_{e,s0} = 1.
    /// μ^{s0}_{e,s0}: shift = 1 + L(e) - L(s0) = 1+0-1 = 0; P shifted 0 = 1;
    /// zero_part = 1.  So mu(0, 0, 1) == Laurent::one().
    #[test]
    fn a1_hand_table() {
        let elms = a1_table();
        let mut tbl = KlTable::new_empty(elms, vec![1u32], MuMode::Implicit);

        // Row 0: only y=0 (the element e itself).
        // pol[0] = NOT_LEQ is wrong — e ≤ e, so pol[0] = index of P_{e,e} = 0.
        // But for y == w we short-circuit to pols[0], so pol entry doesn't matter
        // for y==w.  We still set it to pols[0] idx = 0 to be consistent.
        // mu_present for row 0: 1 element × 1 generator = 1 entry.
        let row0 = KlRow {
            pol: vec![0], // P_{e,e} = pols[0] = one
            mu: None,
            mu_present: Some(vec![false]),
        };

        // Row 1: y ∈ {0, 1}.
        // pol[0] = 0 (P_{e,s0} = 1 = pols[0])
        // pol[1] = NOT_LEQ is wrong — but y==w case is short-circuited; we use 0.
        // mu_present: 2 elements × 1 generator = 2 entries.
        //   mu_present[0*1 + 0] = true  (slot for y=e, s=s0 is present → mu=1)
        //   mu_present[1*1 + 0] = false (y==w, no mu)
        let row1 = KlRow {
            pol: vec![0, 0],
            mu: None,
            mu_present: Some(vec![true, false]),
        };

        tbl.rows.push(row0);
        tbl.rows.push(row1);

        // n() == 2
        assert_eq!(tbl.n(), 2);

        // bruhat_leq
        assert!(tbl.bruhat_leq(0, 1), "e ≤ s0");
        assert!(tbl.bruhat_leq(0, 0), "e ≤ e (y==w)");
        assert!(tbl.bruhat_leq(1, 1), "s0 ≤ s0 (y==w)");

        // pol
        assert_eq!(tbl.pol(0, 1), Some(&Laurent::one()), "P_{{e,s0}} = 1");
        assert_eq!(tbl.pol(0, 0), Some(&Laurent::one()), "P_{{e,e}} = 1");

        // mu(s=0, y=0, w=1): shift=0, P=1, zero_part(1)=1 → Laurent::one()
        assert_eq!(tbl.mu(0, 0, 1), Laurent::one(), "mu(s0, e, s0) = 1");
        assert!(tbl.mu_is_nonzero(0, 0, 1), "mu_is_nonzero(s0, e, s0)");

        // mu(s=0, y=1, w=1): y==w → zero
        assert_eq!(tbl.mu(0, 1, 1), Laurent::zero(), "mu(s0, s0, s0) = 0");
        assert!(
            !tbl.mu_is_nonzero(0, 1, 1),
            "mu_is_nonzero(s0, s0, s0) false"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: stored_mode_lookup
    // -----------------------------------------------------------------------
    /// Verify Stored mode: pool seeding and mu lookup via indices.
    #[test]
    fn stored_mode_lookup() {
        let elms = a1_table();
        let mut tbl = KlTable::new_empty(elms, vec![1u32], MuMode::Stored);

        // Pools are seeded: pols[0] == one, mues[0][0] == zero
        assert_eq!(tbl.pols[0], Laurent::one(), "pols[0] seeded to one");
        assert_eq!(tbl.mues[0][0], Laurent::zero(), "mues[0][0] seeded to zero");

        // Append Laurent::one() to mues[0] (index 1 in mues[0])
        tbl.mues[0].push(Laurent::one());

        // Row 0
        let row0 = KlRow {
            pol: vec![0],
            mu: Some(vec![NO_MU]),
            mu_present: None,
        };
        // Row 1: mu for (y=0, s=0, w=1) → mues[0][1] = one
        let row1 = KlRow {
            pol: vec![0, 0],
            mu: Some(vec![
                1,     // y=0, s=0: mues[0][1] = one
                NO_MU, // y=1, s=0: no slot
            ]),
            mu_present: None,
        };
        tbl.rows.push(row0);
        tbl.rows.push(row1);

        // mu(s=0, y=0, w=1) should return mues[0][1] = Laurent::one()
        assert_eq!(
            tbl.mu(0, 0, 1),
            Laurent::one(),
            "stored mu(s0, e, s0) = one"
        );
        assert!(tbl.mu_is_nonzero(0, 0, 1));

        // mu(s=0, y=1, w=1): y==w → zero
        assert_eq!(tbl.mu(0, 1, 1), Laurent::zero());
    }

    // -----------------------------------------------------------------------
    // Test 3: not_leq_sentinel
    // -----------------------------------------------------------------------
    /// A row with a NOT_LEQ entry → bruhat_leq false, pol None, mu zero.
    #[test]
    fn not_leq_sentinel() {
        // Synthesize a 3-element scenario using A2 (6 elements), but we only
        // push one row manually for element index 2 and set y=0 as NOT_LEQ.
        let group = CoxeterGroup::from_type("A2").unwrap();
        let elms = ElementTable::build(&group);
        let mut tbl = KlTable::new_empty(elms, vec![1u32; 2], MuMode::Implicit);

        // Row for w=2 (length-1 element [1]): only y=1 is ≤ w; y=0 is NOT_LEQ.
        // Rank = 2, so mu_present has 3*2 = 6 entries.
        // We'll set NOT_LEQ for y=0 and valid index for y=1.
        let row0 = KlRow {
            pol: vec![0],
            mu: None,
            mu_present: Some(vec![false, false]),
        };
        let row1 = KlRow {
            pol: vec![0, 0],
            mu: None,
            mu_present: Some(vec![false, false, false, false]),
        };
        let row2 = KlRow {
            pol: vec![NOT_LEQ, 0, 0],
            mu: None,
            mu_present: Some(vec![
                false, false, // y=0: not ≤ w, so flags don't matter
                true, false, // y=1: slot for s=0 present
                false, false, // y=2: y==w
            ]),
        };
        tbl.rows.push(row0);
        tbl.rows.push(row1);
        tbl.rows.push(row2);

        // y=0 not ≤ w=2
        assert!(!tbl.bruhat_leq(0, 2), "0 not ≤ 2");
        assert_eq!(tbl.pol(0, 2), None, "pol(0,2) = None");
        assert_eq!(tbl.mu(0, 0, 2), Laurent::zero(), "mu for NOT_LEQ = zero");
        assert!(!tbl.mu_is_nonzero(0, 0, 2));

        // y=1 ≤ w=2
        assert!(tbl.bruhat_leq(1, 2), "1 ≤ 2");
        assert_eq!(tbl.pol(1, 2), Some(&Laurent::one()));
    }
}
