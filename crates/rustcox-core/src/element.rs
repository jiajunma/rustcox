//! Shared element and permutation types for Coxeter group elements.
//!
//! These types are shared between roots.rs (Task 4) and the element calculus
//! (Task 5).  Conversion logic between `Perm` and `CoxElm` is added in Task 5.

/// A generator index (simple reflection), 0-indexed.
pub type Gen = u8;

/// An index into the root list (0..2N).
pub type RootIdx = u32;

/// An index into the element table.
pub type ElmIdx = u32;

/// A word in the generators.
pub type Word = Vec<Gen>;

/// A permutation of all 2N roots.
///
/// `perm.0[i]` is the image index of root `i` under this permutation.
/// Length is always 2N where N is the number of positive roots.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Perm(pub Box<[RootIdx]>);

impl Perm {
    /// Return the identity permutation of length `n2` (= 2N).
    ///
    /// Composition note: `identity.then(p) == p` and `p.then(identity) == p`.
    #[inline]
    pub fn identity(n2: usize) -> Perm {
        Perm((0..n2 as u32).collect::<Vec<_>>().into_boxed_slice())
    }

    /// Compose `self` followed by `q`: result[i] = q[self[i]].
    ///
    /// This is PyCox's `permmult(self, q)`:
    /// `then(p, q)[i] = q[p[i]]` — apply `self` first, then `q`.
    #[inline]
    pub fn then(&self, q: &Perm) -> Perm {
        let data: Vec<RootIdx> = self.0.iter().map(|&i| q.0[i as usize]).collect();
        Perm(data.into_boxed_slice())
    }

    /// Return the inverse permutation: inv[p[i]] = i.
    ///
    /// Replicates PyCox `perminverse`.
    pub fn inverse(&self) -> Perm {
        let n = self.0.len();
        let mut inv = vec![0u32; n];
        for (i, &pi) in self.0.iter().enumerate() {
            inv[pi as usize] = i as u32;
        }
        Perm(inv.into_boxed_slice())
    }

    /// Extract the first `rank` entries as a `CoxElm`.
    #[inline]
    pub fn coxelm(&self, rank: usize) -> CoxElm {
        CoxElm(self.0[..rank].to_vec().into_boxed_slice())
    }
}

/// A Coxeter group element, stored as the first `rank` entries of the
/// corresponding `Perm`.
///
/// Length is always `rank`.  Entry `elm.0[s]` is the image of simple root `s`
/// under the element's permutation.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CoxElm(pub Box<[RootIdx]>);
