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

/// A Coxeter group element, stored as the first `rank` entries of the
/// corresponding `Perm`.
///
/// Length is always `rank`.  Entry `elm.0[s]` is the image of simple root `s`
/// under the element's permutation.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CoxElm(pub Box<[RootIdx]>);
