//! Sparse `nc × nc` block storage for the `relklpols` wavefront.
//!
//! Replaces the current dense `Vec<Vec<SlotState>>` block representation
//! (`relkl.rs:464–500`) with a `(bitmap, Vec<value>)` per row, mirroring the
//! storage used by Atlas of Lie Groups' K-L row format
//! (`sources/stand-alone/KLread.cpp:162–224`).
//!
//! ## Why a sparse representation
//!
//! On the OOM-killed E8 cells run (issue #13), the `mat` block grid for one
//! outer rep with `nc = 378` dominated 235 GB RSS at wavefront layer 239.
//! Each block stored `nc × nc = 142_884` slots regardless of how many were
//! `Absent`; only the `Done` slots carry information.  Empirically (on
//! H4 / D6 / E6, from `extremal_stats.rs`) the extremal subset captures every
//! distinct polynomial value, so non-extremal slots are pure storage
//! redundancy — they can be reconstructed at lookup via
//! [`crate::kl::push_to_extremal`].
//!
//! ## Design
//!
//! This module is intentionally storage-only and predicate-agnostic:
//!
//! - [`BitMask`] — a fixed-length packed bitset (`u64` words) with O(1) set,
//!   O(1) contains, O(words) popcount, and a linear iterator over set
//!   positions.
//! - [`SparseRow<T>`] — `BitMask` for membership plus a `Vec<T>` for values,
//!   one entry per set bit, in position-ascending order.
//! - [`SparseBlock<T>`] — `nc` rows, each independently sparse.
//!
//! The relkl wavefront constructs the bitmasks once per block from the
//! extremality predicate, then fills values during Phase 1 (parallel inline
//! compute) and Phase 2 (sequential intern).  No `Absent` storage; no
//! `Pending` storage either — the wavefront tracks pendingness implicitly
//! via the `lw[x] + lw1[u] < lw[y] + lw1[v]` condition.
//!
//! ## Determinism contract
//!
//! Iteration over a row's set positions visits them in `u`-ascending order,
//! exactly matching the dense `(v, u)` lex order required by the Phase 2
//! intern walk (`relkl.rs:638–693`) — see the determinism argument in issue
//! #13 and in the plan doc.

use std::iter::{ExactSizeIterator, FusedIterator};

// ---------------------------------------------------------------------------
// BitMask
// ---------------------------------------------------------------------------

/// A fixed-length packed bitset of width `nc`.
///
/// Backed by `Vec<u64>` for cache-friendly iteration: typical row widths
/// (`nc` up to a few hundred) fit in a handful of cache lines.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BitMask {
    /// Packed bits, LSB-first within each word.
    words: Vec<u64>,
    /// Number of valid bits (positions `0..nc`).
    nc: usize,
}

impl BitMask {
    /// Create an all-zero mask of width `nc`.
    pub fn new(nc: usize) -> Self {
        let n_words = nc.div_ceil(64);
        BitMask {
            words: vec![0u64; n_words],
            nc,
        }
    }

    /// Mask width.
    #[inline]
    pub fn len(&self) -> usize {
        self.nc
    }

    /// True iff width is zero.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nc == 0
    }

    /// Set position `i`.  Panics if `i >= len()`.
    #[inline]
    pub fn set(&mut self, i: usize) {
        assert!(i < self.nc, "BitMask::set: i={i} >= nc={}", self.nc);
        let w = i >> 6;
        let b = i & 63;
        self.words[w] |= 1u64 << b;
    }

    /// True iff position `i` is set.  Panics if `i >= len()`.
    #[inline]
    pub fn contains(&self, i: usize) -> bool {
        assert!(i < self.nc, "BitMask::contains: i={i} >= nc={}", self.nc);
        let w = i >> 6;
        let b = i & 63;
        (self.words[w] >> b) & 1 == 1
    }

    /// Total number of set bits.  O(words).
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// 0-based position of bit `i` among the set bits — i.e. how many set
    /// bits precede position `i`.  Defined only when `contains(i)`; for an
    /// unset `i` this returns the rank of the first set bit > `i`.
    pub fn rank(&self, i: usize) -> usize {
        assert!(i < self.nc, "BitMask::rank: i={i} >= nc={}", self.nc);
        let w = i >> 6;
        let b = i & 63;
        let mut count = 0usize;
        for k in 0..w {
            count += self.words[k].count_ones() as usize;
        }
        // Count bits in word w below bit b.
        let mask_below = if b == 0 { 0 } else { (1u64 << b) - 1 };
        count += (self.words[w] & mask_below).count_ones() as usize;
        count
    }

    /// Iterate set positions in ascending order.
    pub fn iter_set(&self) -> SetBitsIter<'_> {
        SetBitsIter {
            words: &self.words,
            nc: self.nc,
            cur_word_idx: 0,
            cur_word_remaining: self.words.first().copied().unwrap_or(0),
        }
    }
}

/// Iterator over set bits of a [`BitMask`] in ascending position order.
pub struct SetBitsIter<'a> {
    words: &'a [u64],
    nc: usize,
    cur_word_idx: usize,
    cur_word_remaining: u64,
}

impl Iterator for SetBitsIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if self.cur_word_remaining != 0 {
                let bit_in_word = self.cur_word_remaining.trailing_zeros() as usize;
                self.cur_word_remaining &= self.cur_word_remaining - 1; // clear lowest set bit
                let pos = (self.cur_word_idx << 6) | bit_in_word;
                if pos >= self.nc {
                    // Trailing bits past nc were never set, but guard anyway.
                    return None;
                }
                return Some(pos);
            }
            self.cur_word_idx += 1;
            if self.cur_word_idx >= self.words.len() {
                return None;
            }
            self.cur_word_remaining = self.words[self.cur_word_idx];
        }
    }
}

impl FusedIterator for SetBitsIter<'_> {}

// ---------------------------------------------------------------------------
// SparseRow
// ---------------------------------------------------------------------------

/// A length-`nc` row that stores values only at positions where
/// `present.contains(i)` is true.
///
/// Values are kept in position-ascending order (the canonical lex order
/// required by the relkl Phase 2 intern walk).  The bitmap and the value
/// vector must stay in sync: `values.len() == present.count()` is an
/// invariant maintained by the constructor and every mutator.
#[derive(Clone, Debug)]
pub struct SparseRow<T> {
    pub present: BitMask,
    pub values: Vec<T>,
}

impl<T> SparseRow<T> {
    /// Build a row with the given membership bitmap and a value-producing
    /// closure called once per set position in ascending order.
    pub fn from_mask<F>(present: BitMask, mut make_value: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        let n = present.count();
        let mut values = Vec::with_capacity(n);
        for pos in present.iter_set() {
            values.push(make_value(pos));
        }
        SparseRow { present, values }
    }

    /// Row width (the underlying logical row length, not the value count).
    #[inline]
    pub fn nc(&self) -> usize {
        self.present.len()
    }

    /// Number of stored values (= popcount of `present`).
    #[inline]
    pub fn count(&self) -> usize {
        self.values.len()
    }

    /// True iff position `i` is in the present set.  Panics if out of range.
    #[inline]
    pub fn contains(&self, i: usize) -> bool {
        self.present.contains(i)
    }

    /// Borrow the value at position `i`, or `None` if not present.
    pub fn get(&self, i: usize) -> Option<&T> {
        if !self.present.contains(i) {
            return None;
        }
        Some(&self.values[self.present.rank(i)])
    }

    /// Mutably borrow the value at position `i`, or `None` if not present.
    pub fn get_mut(&mut self, i: usize) -> Option<&mut T> {
        if !self.present.contains(i) {
            return None;
        }
        let r = self.present.rank(i);
        Some(&mut self.values[r])
    }

    /// Overwrite the value at position `i`.  Panics if `i` is not in the
    /// present set — sparse rows are built with the membership fixed; values
    /// are only updated in-place.
    pub fn set(&mut self, i: usize, value: T) {
        assert!(
            self.present.contains(i),
            "SparseRow::set: position {i} not in present set (nc={})",
            self.nc()
        );
        let r = self.present.rank(i);
        self.values[r] = value;
    }

    /// Iterate `(position, &value)` pairs in ascending-position order.
    pub fn iter(&self) -> SparseRowIter<'_, T> {
        SparseRowIter {
            inner: self.present.iter_set(),
            values: &self.values,
            next_value_idx: 0,
        }
    }
}

/// Iterator over `(position, &value)` of a [`SparseRow`].
pub struct SparseRowIter<'a, T> {
    inner: SetBitsIter<'a>,
    values: &'a [T],
    next_value_idx: usize,
}

impl<'a, T> Iterator for SparseRowIter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<(usize, &'a T)> {
        let pos = self.inner.next()?;
        let v = &self.values[self.next_value_idx];
        self.next_value_idx += 1;
        Some((pos, v))
    }
}

impl<T> ExactSizeIterator for SparseRowIter<'_, T> {
    fn len(&self) -> usize {
        self.values.len() - self.next_value_idx
    }
}

impl<T> FusedIterator for SparseRowIter<'_, T> {}

// ---------------------------------------------------------------------------
// SparseBlock
// ---------------------------------------------------------------------------

/// A sparse `nc × nc` block.  Each of the `nc` rows is independently sparse:
/// the present sets are not required to coincide across rows.
///
/// The relkl wavefront constructs one `SparseBlock` per `(y, x)` cell of the
/// outer block grid.  Membership is determined by the extremal predicate on
/// the underlying group pairs.
#[derive(Clone, Debug)]
pub struct SparseBlock<T> {
    pub rows: Vec<SparseRow<T>>,
    pub nc: usize,
}

impl<T> SparseBlock<T> {
    /// Build a block of `nc` rows.
    pub fn with_rows(nc: usize, rows: Vec<SparseRow<T>>) -> Self {
        assert_eq!(
            rows.len(),
            nc,
            "SparseBlock: row count {} != nc {}",
            rows.len(),
            nc
        );
        for (v, row) in rows.iter().enumerate() {
            assert_eq!(
                row.nc(),
                nc,
                "SparseBlock row {v}: nc={} != block nc={}",
                row.nc(),
                nc
            );
        }
        SparseBlock { rows, nc }
    }

    /// Borrow the value at `(v, u)`, or `None` if not present.
    pub fn get(&self, v: usize, u: usize) -> Option<&T> {
        self.rows[v].get(u)
    }

    /// Mutably borrow the value at `(v, u)`, or `None` if not present.
    pub fn get_mut(&mut self, v: usize, u: usize) -> Option<&mut T> {
        self.rows[v].get_mut(u)
    }

    /// Set the value at `(v, u)`.  Panics if `u` is not in row `v`'s present
    /// set.
    pub fn set(&mut self, v: usize, u: usize, value: T) {
        self.rows[v].set(u, value)
    }

    /// Total stored values across all rows.
    pub fn count(&self) -> usize {
        self.rows.iter().map(|r| r.count()).sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // BitMask
    // -----------------------------------------------------------------------

    #[test]
    fn bitmask_empty_zero_count() {
        let bm = BitMask::new(0);
        assert!(bm.is_empty());
        assert_eq!(bm.count(), 0);
        assert_eq!(bm.iter_set().count(), 0);
    }

    #[test]
    fn bitmask_set_contains_count() {
        let mut bm = BitMask::new(200);
        assert_eq!(bm.count(), 0);
        for &i in &[0usize, 63, 64, 65, 127, 128, 199] {
            bm.set(i);
        }
        assert_eq!(bm.count(), 7);
        for i in 0..200 {
            let expected = matches!(i, 0 | 63 | 64 | 65 | 127 | 128 | 199);
            assert_eq!(bm.contains(i), expected, "contains({i})");
        }
    }

    #[test]
    fn bitmask_iter_set_ordered() {
        let mut bm = BitMask::new(150);
        let positions = [3usize, 5, 64, 100, 149];
        for &p in &positions {
            bm.set(p);
        }
        let collected: Vec<_> = bm.iter_set().collect();
        assert_eq!(collected, positions);
    }

    #[test]
    fn bitmask_rank() {
        let mut bm = BitMask::new(70);
        for &p in &[5usize, 10, 64, 65] {
            bm.set(p);
        }
        // Rank-of counts set bits with position < i.
        assert_eq!(bm.rank(0), 0);
        assert_eq!(bm.rank(5), 0); // set bits at positions < 5: none
        assert_eq!(bm.rank(6), 1); // set bits at positions < 6: {5}
        assert_eq!(bm.rank(10), 1); // {5}
        assert_eq!(bm.rank(11), 2); // {5, 10}
        assert_eq!(bm.rank(64), 2); // {5, 10}
        assert_eq!(bm.rank(65), 3); // {5, 10, 64}
        assert_eq!(bm.rank(66), 4); // {5, 10, 64, 65}
    }

    #[test]
    fn bitmask_dense_then_sparse_width_300() {
        // Cross a 64-bit word boundary multiple times.
        let mut bm = BitMask::new(300);
        for i in 0..300 {
            if i % 7 == 0 {
                bm.set(i);
            }
        }
        let set_positions: Vec<_> = bm.iter_set().collect();
        let expected: Vec<usize> = (0..300).filter(|i| i % 7 == 0).collect();
        assert_eq!(set_positions, expected);
        assert_eq!(bm.count(), expected.len());
    }

    #[test]
    #[should_panic(expected = "BitMask::set: i=10 >= nc=10")]
    fn bitmask_set_out_of_range_panics() {
        let mut bm = BitMask::new(10);
        bm.set(10);
    }

    // -----------------------------------------------------------------------
    // SparseRow
    // -----------------------------------------------------------------------

    #[test]
    fn sparse_row_from_mask_invariant() {
        let mut present = BitMask::new(20);
        for &p in &[2usize, 7, 8, 15] {
            present.set(p);
        }
        let row: SparseRow<u32> = SparseRow::from_mask(present, |pos| pos as u32 * 10);
        // Length invariant.
        assert_eq!(row.count(), 4);
        assert_eq!(row.values.len(), row.present.count());
        // Lookup vs dense reference.
        for i in 0..20 {
            let expected = matches!(i, 2 | 7 | 8 | 15).then_some(i as u32 * 10);
            assert_eq!(row.get(i).copied(), expected, "get({i})");
        }
        // Iter order matches sorted set positions.
        let collected: Vec<_> = row.iter().map(|(p, &v)| (p, v)).collect();
        assert_eq!(
            collected,
            vec![(2, 20), (7, 70), (8, 80), (15, 150)]
        );
    }

    #[test]
    fn sparse_row_set_in_place_updates_value() {
        let mut present = BitMask::new(8);
        present.set(3);
        present.set(6);
        let mut row: SparseRow<i64> = SparseRow::from_mask(present, |_| 0);
        row.set(3, 42);
        row.set(6, -7);
        assert_eq!(row.get(3), Some(&42));
        assert_eq!(row.get(6), Some(&-7));
        // get_mut for in-place edit.
        *row.get_mut(3).unwrap() += 1;
        assert_eq!(row.get(3), Some(&43));
    }

    #[test]
    #[should_panic(expected = "position 4 not in present set")]
    fn sparse_row_set_non_present_panics() {
        let mut present = BitMask::new(8);
        present.set(3);
        let mut row: SparseRow<i64> = SparseRow::from_mask(present, |_| 0);
        row.set(4, 99);
    }

    #[test]
    fn sparse_row_empty_present() {
        let present = BitMask::new(5);
        let row: SparseRow<u8> = SparseRow::from_mask(present, |_| 0);
        assert_eq!(row.count(), 0);
        assert_eq!(row.iter().count(), 0);
        for i in 0..5 {
            assert!(row.get(i).is_none());
        }
    }

    // -----------------------------------------------------------------------
    // SparseBlock — invariant equivalence with a dense reference
    // -----------------------------------------------------------------------

    /// Build a `SparseBlock<u32>` and a dense `Vec<Vec<Option<u32>>>` from
    /// the same data and verify they agree.  This is the property the relkl
    /// wavefront depends on: dense lookups and sparse lookups must coincide
    /// on the present set, and the iteration order must follow ascending u.
    #[test]
    fn sparse_block_equivalent_to_dense_reference() {
        let nc = 12;
        // Dense reference: store Some(value) where present, None otherwise.
        let dense_present: Vec<Vec<bool>> = (0..nc)
            .map(|v| {
                (0..nc)
                    .map(|u| (v * 3 + u * 5 + 1) % 7 < 3) // pseudo-random sparsity
                    .collect()
            })
            .collect();
        let value_of = |v: usize, u: usize| -> u32 { (v as u32) * 100 + (u as u32) };
        let dense: Vec<Vec<Option<u32>>> = dense_present
            .iter()
            .enumerate()
            .map(|(v, row)| {
                row.iter()
                    .enumerate()
                    .map(|(u, &present)| present.then_some(value_of(v, u)))
                    .collect()
            })
            .collect();

        // Sparse: build from the same membership and value source.
        let rows: Vec<SparseRow<u32>> = dense_present
            .iter()
            .enumerate()
            .map(|(v, row_present)| {
                let mut mask = BitMask::new(nc);
                for (u, &p) in row_present.iter().enumerate() {
                    if p {
                        mask.set(u);
                    }
                }
                SparseRow::from_mask(mask, |u| value_of(v, u))
            })
            .collect();
        let block = SparseBlock::with_rows(nc, rows);

        // Per-slot agreement.
        for v in 0..nc {
            for u in 0..nc {
                assert_eq!(
                    block.get(v, u).copied(),
                    dense[v][u],
                    "disagreement at (v={v}, u={u})"
                );
            }
        }

        // Iteration order is ascending u within each row.
        for v in 0..nc {
            let mut last: Option<usize> = None;
            for (u, _) in block.rows[v].iter() {
                if let Some(prev) = last {
                    assert!(
                        u > prev,
                        "row {v} iteration not ascending: {prev} -> {u}"
                    );
                }
                last = Some(u);
            }
        }

        // Total count matches.
        let dense_count: usize = dense
            .iter()
            .map(|row| row.iter().filter(|s| s.is_some()).count())
            .sum();
        assert_eq!(block.count(), dense_count);
    }

    /// Stress-size block: 64 × 64 with ~30% density.  Verifies BitMask
    /// performance and row construction at sizes relevant to medium groups
    /// (D6 / E6) without the overhead of running a full KL computation.
    #[test]
    fn sparse_block_64x64_dense_check() {
        let nc = 64;
        let value_of = |v: usize, u: usize| -> i32 { (v as i32) - (u as i32) };

        let dense: Vec<Vec<Option<i32>>> = (0..nc)
            .map(|v: usize| {
                (0..nc)
                    .map(|u: usize| {
                        // ~30% density: hash and threshold.
                        let h = (v.wrapping_mul(2654435761) ^ u.wrapping_mul(40503)) & 0xff;
                        if h < 80 {
                            Some(value_of(v, u))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        let rows: Vec<SparseRow<i32>> = (0..nc)
            .map(|v| {
                let mut mask = BitMask::new(nc);
                for (u, slot) in dense[v].iter().enumerate() {
                    if slot.is_some() {
                        mask.set(u);
                    }
                }
                SparseRow::from_mask(mask, |u| value_of(v, u))
            })
            .collect();
        let block = SparseBlock::with_rows(nc, rows);

        for v in 0..nc {
            for u in 0..nc {
                assert_eq!(block.get(v, u).copied(), dense[v][u]);
            }
        }
    }
}
