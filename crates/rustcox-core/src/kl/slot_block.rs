//! `SparseBlock<SlotState>` with backward-compatible `[v][u]` indexing.
//!
//! The `relkl` wavefront previously stored each `(y, x)` block as a dense
//! `Vec<Vec<SlotState>>` of shape `nc × nc`.  This module swaps the storage
//! to [`SparseBlock<SlotState>`] but exposes the same `block[v][u]` access
//! pattern via specialized [`Index`] / [`IndexMut`] impls, so call sites in
//! `relkl.rs` and `relkl_recur.rs` continue to compile unchanged.
//!
//! ## Semantics summary
//!
//! - `block[v][u]` reads `&SlotState`.  Absent slots return a `&'static`
//!   reference to [`SlotState::Absent`] — matching the dense interpretation,
//!   where an absent slot stored the explicit `Absent` variant.
//! - `block[v][u] = new_state` writes via `IndexMut`.  This asserts the slot
//!   `(v, u)` is in row `v`'s `present` set.  The current `relkl` code never
//!   writes `Absent` back into a block, only `Pending` (at init) or `Done`
//!   (during Phase 2), so this invariant is naturally maintained.
//!
//! ## Memory model
//!
//! `nc × nc` dense `SlotState` cells (12 bytes each, for the 3-variant enum)
//! collapse to:
//! - one `BitMask` per row of width `nc` (≈ `nc / 8 + 24` bytes per row);
//! - one `SlotState` value per *present* slot, where presence is determined
//!   by the membership computed at init (`lw[x] + lw1[u] < lw[y] + lw1[v]` on
//!   off-diagonals; `cell1.klmat`-derived on diagonals).
//!
//! Absent slots cost no storage.  See plan
//! [`docs/superpowers/plans/2026-06-18-memory-compression-relkl.md`] and
//! issue #13 for the full memory accounting and the determinism contract.

use std::ops::{Index, IndexMut};

use super::relkl_recur::SlotState;
use super::sparse_block::{BitMask, SparseBlock, SparseRow};

/// A static so [`Index`] for `SparseRow<SlotState>` can hand back a real
/// `&SlotState` reference for absent positions.
///
/// `SlotState` is `Copy` and contains only `u32` payloads, so it is trivially
/// `Sync` and a static reference of any lifetime is safe.
static ABSENT_SLOT: SlotState = SlotState::Absent;

// ---------------------------------------------------------------------------
// Indexing impls: block[v] → &SparseRow<SlotState>, row[u] → &SlotState
// ---------------------------------------------------------------------------

impl Index<usize> for SparseBlock<SlotState> {
    type Output = SparseRow<SlotState>;
    #[inline]
    fn index(&self, v: usize) -> &Self::Output {
        &self.rows[v]
    }
}

impl IndexMut<usize> for SparseBlock<SlotState> {
    #[inline]
    fn index_mut(&mut self, v: usize) -> &mut Self::Output {
        &mut self.rows[v]
    }
}

impl Index<usize> for SparseRow<SlotState> {
    type Output = SlotState;

    /// `row[u]` returns `&SlotState`.  For absent `u`, returns a static
    /// `&SlotState::Absent`; for present `u`, returns the stored value.
    ///
    /// Panics on out-of-bounds `u >= nc` (matching `Vec::index`).
    #[inline]
    fn index(&self, u: usize) -> &Self::Output {
        if !self.present.contains(u) {
            return &ABSENT_SLOT;
        }
        &self.values[self.present.rank(u)]
    }
}

impl IndexMut<usize> for SparseRow<SlotState> {
    /// `row[u] = new_state` writes into a present slot.  Panics if `u` is not
    /// already in `present` — the caller must have built the row with `u` as
    /// a present position.  This matches the relkl invariant: every slot the
    /// wavefront writes was already marked at init time.
    #[inline]
    fn index_mut(&mut self, u: usize) -> &mut Self::Output {
        assert!(
            self.present.contains(u),
            "SparseRow<SlotState>::index_mut: u={u} not in present set (nc={})",
            self.present.len()
        );
        let r = self.present.rank(u);
        &mut self.values[r]
    }
}

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

/// Build a `SparseBlock<SlotState>` row-by-row from a `present` predicate plus
/// an initial-value closure.  For each row `v ∈ 0..nc`, every `u` where
/// `present(v, u)` returns true is added to the row's bitmask and gets its
/// initial value from `value(v, u)`.
///
/// The result has `nc` rows and is suitable for use as a working block in
/// the `relkl` wavefront.
pub(super) fn build_block<P, V>(nc: usize, mut present: P, mut value: V) -> SparseBlock<SlotState>
where
    P: FnMut(usize, usize) -> bool,
    V: FnMut(usize, usize) -> SlotState,
{
    let mut rows = Vec::with_capacity(nc);
    for v in 0..nc {
        let mut mask = BitMask::new(nc);
        for u in 0..nc {
            if present(v, u) {
                mask.set(u);
            }
        }
        let row = SparseRow::from_mask(mask, |u| value(v, u));
        rows.push(row);
    }
    SparseBlock::with_rows(nc, rows)
}

// ---------------------------------------------------------------------------
// Dense ↔ sparse conversion (used at the block-log boundary; the on-disk
// log format remains dense `Vec<Vec<SlotState>>` for backwards compatibility
// while issue #13's BLK_VERSION bump is deferred to a follow-up.)
// ---------------------------------------------------------------------------

/// Re-pack a sparse block as a dense `nc × nc` grid of `SlotState`.
///
/// Absent slots become explicit `SlotState::Absent`.  Used to serialize a
/// `SparseBlock<SlotState>` into the current Q4 layer log format
/// (`relkl_ckpt::LayerRecord::blocks`).
pub(super) fn block_to_dense(block: &SparseBlock<SlotState>) -> Vec<Vec<SlotState>> {
    let nc = block.nc;
    (0..nc)
        .map(|v| {
            let row = &block.rows[v];
            (0..nc)
                .map(|u| {
                    if row.present.contains(u) {
                        row.values[row.present.rank(u)]
                    } else {
                        SlotState::Absent
                    }
                })
                .collect()
        })
        .collect()
}

/// Construct a sparse block from a dense `nc × nc` grid.
///
/// The present set is exactly the positions where the dense grid is not
/// `Absent`; values are copied over in `u`-ascending order so the resulting
/// row matches the iteration order required by the Phase 2 intern walk.
///
/// Used when replaying a Q4 layer log into a freshly initialized `mat` after
/// a SLURM-kill resume.  Panics on shape mismatch.
pub(super) fn block_from_dense(dense: &[Vec<SlotState>]) -> SparseBlock<SlotState> {
    let nc = dense.len();
    let rows: Vec<SparseRow<SlotState>> = dense
        .iter()
        .enumerate()
        .map(|(v, row)| {
            assert_eq!(
                row.len(),
                nc,
                "block_from_dense: row {v} has len {} != nc {nc}",
                row.len()
            );
            let mut mask = BitMask::new(nc);
            for (u, s) in row.iter().enumerate() {
                if !matches!(s, SlotState::Absent) {
                    mask.set(u);
                }
            }
            SparseRow::from_mask(mask, |u| row[u])
        })
        .collect();
    SparseBlock::with_rows(nc, rows)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: dense → sparse → dense, agreement.
    #[test]
    fn block_roundtrip_dense_sparse() {
        // Hand-crafted nc=4 grid with all three SlotState variants.
        let nc = 4;
        let dense: Vec<Vec<SlotState>> = vec![
            vec![
                SlotState::Done { rk: 0, mu: 0 },
                SlotState::Absent,
                SlotState::Pending,
                SlotState::Absent,
            ],
            vec![
                SlotState::Absent,
                SlotState::Done { rk: 5, mu: 7 },
                SlotState::Absent,
                SlotState::Pending,
            ],
            vec![
                SlotState::Pending,
                SlotState::Pending,
                SlotState::Done { rk: 1, mu: 0 },
                SlotState::Absent,
            ],
            vec![
                SlotState::Absent,
                SlotState::Absent,
                SlotState::Absent,
                SlotState::Done { rk: 1, mu: 0 },
            ],
        ];

        let block = block_from_dense(&dense);
        let round = block_to_dense(&block);

        for v in 0..nc {
            for u in 0..nc {
                assert_eq!(
                    round[v][u], dense[v][u],
                    "round-trip mismatch at (v={v}, u={u})"
                );
            }
        }
    }

    /// `block[v][u]` (Index) returns the right value for present + absent
    /// positions, matching dense semantics.
    #[test]
    fn block_index_semantics() {
        let dense: Vec<Vec<SlotState>> = vec![
            vec![SlotState::Pending, SlotState::Absent],
            vec![SlotState::Absent, SlotState::Done { rk: 9, mu: 3 }],
        ];
        let block = block_from_dense(&dense);

        assert_eq!(block[0][0], SlotState::Pending);
        assert_eq!(block[0][1], SlotState::Absent);
        assert_eq!(block[1][0], SlotState::Absent);
        assert_eq!(block[1][1], SlotState::Done { rk: 9, mu: 3 });

        // `is_marked()` reads via auto-deref.
        assert!(block[0][0].is_marked());
        assert!(!block[0][1].is_marked());
        assert!(block[1][1].is_marked());
        assert_eq!(block[1][1].rk(), Some(9));
        assert_eq!(block[1][1].mu(), Some(3));
    }

    /// `block[v][u] = new_state` works at present positions; panics at
    /// absent positions (preventing silent bitmap drift).
    #[test]
    fn block_index_mut_writes_present_slot() {
        let dense: Vec<Vec<SlotState>> = vec![
            vec![SlotState::Pending, SlotState::Absent],
            vec![SlotState::Absent, SlotState::Pending],
        ];
        let mut block = block_from_dense(&dense);

        // Promote Pending → Done in place.
        block[0][0] = SlotState::Done { rk: 4, mu: 2 };
        assert_eq!(block[0][0], SlotState::Done { rk: 4, mu: 2 });

        block[1][1] = SlotState::Done { rk: 0, mu: 0 };
        assert_eq!(block[1][1], SlotState::Done { rk: 0, mu: 0 });
    }

    #[test]
    #[should_panic(expected = "u=1 not in present set")]
    fn block_index_mut_absent_panics() {
        // Square 2×2 grid: (0,0) = Pending (present), (0,1) = Absent (writing
        // here must panic), (1,0) absent, (1,1) Pending.
        let dense: Vec<Vec<SlotState>> = vec![
            vec![SlotState::Pending, SlotState::Absent],
            vec![SlotState::Absent, SlotState::Pending],
        ];
        let mut block = block_from_dense(&dense);
        block[0][1] = SlotState::Done { rk: 1, mu: 1 };
    }

    /// `build_block` constructs the same shape as `block_from_dense` for a
    /// matching predicate.
    #[test]
    fn build_block_matches_dense_with_predicate() {
        let nc = 5;
        // Predicate: present iff (v + u) is odd.
        let present_fn = |v: usize, u: usize| (v + u) % 2 == 1;
        let value_fn = |v: usize, u: usize| SlotState::Done {
            rk: (v * 10 + u) as u32,
            mu: 0,
        };

        let dense: Vec<Vec<SlotState>> = (0..nc)
            .map(|v| {
                (0..nc)
                    .map(|u| {
                        if present_fn(v, u) {
                            value_fn(v, u)
                        } else {
                            SlotState::Absent
                        }
                    })
                    .collect()
            })
            .collect();

        let from_dense = block_from_dense(&dense);
        let built = build_block(nc, present_fn, value_fn);

        for v in 0..nc {
            for u in 0..nc {
                assert_eq!(
                    from_dense[v][u], built[v][u],
                    "predicate vs dense at (v={v}, u={u})"
                );
            }
        }
    }
}
