//! Deterministic layered parallel KL driver (Task 12).
//!
//! This is the HPC centerpiece: [`klpolynomials`] computes the same `KlTable`
//! as the sequential reference [`klpolynomials_seq`], but spreads the work of
//! each length layer across a Rayon thread pool — while remaining
//! **byte-identical** to the sequential output (full `KlTable` `PartialEq`,
//! including pool insertion order).
//!
//! # Why it is deterministic
//!
//! The row kernel [`compute_row`] is pure: it reads only the **frozen** lower
//! layers (completed rows + frozen pools) and the in-progress row's own buffer.
//! The *only* same-length cross-row dependency is the inverse-symmetry read of
//! row `inva[w]` (taken when `inva[w] < w`).  We confine that dependency to
//! **pair-units** `{w, inva[w]}`: a unit owns both rows, computes the smaller
//! (`min`) first, then computes the larger (`max`) with partner access to the
//! `min` row's freshly computed [`RowResult`].  No unit reads another unit's
//! in-flight data, so units within a layer are embarrassingly parallel.
//!
//! Interning (pool growth) is the *only* order-sensitive step, and it is done
//! **sequentially, per layer, with rows sorted by `w` ascending** — exactly the
//! order the sequential driver uses.  Parallelism affects only *when* a row is
//! computed, never the order in which its values are interned.  Hence the pools
//! grow identically regardless of thread count or layer chunking.
//!
//! # Phases (per length layer `l`)
//!
//! 1. **Partition** the layer's contiguous index range `[start, end)` into
//!    units: a singleton `{w}` when `inva[w] == w`, else a pair `{min, max}`
//!    keyed by `min`.  Units are listed in ascending `min` order.
//! 2. **Compute** (parallel): each unit yields its row results.  With
//!    `layer_chunk = Some(k)` the units are processed in consecutive chunks of
//!    `k` to bound in-flight work; the result is independent of `k`.
//! 3. **Intern** (sequential): flatten the layer's `(w, RowResult)` pairs, sort
//!    by `w` ascending, and intern through the shared [`Interner`] — identical
//!    pool growth to the sequential driver.

use crate::{
    element::ElmIdx,
    group::CoxeterGroup,
    kl::{
        compute::{compute_row, klpolynomials_seq, new_kl_table, Interner, KlCtx, RowResult},
        table::KlTable,
        KlError, KlOpts,
    },
};
use rayon::prelude::*;

/// Compute the full KL table using the deterministic layered parallel driver.
///
/// The result is byte-identical to [`klpolynomials_seq`] for the same group and
/// options (verified across thread counts and layer chunkings in
/// `tests/parallel_eq.rs`).
///
/// Threading:
/// - `threads = None`        → the global Rayon pool.
/// - `threads = Some(0 | 1)` → falls back to [`klpolynomials_seq`] (no pool
///   construction; the layered machinery is pure overhead for one thread).
/// - `threads = Some(t > 1)` → a private pool of `t` threads for this call.
pub fn klpolynomials(group: &CoxeterGroup, opts: &KlOpts) -> Result<KlTable, KlError> {
    opts.validate(group)?;

    // Single-threaded request ⇒ use the reference driver directly.  Documented
    // fallback: the layered partition/intern overhead buys nothing at t ≤ 1.
    if matches!(opts.threads, Some(0) | Some(1)) {
        return klpolynomials_seq(group, opts);
    }

    match opts.threads {
        Some(t) => {
            // Private pool for this call; `install` runs the closure on it.
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .map_err(|e| KlError::Parallel(e.to_string()))?;
            pool.install(|| run_layered(group, opts))
        }
        None => run_layered(group, opts),
    }
}

/// A computation unit within a length layer.
///
/// Either a single self-inverse row, or an inverse pair owning both rows.  The
/// pair always computes `min` before `max` so the `max` row's inverse-symmetry
/// read resolves against the `min` row's [`RowResult`] (the partner).
#[derive(Clone, Copy)]
enum Unit {
    /// `inva[w] == w`: no same-layer partner.
    Single(ElmIdx),
    /// `{min, max}` with `min < max` and `inva[min] == max`.
    Pair { min: ElmIdx, max: ElmIdx },
}

/// The layered driver body (runs inside the chosen Rayon pool).
fn run_layered(group: &CoxeterGroup, opts: &KlOpts) -> Result<KlTable, KlError> {
    let uneq = opts.weights.iter().any(|&wt| wt != 1);
    let mut table = new_kl_table(group, opts, uneq);
    let n = table.elms.len();
    let n_pos = table.elms.lengths.iter().copied().max().unwrap_or(0);
    let rank = table.elms.rank;

    let mut interner = Interner::new(rank, uneq);

    // Process length layers 1..=N in order.  Layer 0 (the identity) is already
    // seeded by `new_kl_table`.  Elements are sorted by (length, word lex), so
    // each layer is a contiguous index range; we sweep `w` from 1 to advance
    // through layers without scanning.
    let mut w = 1usize;
    while w < n {
        let layer_len = table.elms.lengths[w];
        // Contiguous range of all elements of this length.
        let start = w;
        let mut end = w + 1;
        while end < n && table.elms.lengths[end] == layer_len {
            end += 1;
        }

        let units = build_units(&table, start as ElmIdx, end as ElmIdx);

        // ---- Compute phase (parallel over units, chunked) ----
        // Each unit yields (w, RowResult) entries.  Chunking bounds in-flight
        // work but does not affect the result (interning is whole-layer).
        let chunk = opts.layer_chunk.filter(|&k| k > 0).unwrap_or(units.len());
        let mut layer_results: Vec<(ElmIdx, RowResult)> = Vec::with_capacity(end - start);

        for unit_chunk in units.chunks(chunk.max(1)) {
            let mut chunk_out: Vec<(ElmIdx, RowResult)> = unit_chunk
                .par_iter()
                .flat_map_iter(|unit| compute_unit(*unit, n_pos, uneq, &table).into_iter())
                .collect();
            layer_results.append(&mut chunk_out);
        }

        // ---- Intern phase (sequential, w ascending) ----
        // Sort by w so the pools grow in the sequential driver's exact order.
        layer_results.sort_by_key(|(idx, _)| *idx);
        debug_assert_eq!(
            layer_results.len(),
            end - start,
            "layer [{start},{end}) produced {} rows",
            layer_results.len()
        );
        for (idx, result) in layer_results {
            debug_assert_eq!(idx as usize, table.rows.len(), "rows interned out of order");
            let row = interner.intern_row(&mut table.pols, &mut table.mues, result, rank);
            table.rows.push(row);
        }

        w = end;
    }

    Ok(table)
}

/// Partition a length layer's index range into computation units.
///
/// A self-inverse row `inva[w] == w` becomes a [`Unit::Single`]; an inverse
/// pair `{w, inva[w]}` becomes a [`Unit::Pair`] keyed by its minimum (so each
/// pair is emitted exactly once).  Both members of a pair share the same length
/// (inversion preserves length), so they always lie in the same layer.
fn build_units(table: &KlTable, start: ElmIdx, end: ElmIdx) -> Vec<Unit> {
    let mut units = Vec::new();
    for w in start..end {
        let iw = table.elms.inva[w as usize];
        debug_assert_eq!(
            table.elms.lengths[iw as usize], table.elms.lengths[w as usize],
            "inva must preserve length (w={w}, iw={iw})"
        );
        if iw == w {
            units.push(Unit::Single(w));
        } else if iw > w {
            // Key the pair by its minimum so it is emitted once.
            units.push(Unit::Pair { min: w, max: iw });
        }
        // iw < w: this pair was already emitted when we visited `iw`.
    }
    units
}

/// Build the read-only kernel context over the frozen lower layers, with an
/// optional same-layer partner.
fn make_ctx<'a>(
    table: &'a KlTable,
    n_pos: u32,
    uneq: bool,
    partner: Option<(ElmIdx, &'a RowResult)>,
) -> KlCtx<'a> {
    KlCtx {
        elms: &table.elms,
        n_pos,
        lweights: &table.lweights,
        weights: &table.weights,
        uneq,
        rows: &table.rows,
        pols: &table.pols,
        mues: &table.mues,
        partner,
    }
}

/// Compute the row(s) of one unit, returning `(w, RowResult)` entries.
///
/// A [`Unit::Pair`] computes `min` first (no partner), then `max` with partner
/// access to the `min` result so its inverse-symmetry read of row `min` is
/// served from `min`'s freshly computed values rather than the not-yet-interned
/// global rows.
fn compute_unit(unit: Unit, n_pos: u32, uneq: bool, table: &KlTable) -> Vec<(ElmIdx, RowResult)> {
    match unit {
        Unit::Single(w) => {
            let ctx = make_ctx(table, n_pos, uneq, None);
            vec![(w, compute_row(w, &ctx))]
        }
        Unit::Pair { min, max } => {
            // Compute the smaller index first, with no partner.
            let min_ctx = make_ctx(table, n_pos, uneq, None);
            let min_res = compute_row(min, &min_ctx);
            // Compute the larger index with partner access to `min`.  The only
            // same-layer read of `max`'s kernel is the inverse-symmetry read of
            // row `min` (== inva[max]), resolved from `min_res`.
            let max_ctx = make_ctx(table, n_pos, uneq, Some((min, &min_res)));
            let max_res = compute_row(max, &max_ctx);
            vec![(min, min_res), (max, max_res)]
        }
    }
}
