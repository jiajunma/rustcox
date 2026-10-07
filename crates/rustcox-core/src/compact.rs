//! Compact equal-parameter KL engine. The existing `kl` engine remains the oracle.
//!
//! Persistent rows contain a Bruhat bitset, nontrivial extremal polynomial IDs,
//! and scalar nonzero mu support. A single current row uses IDs, not Laurent
//! clones. Recurrences and publication follow the reference's canonical order.
//! This is a sequential full-table engine, NOT an E8 all-pairs engine.
mod interval_cache;

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::mem::size_of;

use crate::{
    element::ElmIdx, enumerate::ElementTable, group::CoxeterGroup, kl::KlError, laurent::Laurent,
};
use interval_cache::IntervalCache;
pub use interval_cache::IntervalLimits;

/// ID zero is the polynomial one; MAX denotes an incomparable current-row slot.
const MISSING: u32 = u32::MAX;

#[derive(Clone, Debug)]
pub struct CompactOpts {
    /// Refuse a quadratic Bruhat allocation larger than this bound, before
    /// enumerating the group. This is not a bound on total process RSS.
    pub max_bruhat_bytes: u64,
    /// Experimental, exact-match-only interval memoization. Off by default.
    pub intervals: Option<IntervalLimits>,
}

impl Default for CompactOpts {
    fn default() -> Self {
        Self {
            max_bruhat_bytes: 512 * 1024 * 1024,
            intervals: None,
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct CompactStats {
    pub comparable_strict: u64,
    pub inverse_hits: u64,
    pub left_hits: u64,
    pub right_hits: u64,
    pub short_hits: u64,
    pub full_candidates: u64,
    pub recurrences: u64,
    pub legacy_z_candidates: u64,
    pub sparse_mu_candidates: u64,
    pub contributing_mu_terms: u64,
    pub interval_attempts: u64,
    pub interval_hits: u64,
    pub interval_build_aborts: u64,
    pub interval_search_aborts: u64,
    pub interval_classes: u64,
    /// Conservative cache accounting, excluding the bounded lookup scratch.
    pub interval_cache_bytes: usize,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct CompactStorage {
    pub bruhat_words_bytes: usize,
    pub extremal_ids_bytes: usize,
    pub mu_support_bytes: usize,
    pub covers_bytes: usize,
    pub row_headers_bytes: usize,
    pub polynomial_coeff_payload_bytes: usize,
    pub mu_flag_bytes: usize,
}

struct Row {
    below: Box<[u64]>,
    /// Ascending bottom indices; polynomial one is omitted.
    nontrivial: Box<[(ElmIdx, u32)]>,
    /// Ascending bottom indices; zeros omitted, generators not duplicated.
    mu: Box<[(ElmIdx, i64)]>,
    /// Only constructed when interval memoization is enabled.
    covers: Box<[ElmIdx]>,
}

pub struct CompactKlTable {
    pub elms: ElementTable,
    /// Same first-occurrence order as the original sequential engine.
    pub pols: Vec<Laurent>,
    pub stats: CompactStats,
    rows: Vec<Row>,
    left: Vec<u64>,
    right: Vec<u64>,
}

impl CompactKlTable {
    /// Safe on either canonical index order; outside-table queries are false.
    #[inline]
    pub fn leq(&self, y: ElmIdx, w: ElmIdx) -> bool {
        if y > w || w as usize >= self.rows.len() {
            return false;
        }
        let y = y as usize;
        (self.rows[w as usize].below[y / 64] >> (y % 64)) & 1 != 0
    }

    #[inline]
    fn rft(&self, y: ElmIdx, s: usize) -> ElmIdx {
        self.elms.inva[self.elms.lft(self.elms.inva[y as usize], s) as usize]
    }

    #[inline]
    fn extremal(&self, y: ElmIdx, w: ElmIdx) -> bool {
        self.left[w as usize] & !self.left[y as usize] == 0
            && self.right[w as usize] & !self.right[y as usize] == 0
    }

    /// Reconstruct omitted entries by the SAME descent reductions as PyCox.
    /// This never treats an uncomputed row as a completed all-one row: callers
    /// of the recurrence only query strictly earlier rows.
    pub fn pol_id(&self, mut y: ElmIdx, w: ElmIdx) -> Option<u32> {
        if !self.leq(y, w) {
            return None;
        }
        loop {
            if y == w {
                return Some(0);
            }
            let l = self.left[w as usize] & !self.left[y as usize];
            if l != 0 {
                y = self.elms.lft(y, l.trailing_zeros() as usize);
                continue;
            }
            let r = self.right[w as usize] & !self.right[y as usize];
            if r != 0 {
                y = self.rft(y, r.trailing_zeros() as usize);
                continue;
            }
            break;
        }
        let entries = &self.rows[w as usize].nontrivial;
        Some(match entries.binary_search_by_key(&y, |&(z, _)| z) {
            Ok(i) => entries[i].1,
            Err(_) => 0,
        })
    }

    pub fn pol(&self, y: ElmIdx, w: ElmIdx) -> Option<&Laurent> {
        self.pol_id(y, w).map(|i| &self.pols[i as usize])
    }

    /// Generator-conditioned scalar mu; reproduces reference slot semantics.
    pub fn mu_for(&self, s: usize, y: ElmIdx, w: ElmIdx) -> i64 {
        if s >= self.elms.rank || y >= w || !self.leq(y, w) {
            return 0;
        }
        if self.left[y as usize] & !self.left[w as usize] & (1u64 << s) == 0 {
            return 0;
        }
        let support = &self.rows[w as usize].mu;
        match support.binary_search_by_key(&y, |&(z, _)| z) {
            Ok(i) => support[i].1,
            Err(_) => 0,
        }
    }

    /// Logical payload, not allocator overhead and not peak RSS.
    pub fn storage(&self) -> CompactStorage {
        CompactStorage {
            bruhat_words_bytes: self.rows.iter().map(|r| r.below.len() * 8).sum(),
            extremal_ids_bytes: self
                .rows
                .iter()
                .map(|r| r.nontrivial.len() * size_of::<(ElmIdx, u32)>())
                .sum(),
            mu_support_bytes: self
                .rows
                .iter()
                .map(|r| r.mu.len() * size_of::<(ElmIdx, i64)>())
                .sum(),
            covers_bytes: self.rows.iter().map(|r| r.covers.len() * 4).sum(),
            row_headers_bytes: self.rows.capacity() * size_of::<Row>(),
            polynomial_coeff_payload_bytes: self.pols.iter().map(|p| p.coeffs().len() * 8).sum(),
            mu_flag_bytes: 0,
        }
    }
}

/// Ephemeral fingerprint index. Coefficients are owned ONLY by table.pols.
#[derive(Default)]
struct PoolIndex(HashMap<u64, Vec<u32>>);

impl PoolIndex {
    fn intern(&mut self, pool: &mut Vec<Laurent>, p: Laurent) -> u32 {
        let mut hasher = DefaultHasher::new();
        p.hash(&mut hasher);
        let candidates = self.0.entry(hasher.finish()).or_default();
        if let Some(&id) = candidates.iter().find(|&&id| pool[id as usize] == p) {
            return id;
        }
        let id = u32::try_from(pool.len()).expect("polynomial pool exceeds u32 IDs");
        assert_ne!(id, MISSING, "polynomial ID sentinel exhausted");
        pool.push(p);
        candidates.push(id);
        id
    }
}

/// Compute all ordinary, equal-parameter KL values in compact form.
pub fn compute(group: &CoxeterGroup, opts: &CompactOpts) -> Result<CompactKlTable, KlError> {
    if group.rank > 64 {
        return Err(KlError::Unimplemented(
            "compact descent masks require rank <= 64",
        ));
    }
    let n = u32::try_from(group.order)
        .map_err(|_| KlError::Internal("compact element IDs exceed u32".into()))?
        as u128;
    // Upper bound for sum_w ceil((w+1)/64) * 8, including row padding.
    let bound = (n * (n + 1) / 2 + 63 * n) / 64 * 8;
    if bound > opts.max_bruhat_bytes as u128 {
        return Err(KlError::Internal(format!(
            "compact Bruhat storage bound {bound} exceeds budget {}; use the cells engine for very large groups",
            opts.max_bruhat_bytes
        )));
    }
    let elms = ElementTable::build(group);
    let n = elms.len();
    let left: Vec<u64> = (0..n)
        .map(|w| {
            (0..group.rank).fold(0, |m, s| {
                m | (u64::from(elms.lft(w as u32, s) < w as u32) << s)
            })
        })
        .collect();
    let right = elms.inva.iter().map(|&iw| left[iw as usize]).collect();
    let mut table = CompactKlTable {
        elms,
        pols: Vec::new(),
        stats: CompactStats::default(),
        rows: Vec::with_capacity(n),
        left,
        right,
    };
    let mut index = PoolIndex::default();
    index.intern(&mut table.pols, Laurent::one());
    table.rows.push(Row {
        below: vec![1].into_boxed_slice(),
        nontrivial: Box::new([]),
        mu: Box::new([]),
        covers: Box::new([]),
    });
    let mut stats = CompactStats::default();
    let mut cache = opts.intervals.clone().map(IntervalCache::new);
    for w in 1..n as u32 {
        let s = table.left[w as usize].trailing_zeros() as usize;
        let sw = table.elms.lft(w, s);
        table
            .rows
            .push(skeleton(&table, w, s, group.n_pos, cache.is_some()));
        let mut current = vec![MISSING; w as usize + 1];
        current[w as usize] = 0;
        let mut nontrivial = Vec::new();
        let mut mu = Vec::new();
        for y in (0..w).rev() {
            if !table.leq(y, w) {
                continue;
            }
            stats.comparable_strict += 1;
            let id = if let Some(id) = shortcut(&table, &current, y, w, &mut stats) {
                id
            } else {
                stats.full_candidates += 1;
                let graph = if let Some(c) = cache.as_ref() {
                    if table.rows[sw as usize].mu.len() >= c.limits.min_mu_candidates {
                        stats.interval_attempts += 1;
                        let g = c.build(&table, y, w);
                        if g.is_none() {
                            stats.interval_build_aborts += 1;
                        }
                        g
                    } else {
                        None
                    }
                } else {
                    None
                };
                let cached = match (cache.as_mut(), graph.as_ref()) {
                    (Some(c), Some(g)) => c.lookup(g, &mut stats),
                    _ => None,
                };
                if let Some(id) = cached {
                    stats.interval_hits += 1;
                    id
                } else {
                    let h = recurrence(&table, y, w, s, &mut stats);
                    let id = index.intern(&mut table.pols, h);
                    if let (Some(c), Some(g)) = (cache.as_mut(), graph) {
                        c.insert(g, id);
                    }
                    id
                }
            };
            debug_assert_ne!(id, MISSING);
            current[y as usize] = id;
            let extremal = table.extremal(y, w);
            if extremal && id != 0 {
                nontrivial.push((y, id));
            }
            let gap = table.elms.lengths[w as usize] - table.elms.lengths[y as usize];
            // Non-extremal pairs with gap > 1 have mu=0 by the descent reduction
            // plus the degree bound. The ORIGINAL gap must be used here.
            if gap == 1 || (extremal && gap % 2 == 1) {
                let m = table.pols[id as usize].coeff(gap as i32 - 1);
                if m != 0 {
                    mu.push((y, m));
                }
            }
        }
        nontrivial.reverse();
        mu.reverse();
        table.rows[w as usize].nontrivial = nontrivial.into_boxed_slice();
        table.rows[w as usize].mu = mu.into_boxed_slice();
    }
    if let Some(c) = cache {
        stats.interval_classes = c.len() as u64;
        stats.interval_cache_bytes = c.bytes();
    }
    table.stats = stats;
    Ok(table)
}

fn skeleton(t: &CompactKlTable, w: u32, s: usize, n_pos: u32, covers: bool) -> Row {
    let mut below = vec![0u64; (w as usize + 64) / 64];
    let sw = t.elms.lft(w, s);
    let lw = t.elms.lengths[w as usize];
    let mut cv = Vec::new();
    for y in 0..=w {
        let ly = t.elms.lengths[y as usize];
        let yes = if y == 0 || y == w {
            true
        } else if ly == lw {
            false
        } else if lw + ly > n_pos {
            t.leq(t.elms.aw0[w as usize], t.elms.aw0[y as usize])
        } else {
            let sy = t.elms.lft(y, s);
            t.leq(if sy < y { sy } else { y }, sw)
        };
        if yes {
            below[y as usize / 64] |= 1u64 << (y % 64);
            if covers && ly + 1 == lw {
                cv.push(y);
            }
        }
    }
    Row {
        below: below.into_boxed_slice(),
        nontrivial: Box::new([]),
        mu: Box::new([]),
        covers: cv.into_boxed_slice(),
    }
}

fn shortcut(
    t: &CompactKlTable,
    cur: &[u32],
    y: u32,
    w: u32,
    stats: &mut CompactStats,
) -> Option<u32> {
    let iw = t.elms.inva[w as usize];
    let iy = t.elms.inva[y as usize];
    if iw < w || (iw == w && iy > y) {
        stats.inverse_hits += 1;
        return if iw == w {
            Some(cur[iy as usize])
        } else {
            t.pol_id(iy, iw)
        };
    }
    let left = t.left[w as usize] & !t.left[y as usize];
    if left != 0 {
        stats.left_hits += 1;
        return Some(cur[t.elms.lft(y, left.trailing_zeros() as usize) as usize]);
    }
    let right = t.right[w as usize] & !t.right[y as usize];
    if right != 0 {
        stats.right_hits += 1;
        return Some(cur[t.rft(y, right.trailing_zeros() as usize) as usize]);
    }
    if t.elms.lengths[w as usize] - t.elms.lengths[y as usize] <= 2 {
        stats.short_hits += 1;
        return Some(0);
    }
    None
}

fn recurrence(t: &CompactKlTable, y: u32, w: u32, s: usize, stats: &mut CompactStats) -> Laurent {
    stats.recurrences += 1;
    let sw = t.elms.lft(w, s);
    let sy = t.elms.lft(y, s);
    let mut h = t.pol(sy, sw).expect("recurrence base comparable").clone();
    if let Some(p) = t.pol(y, sw) {
        h += &p.shifted(2);
    }
    stats.legacy_z_candidates += u64::from(sw.saturating_sub(y));
    for &(z, m) in t.rows[sw as usize].mu.iter().rev() {
        if z < y {
            break;
        }
        stats.sparse_mu_candidates += 1;
        if t.left[z as usize] & (1u64 << s) == 0 || !t.leq(y, z) {
            continue;
        }
        let p = t.pol(y, z).expect("mu contribution comparable");
        let shift = (t.elms.lengths[w as usize] - t.elms.lengths[z as usize]) as i32;
        h -= &p.shift_scaled(shift, m);
        stats.contributing_mu_terms += 1;
    }
    h
}
