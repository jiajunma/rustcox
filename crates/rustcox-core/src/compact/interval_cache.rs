//! Optional in-run cache for ordinary equal-parameter KL polynomials.
//!
//! Keys and WL signatures are FILTERS, never certificates. Every reuse checks
//! a full level-preserving directed graph isomorphism. Any construction/search
//! limit is a cache miss. No assumption is made for unequal/relative KL.
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::{CompactKlTable, CompactStats};

#[derive(Clone, Debug)]
pub struct IntervalLimits {
    pub max_vertices: usize,
    pub max_walk_edges: usize,
    pub max_search_nodes: usize,
    pub max_bytes: usize,
    pub min_mu_candidates: usize,
}

impl Default for IntervalLimits {
    fn default() -> Self {
        Self {
            max_vertices: 64,
            max_walk_edges: 512,
            max_search_nodes: 10_000,
            max_bytes: 8 * 1024 * 1024,
            min_mu_candidates: 32,
        }
    }
}

#[derive(Clone)]
pub(super) struct SmallGraph {
    levels: Vec<u32>,
    up: Vec<u64>,
    down: Vec<u64>,
    sig: Vec<u64>,
    key: u64,
}

fn hash<T: Hash>(v: &T) -> u64 {
    let mut h = DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

impl SmallGraph {
    fn new(levels: Vec<u32>, up: Vec<u64>, down: Vec<u64>) -> Self {
        let n = levels.len();
        let mut sig: Vec<u64> = (0..n)
            .map(|i| hash(&(levels[i], up[i].count_ones(), down[i].count_ones())))
            .collect();
        // Fixed round count makes signatures comparable across graphs.
        for _ in 0..3 {
            sig = (0..n)
                .map(|i| {
                    let mut us = Vec::new();
                    let mut ds = Vec::new();
                    for (j, &sj) in sig.iter().enumerate() {
                        if up[i] & (1u64 << j) != 0 { us.push(sj); }
                        if down[i] & (1u64 << j) != 0 { ds.push(sj); }
                    }
                    us.sort_unstable();
                    ds.sort_unstable();
                    hash(&(sig[i], us, ds))
                })
                .collect();
        }
        let mut profile = sig.clone();
        profile.sort_unstable();
        let key = hash(&(n, profile));
        Self { levels, up, down, sig, key }
    }

    fn accounted_bytes(&self) -> usize {
        // 28*n payload plus room for Vec/entry headers and hash-table growth.
        self.levels.len() * 32 + 512
    }
}

pub(super) struct IntervalCache {
    pub limits: IntervalLimits,
    entries: Vec<(SmallGraph, u32)>,
    buckets: HashMap<u64, Vec<usize>>,
    bytes: usize,
}

impl IntervalCache {
    pub fn new(limits: IntervalLimits) -> Self {
        Self { limits, entries: Vec::new(), buckets: HashMap::new(), bytes: 0 }
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn bytes(&self) -> usize { self.bytes }

    pub fn build(&self, t: &CompactKlTable, bottom: u32, top: u32) -> Option<SmallGraph> {
        if self.limits.max_bytes == 0 || self.limits.max_vertices < 2 { return None; }
        let cap = self.limits.max_vertices.min(64);
        let mut members = vec![top];
        let mut cursor = 0;
        let mut work = 0;
        while cursor < members.len() {
            let z = members[cursor];
            cursor += 1;
            if z == bottom { continue; }
            for &a in t.rows[z as usize].covers.iter() {
                work += 1;
                if work > self.limits.max_walk_edges { return None; }
                if t.leq(bottom, a) && !members.contains(&a) {
                    if members.len() == cap { return None; }
                    members.push(a);
                }
            }
        }
        members.sort_unstable();
        if members.first().copied() != Some(bottom) { return None; }
        let n = members.len();
        let base = t.elms.lengths[bottom as usize];
        let levels = members.iter().map(|&a| t.elms.lengths[a as usize] - base).collect();
        let mut up = vec![0; n];
        let mut down = vec![0; n];
        for (j, &b) in members.iter().enumerate() {
            for &a in t.rows[b as usize].covers.iter() {
                if let Ok(i) = members.binary_search(&a) {
                    up[i] |= 1u64 << j;
                    down[j] |= 1u64 << i;
                }
            }
        }
        Some(SmallGraph::new(levels, up, down))
    }

    pub fn lookup(&self, g: &SmallGraph, stats: &mut CompactStats) -> Option<u32> {
        let candidates = self.buckets.get(&g.key)?;
        let mut budget = self.limits.max_search_nodes;
        for &i in candidates {
            let (other, id) = &self.entries[i];
            match exact_isomorphism(g, other, &mut budget) {
                Some(true) => return Some(*id),
                Some(false) => {},
                None => { stats.interval_search_aborts += 1; return None; },
            }
        }
        None
    }

    pub fn insert(&mut self, g: SmallGraph, id: u32) {
        let cost = g.accounted_bytes();
        if cost > self.limits.max_bytes.saturating_sub(self.bytes) { return; }
        self.buckets.entry(g.key).or_default().push(self.entries.len());
        self.entries.push((g, id));
        self.bytes += cost;
    }
}

/// Returning true requires a complete bijection preserving all directed edges
/// AND all nonedges, and explicitly preserving levels (hashes may collide).
fn exact_isomorphism(a: &SmallGraph, b: &SmallGraph, budget: &mut usize) -> Option<bool> {
    let n = a.levels.len();
    if n != b.levels.len() { return Some(false); }
    let candidates: Vec<u64> = (0..n)
        .map(|i| (0..n).fold(0, |mask, j| {
            if a.levels[i] == b.levels[j]
                && a.sig[i] == b.sig[j]
                && a.up[i].count_ones() == b.up[j].count_ones()
                && a.down[i].count_ones() == b.down[j].count_ones()
            { mask | (1u64 << j) } else { mask }
        }))
        .collect();
    if candidates.contains(&0) { return Some(false); }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| candidates[i].count_ones());
    let mut mapping = vec![0usize; n];
    let ctx = Search { a, b, candidates: &candidates, order: &order };
    search(&ctx, 0, 0, &mut mapping, budget)
}

struct Search<'a> {
    a: &'a SmallGraph,
    b: &'a SmallGraph,
    candidates: &'a [u64],
    order: &'a [usize],
}

fn search(ctx: &Search<'_>, depth: usize, used: u64, mapping: &mut [usize], budget: &mut usize) -> Option<bool> {
    if depth == ctx.order.len() { return Some(true); }
    let i = ctx.order[depth];
    let mut cands = ctx.candidates[i] & !used;
    while cands != 0 {
        if *budget == 0 { return None; }
        *budget -= 1;
        let j = cands.trailing_zeros() as usize;
        cands &= cands - 1;
        let consistent = ctx.order[..depth].iter().all(|&k| {
            let mk = mapping[k];
            ((ctx.a.up[i] >> k) & 1) == ((ctx.b.up[j] >> mk) & 1)
                && ((ctx.a.down[i] >> k) & 1) == ((ctx.b.down[j] >> mk) & 1)
        });
        if !consistent { continue; }
        mapping[i] = j;
        match search(ctx, depth + 1, used | (1u64 << j), mapping, budget) {
            Some(true) => return Some(true),
            Some(false) => {},
            None => return None,
        }
    }
    Some(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_and_zero_budget() {
        let a = SmallGraph::new(vec![0, 1, 1, 2], vec![6, 8, 8, 0], vec![0, 1, 1, 6]);
        let b = a.clone();
        assert_eq!(exact_isomorphism(&a, &b, &mut 100), Some(true));
        assert_eq!(exact_isomorphism(&a, &b, &mut 0), None);
    }

    #[test]
    fn collisions_never_certify_different_edges() {
        let mut a = SmallGraph::new(vec![0, 1, 1, 2], vec![6, 8, 8, 0], vec![0, 1, 1, 6]);
        let mut b = SmallGraph::new(vec![0, 1, 1, 2], vec![2, 8, 8, 0], vec![0, 1, 0, 6]);
        a.sig.fill(0);
        b.sig.fill(0);
        assert_eq!(exact_isomorphism(&a, &b, &mut 100), Some(false));
    }
}
