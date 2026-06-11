//! Full PyCox-semantics W-graphs for induced cells (Phase 2).
//!
//! TODO(phase-2 follow-up): this module is ~1050 lines, over the ~800-line house
//! limit (CLAUDE.md §5). The W-graph data type, the `from_relkl`/`to_relkl` sign-
//! flip transform, and `decompose` are cohesive but could be split into
//! `cellgraph/mod.rs` + `cellgraph/transform.rs` + `cellgraph/decompose.rs` in a
//! dedicated refactor commit. Deferred to keep Phase-2 review diffs reviewable.
//!
//! [`CellGraph`] is a **self-contained** W-graph: it owns its elements (as
//! canonical words plus hashable `CoxElm` identities) and does not reference a
//! full `KlTable`.  This makes it suitable for cells of groups too large to
//! enumerate (E7 has ≈2.9M elements; its individual cells are small).  It is the
//! Phase-2 counterpart of the Phase-1 [`WGraph`](crate::wgraph::WGraph), which is
//! keyed by `ElmIdx` into a full table and therefore cannot represent induced
//! cells.  Both types coexist; the Phase-1 `WGraph` serves the full-table path.
//!
//! Normative source: `pycox-ref/pycox_ref.py` class `wgraph` (≈9699–10051),
//! `klcellw0` (11971–11986) and `wgraphstarorbit` (11989–12010).  On any
//! discrepancy, the Python source wins.
//!
//! # The PyCox `mmat` / `mpols` representation
//!
//! A W-graph stores, for every relevant pair `(y, x)` of vertices and every
//! W-generator `s`, a coefficient `m^s_{x,y} ∈ A` (Laurent ring).  PyCox keeps
//! per-generator pools `mpols[s]` (each seeded `[0, 1]`) and a dictionary
//! `mmat[(y, x)]` whose value is a string `'c<i0>c<i1>…c<i_{rank-1}>'` pointing
//! into those pools (an empty field `'c'` means "no slot for this generator").
//!
//! We represent this as:
//! - [`CellGraph::mpols`]: `Vec<Vec<Laurent>>`, one pool per W-generator.
//! - [`CellGraph::mmat`]: `HashMap<(u32, u32), Vec<u32>>`; the value is a
//!   length-`rank` vector of pool indices, with [`NO_SLOT`] (`u32::MAX`) marking
//!   the empty `'c'` field.
//!
//! # The dict-path constructor and the length-parity sign flip
//!
//! [`CellGraph::from_relkl`] ports the PyCox constructor's `xset` (dict) path
//! (lines 9813–9883).  The defining transform is the **sign flip**
//! `m = −(−1)^(ℓ(y)+ℓ(x)) · pool[idx]` applied to every interned mu value, plus
//! the generator-bijection block (lines 9868–9878).  [`CellGraph::to_relkl`]
//! (`wgraphtoklmat`) is its exact inverse (`eps = −(−1)^(len X[i]+len X[j])`).

use std::collections::{HashMap, HashSet};

use crate::{
    element::{CoxElm, Gen, Word},
    group::CoxeterGroup,
    kl::scc::tarjan_scc,
    laurent::Laurent,
};

/// Sentinel for an empty per-generator slot (PyCox `'c'` with no index).
pub const NO_SLOT: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// RelKlInput — the P4 interface contract (= PyCox dict {'elms','klmat','mpols'})
// ---------------------------------------------------------------------------

/// Input/output contract shared between [`CellGraph`] and `relklpols` (Task P4).
///
/// This is the Rust analogue of the PyCox dictionary
/// `{'elms', 'klmat', 'mpols'}` produced by `wgraph.wgraphtoklmat()` and
/// consumed by the `wgraph(W, 1, dict, v)` dict-path constructor.
///
/// - `elms`: reduced words (in the group's own generator labels), increasing
///   length — the W-graph vertex set.
/// - `klmat`: a **lower-triangular** matrix of slots; `klmat[j][i]` (with
///   `i < j`) is the slot for the ordered pair `(j, i)`.  `None` ⇔ PyCox `'f'`
///   (no edge); `Some(SlotData)` ⇔ PyCox `'c0' + 'c<idx>'…` (the leading `'c0'`
///   is the placeholder KL-polynomial index, never used by the constructor).
/// - `mpols`: the mu pools, either per-generator ([`MuPools::PerGen`], as
///   produced by `wgraphtoklmat`) or a single global pool ([`MuPools::Global`],
///   as produced by `relklpols`).  Each pool starts `[0, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct RelKlInput {
    /// Vertex words, sorted by increasing length.
    pub elms: Vec<Word>,
    /// Lower-triangular slot matrix; `klmat[j]` has length `j` (entries for
    /// `i = 0..j`).  `klmat[j][i]` is the slot of the pair `(j, i)`.
    pub klmat: Vec<Vec<KlSlot>>,
    /// The mu pools (per-generator or single global).
    pub mpols: MuPools,
}

/// A single slot in [`RelKlInput::klmat`].  `None` is PyCox `'f'`.
pub type KlSlot = Option<SlotData>;

/// The per-generator (or single global) mu-pool indices of one filled slot.
///
/// In PyCox the slot string is `'c0' + 'c<i0>c<i1>…'`.  The leading `'c0'`
/// (placeholder KL index) is dropped; `mu` holds the remaining indices:
/// - length `rank` for [`MuPools::PerGen`] input (`wgraphtoklmat` output);
/// - length `1` for [`MuPools::Global`] input (`relklpols` output).
///
/// An index of [`NO_SLOT`] encodes the empty per-generator field PyCox writes as
/// `'c'` (no index).  In practice `wgraphtoklmat` writes `0` for empty
/// generators, so `NO_SLOT` is only needed for hand-built inputs that mirror the
/// constructor's `'c'` convention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotData {
    /// Pool indices, one per generator (`PerGen`) or a single index (`Global`).
    pub mu: Vec<u32>,
}

/// The mu pools carried by a [`RelKlInput`].
#[derive(Clone, Debug, PartialEq)]
pub enum MuPools {
    /// Per-generator pools — one `Vec<Laurent>` per W-generator, each seeded
    /// `[0, 1]`.  Produced by `wgraph.wgraphtoklmat()`.
    PerGen(Vec<Vec<Laurent>>),
    /// A single global pool, seeded `[0, 1]`.  Produced by `relklpols`.
    Global(Vec<Laurent>),
}

// ---------------------------------------------------------------------------
// CellGraph
// ---------------------------------------------------------------------------

/// A full PyCox-semantics W-graph that owns its elements.
///
/// See the [module docs](self) for the meaning of each field and the
/// representation of `mmat` / `mpols`.
#[derive(Clone, Debug)]
pub struct CellGraph {
    /// Vertex words.  After [`normalise`](CellGraph::normalise) these are sorted
    /// by length (stable), matching PyCox `sorted(key=len)`.
    pub x: Vec<Word>,
    /// Hashable identities, parallel to `x`: `Xrep` = `coxelm_sr` of each word's
    /// perm (the same convention as the element-table index keys).
    pub xrep: Vec<CoxElm>,
    /// Left-descent set of each vertex, parallel to `x`.
    pub isets: Vec<Vec<Gen>>,
    /// Per-generator mu pools, each seeded `[zero, one]`.
    pub mpols: Vec<Vec<Laurent>>,
    /// `mmat[(y, x)]` = length-`rank` vector of pool indices (one per
    /// generator); [`NO_SLOT`] marks an empty field.  Keys come from the
    /// constructor's mu block (`y > x`) **and** its generator-bijection block
    /// (`(y, s·y)`, where `s·y` may be larger or smaller than `y`).
    pub mmat: HashMap<(u32, u32), Vec<u32>>,
    /// Generator weights `L(s)`.
    pub weights: Vec<u32>,
}

impl CellGraph {
    /// Number of vertices.
    #[inline]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Whether the graph has no vertices.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// The rank (number of generators) of the underlying group.
    #[inline]
    fn rank(&self) -> usize {
        self.weights.len()
    }

    // -----------------------------------------------------------------------
    // from_relkl — the dict-path constructor (PyCox 9813–9883, equal params)
    // -----------------------------------------------------------------------

    /// Build a [`CellGraph`] from a [`RelKlInput`] (PyCox dict-path
    /// constructor, equal-parameter branch `uneq == False`, lines 9813–9883).
    ///
    /// `weights` must be all-1 (equal parameters); only that branch is ported.
    ///
    /// # Semantics
    ///
    /// `Isets` are the left descent sets of the vertex perms.  For every ordered
    /// pair `(y, x)` with `x < y` whose `klmat[y][x]` slot is filled, and every
    /// generator `s ∈ I(x) \ I(y)`, the slot's pool value is read:
    /// - [`MuPools::PerGen`]: index `slot.mu[s]`;
    /// - [`MuPools::Global`]: index `slot.mu[0]`;
    ///
    /// and, if real (`≠ NO_SLOT`) and non-zero/non-one-placeholder, interned via
    /// the **sign flip** `m = −(−1)^(ℓ(y)+ℓ(x)) · pool[idx]` into `mpols[s]`.
    ///
    /// In addition (lines 9868–9878), for every `y` and every `s ∉ I(y)` with
    /// `s·y` present in the vertex set, a generator-bijection entry is recorded
    /// at key `(y, idx(s·y))`: index `1` for generator `s`, index `0` for all
    /// others (i.e. the pool value `mpols[s][1] = 1`).
    pub fn from_relkl(g: &CoxeterGroup, weights: &[u32], rk: &RelKlInput) -> CellGraph {
        debug_assert!(
            weights.iter().all(|&w| w == 1),
            "from_relkl: only the equal-parameter branch is ported"
        );
        let rank = g.rank;
        let n = rk.elms.len();

        // Perms, lengths, Isets, Xrep.
        let perms: Vec<_> = rk.elms.iter().map(|w| g.word_to_perm(w)).collect();
        let ll: Vec<u32> = perms.iter().map(|p| g.perm_length(p)).collect();
        let isets: Vec<Vec<Gen>> = perms.iter().map(|p| g.left_descents(p)).collect();
        let xrep: Vec<CoxElm> = perms.iter().map(|p| p.coxelm_sr(&g.simple_root)).collect();

        // Index from coxelm → position, for the s·y lookups.
        let pos_of: HashMap<CoxElm, u32> = xrep
            .iter()
            .enumerate()
            .map(|(i, ce)| (ce.clone(), i as u32))
            .collect();

        // Per-generator pools, each seeded [0, 1].
        let mut nmues: Vec<Vec<Laurent>> = (0..rank)
            .map(|_| vec![Laurent::zero(), Laurent::one()])
            .collect();
        let mut mmat: HashMap<(u32, u32), Vec<u32>> = HashMap::new();

        for y in 0..n {
            // Mu block: x < y with a filled klmat slot.
            for x in 0..y {
                let Some(slot) = rk.klmat[y][x].as_ref() else {
                    continue;
                };
                // Build the length-rank slot vector for this (y, x) key.
                let mut row = vec![NO_SLOT; rank];
                let mut any_real = false;
                for s in 0..rank {
                    let s_gen = s as Gen;
                    // Only s ∈ I(x) \ I(y) contributes; otherwise leave NO_SLOT
                    // (PyCox 'c').
                    let in_xy = isets[x].contains(&s_gen) && !isets[y].contains(&s_gen);
                    if !in_xy {
                        continue;
                    }
                    // Pick the source pool index for this generator.
                    let src_idx = match &rk.mpols {
                        MuPools::PerGen(_) => slot.mu.get(s).copied().unwrap_or(NO_SLOT),
                        MuPools::Global(_) => slot.mu.first().copied().unwrap_or(NO_SLOT),
                    };
                    // PyCox: only intern when ms[?] != '' and != '0'
                    // (i.e. the slot index is real and not the zero-pool slot 0).
                    if src_idx == NO_SLOT || src_idx == 0 {
                        row[s] = 0; // PyCox 'c0'
                        continue;
                    }
                    let pool_val = match &rk.mpols {
                        MuPools::PerGen(pools) => &pools[s][src_idx as usize],
                        MuPools::Global(pool) => &pool[src_idx as usize],
                    };
                    // Sign flip by length parity.
                    let m = sign_flip(ll[y], ll[x], pool_val);
                    row[s] = intern(&mut nmues[s], m);
                    any_real = true;
                }
                if any_real {
                    mmat.insert((y as u32, x as u32), row);
                }
            }

            // Generator-bijection block (PyCox 9868–9878).
            for s in 0..rank {
                let s_gen = s as Gen;
                if isets[y].contains(&s_gen) {
                    continue;
                }
                // sy = s·y (left-multiply): coxelm of permgens[s] ∘ perm[y].
                let sy = g.permgens[s].then(&perms[y]).coxelm_sr(&g.simple_root);
                if let Some(&syi) = pos_of.get(&sy) {
                    // Value: c1 for generator s, c0 for all others.
                    let mut row = vec![0u32; rank];
                    row[s] = 1;
                    mmat.insert((y as u32, syi), row);
                }
            }
        }

        CellGraph {
            x: rk.elms.clone(),
            xrep,
            isets,
            mpols: nmues,
            mmat,
            weights: weights.to_vec(),
        }
    }

    // -----------------------------------------------------------------------
    // to_relkl — wgraphtoklmat (PyCox 9910–9939), the inverse of from_relkl
    // -----------------------------------------------------------------------

    /// Produce the [`RelKlInput`] (`wgraphtoklmat`) — the exact inverse of
    /// [`from_relkl`](CellGraph::from_relkl).
    ///
    /// For each pair `(j, i)` with `i < j` present in `mmat`, rebuilds the slot
    /// with the **inverse sign flip** `eps = −(−1)^(len X[i]+len X[j])`, interning
    /// `eps · mpols[s][idx]` into freshly seeded per-generator pools.  Empty
    /// generator fields (`NO_SLOT`) become pool index `0`.  The resulting
    /// `mpols` is always [`MuPools::PerGen`].
    pub fn to_relkl(&self, _g: &CoxeterGroup) -> RelKlInput {
        let rank = self.rank();
        let n = self.x.len();
        let lens: Vec<usize> = self.x.iter().map(|w| w.len()).collect();

        let mut mues: Vec<Vec<Laurent>> = (0..rank)
            .map(|_| vec![Laurent::zero(), Laurent::one()])
            .collect();
        let mut klmat: Vec<Vec<KlSlot>> = (0..n).map(|j| vec![None; j]).collect();

        for j in 0..n {
            for i in 0..j {
                let Some(row) = self.mmat.get(&(j as u32, i as u32)) else {
                    continue;
                };
                // eps = -(-1)^(len_i + len_j).
                let parity = (lens[i] + lens[j]) % 2;
                // -(-1)^p : p even → -1, p odd → +1.
                let eps_neg = parity == 0;
                let mut mu = vec![0u32; rank];
                for s in 0..rank {
                    let idx = row[s];
                    if idx == NO_SLOT {
                        mu[s] = 0; // PyCox empty field 'c'
                        continue;
                    }
                    let val = &self.mpols[s][idx as usize];
                    let m = if eps_neg { -val } else { val.clone() };
                    mu[s] = intern(&mut mues[s], m);
                }
                klmat[j][i] = Some(SlotData { mu });
            }
        }

        RelKlInput {
            elms: self.x.clone(),
            klmat,
            mpols: MuPools::PerGen(mues),
        }
    }

    // -----------------------------------------------------------------------
    // normalise — sort vertices by length (PyCox 9888–9909)
    // -----------------------------------------------------------------------

    /// Sort the vertex set by word length (stable), relabelling `xrep`, `isets`
    /// and `mmat` keys accordingly.  No-op (clone) when already sorted.
    ///
    /// Mirrors PyCox `wgraph.normalise`: `lx.sort(key=len)` is a *stable* sort,
    /// so vertices of equal length keep their relative order.
    pub fn normalise(&self, _g: &CoxeterGroup) -> CellGraph {
        let n = self.x.len();
        // Stable sort of indices by word length.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| self.x[i].len());

        // If already in this order, clone (PyCox returns self).
        if order.iter().enumerate().all(|(new, &old)| new == old) {
            return self.clone();
        }

        // new position of old index: inv[old] = new.
        let mut inv = vec![0u32; n];
        for (new, &old) in order.iter().enumerate() {
            inv[old] = new as u32;
        }

        let x: Vec<Word> = order.iter().map(|&i| self.x[i].clone()).collect();
        let xrep: Vec<CoxElm> = order.iter().map(|&i| self.xrep[i].clone()).collect();
        let isets: Vec<Vec<Gen>> = order.iter().map(|&i| self.isets[i].clone()).collect();
        let mmat: HashMap<(u32, u32), Vec<u32>> = self
            .mmat
            .iter()
            .map(|(&(a, b), v)| ((inv[a as usize], inv[b as usize]), v.clone()))
            .collect();

        CellGraph {
            x,
            xrep,
            isets,
            mpols: self.mpols.clone(),
            mmat,
            weights: self.weights.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // cell_w0 — klcellw0 (PyCox 11971–11986)
    // -----------------------------------------------------------------------

    /// Right-multiply every vertex by the longest element `w₀` (`klcellw0`).
    ///
    /// If the cell is `w₀`-stable (the first new perm's `coxelm` is already among
    /// the old `xrep`), returns `self` unchanged.  Otherwise recomputes `Isets`
    /// from the new perms, **transposes** the `mmat` keys `(y, x) → (x, y)`,
    /// reuses `mpols`, and finally [`normalise`](CellGraph::normalise)s.
    pub fn cell_w0(&self, g: &CoxeterGroup) -> CellGraph {
        let w0 = g.longest_perm();
        // pc = perms of the current vertices; np = pc · w0.
        let pc: Vec<_> = self.x.iter().map(|w| g.word_to_perm(w)).collect();
        let np: Vec<_> = pc.iter().map(|p| p.then(w0)).collect();

        // w0-stable test: np[0]'s coxelm already among old xrep.
        let np0_ce = np[0].coxelm_sr(&g.simple_root);
        let old_set: HashSet<&CoxElm> = self.xrep.iter().collect();
        if old_set.contains(&np0_ce) {
            return self.clone();
        }

        let x: Vec<Word> = np.iter().map(|p| g.perm_to_word(p)).collect();
        let xrep: Vec<CoxElm> = np.iter().map(|p| p.coxelm_sr(&g.simple_root)).collect();
        let isets: Vec<Vec<Gen>> = np.iter().map(|p| g.left_descents(p)).collect();
        // Transpose mmat keys.
        let mmat: HashMap<(u32, u32), Vec<u32>> = self
            .mmat
            .iter()
            .map(|(&(a, b), v)| ((b, a), v.clone()))
            .collect();

        CellGraph {
            x,
            xrep,
            isets,
            mpols: self.mpols.clone(),
            mmat,
            weights: self.weights.clone(),
        }
        .normalise(g)
    }

    // -----------------------------------------------------------------------
    // star_orbit — wgraphstarorbit (PyCox 11989–12010)
    // -----------------------------------------------------------------------

    /// The orbit of this W-graph under the KL star operation
    /// (`wgraphstarorbit`).
    ///
    /// For each orbit member `l` returned by
    /// [`star_orbit_right`](crate::star::star_orbit_right) applied to the
    /// vertex perms, builds a new graph with the **same** `Isets`, `mmat` and
    /// `mpols` (the W-graph data is isomorphic; only the base set is relabelled),
    /// new `X`/`Xrep` from `l`, then [`normalise`](CellGraph::normalise)s.
    ///
    /// The first member is this cell itself, so the orbit always includes a
    /// normalised copy of `self`.
    pub fn star_orbit(&self, g: &CoxeterGroup) -> Vec<CellGraph> {
        let perms: Vec<_> = self.x.iter().map(|w| g.word_to_perm(w)).collect();
        let orbit = crate::star::star_orbit_right(g, &perms);

        orbit
            .into_iter()
            .map(|l| {
                let x: Vec<Word> = l.iter().map(|p| g.perm_to_word(p)).collect();
                let xrep: Vec<CoxElm> = l.iter().map(|p| p.coxelm_sr(&g.simple_root)).collect();
                CellGraph {
                    x,
                    xrep,
                    isets: self.isets.clone(),
                    mpols: self.mpols.clone(),
                    mmat: self.mmat.clone(),
                    weights: self.weights.clone(),
                }
                .normalise(g)
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // decompose — PyCox wgraph.decompose (9940–9971)
    // -----------------------------------------------------------------------

    /// Split this W-graph into its indecomposable components
    /// (`wgraph.decompose`).
    ///
    /// PyCox's `decompose` derives the directed reachability graph **directly
    /// from `mmat.keys()`**: each key `(y, x)` is an arrow `y → x`.  The
    /// strongly-connected components of that graph are the left cells.  We use
    /// the shared Tarjan routine (equivalent to PyCox's mutual-reachability
    /// closure) and restrict every field per component.
    ///
    /// Components are returned canonically: each component's vertices keep their
    /// `self`-order (already length-sorted after `normalise`), and the component
    /// list is sorted by first vertex word.
    pub fn decompose(&self, _g: &CoxeterGroup) -> Vec<CellGraph> {
        let n = self.x.len();
        if n == 0 {
            return vec![];
        }

        // Directed adjacency from mmat keys: (y, x) ⇒ arrow y → x.
        let mut adj_sets: Vec<HashSet<u32>> = vec![HashSet::new(); n];
        for &(y, x) in self.mmat.keys() {
            adj_sets[y as usize].insert(x);
        }
        let adj: Vec<Vec<u32>> = adj_sets
            .into_iter()
            .map(|s| s.into_iter().collect())
            .collect();

        let (comp_of, num_comp) = tarjan_scc(&adj, n);

        // Group positions by component.
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); num_comp];
        for v in 0..n {
            members[comp_of[v]].push(v);
        }

        let mut comps: Vec<CellGraph> = members.into_iter().map(|m| self.subgraph(&m)).collect();

        // Canonical order: by first vertex word.
        comps.sort_by(|a, b| a.x.first().cmp(&b.x.first()));
        comps
    }

    /// Build a sub-[`CellGraph`] on the given vertex positions (within `self`).
    ///
    /// Positions are kept in their `self`-order (sorted ascending here, which
    /// for a normalised graph is length order); `mmat` is restricted to keys with
    /// both endpoints inside the component and re-keyed to local positions.
    fn subgraph(&self, positions: &[usize]) -> CellGraph {
        let mut pos = positions.to_vec();
        pos.sort_unstable();

        let old_to_new: HashMap<u32, u32> = pos
            .iter()
            .enumerate()
            .map(|(new, &old)| (old as u32, new as u32))
            .collect();

        let x: Vec<Word> = pos.iter().map(|&p| self.x[p].clone()).collect();
        let xrep: Vec<CoxElm> = pos.iter().map(|&p| self.xrep[p].clone()).collect();
        let isets: Vec<Vec<Gen>> = pos.iter().map(|&p| self.isets[p].clone()).collect();

        let mut mmat: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (&(a, b), v) in &self.mmat {
            if let (Some(&na), Some(&nb)) = (old_to_new.get(&a), old_to_new.get(&b)) {
                mmat.insert((na, nb), v.clone());
            }
        }

        CellGraph {
            x,
            xrep,
            isets,
            mpols: self.mpols.clone(),
            mmat,
            weights: self.weights.clone(),
        }
    }

    /// The sorted set of vertex words — convenient for set-equality assertions.
    pub fn word_set(&self) -> Vec<Word> {
        let mut v = self.x.clone();
        v.sort();
        v
    }

    /// Restrict this W-graph to the given vertex positions, **preserving their
    /// given order**.
    ///
    /// This is the public counterpart of the private `subgraph` helper, used by
    /// the `klcells` size-tier pre-partition (PyCox 12236–12246): given a bucket
    /// `l` of positions sharing a left-cell invariant (right-descent set or
    /// generalised tau), it builds `wgraph(W, weights, [X[i] for i in l],
    /// var, [Isets[i] for i in l], {relabelled mmat}, mpols, [Xrep[i] for i in
    /// l])`.  `mmat` is restricted to keys with both endpoints inside `l` and
    /// re-keyed to local positions.
    ///
    /// Unlike the private `subgraph` helper the positions are NOT sorted: they
    /// keep their caller-supplied order, mirroring PyCox's `filter`-derived list
    /// `l`.  The subsequent [`decompose`](Self::decompose) derives its
    /// components from `mmat` keys, so the vertex order only affects the
    /// within-component labelling, which is re-canonicalized downstream.
    pub fn restrict(&self, positions: &[usize]) -> CellGraph {
        let old_to_new: HashMap<u32, u32> = positions
            .iter()
            .enumerate()
            .map(|(new, &old)| (old as u32, new as u32))
            .collect();

        let x: Vec<Word> = positions.iter().map(|&p| self.x[p].clone()).collect();
        let xrep: Vec<CoxElm> = positions.iter().map(|&p| self.xrep[p].clone()).collect();
        let isets: Vec<Vec<Gen>> = positions.iter().map(|&p| self.isets[p].clone()).collect();

        let mut mmat: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (&(a, b), v) in &self.mmat {
            if let (Some(&na), Some(&nb)) = (old_to_new.get(&a), old_to_new.get(&b)) {
                mmat.insert((na, nb), v.clone());
            }
        }

        CellGraph {
            x,
            xrep,
            isets,
            mpols: self.mpols.clone(),
            mmat,
            weights: self.weights.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// `m = −(−1)^(ℓy+ℓx) · p`.  Even parity ⇒ `−p`; odd parity ⇒ `+p`.
#[inline]
fn sign_flip(ly: u32, lx: u32, p: &Laurent) -> Laurent {
    if (ly + lx) % 2 == 0 {
        -p
    } else {
        p.clone()
    }
}

/// Intern `m` into `pool`, returning its index (append if absent).
#[inline]
fn intern(pool: &mut Vec<Laurent>, m: Laurent) -> u32 {
    if let Some(i) = pool.iter().position(|q| *q == m) {
        i as u32
    } else {
        pool.push(m);
        (pool.len() - 1) as u32
    }
}

// ---------------------------------------------------------------------------
// Bootstrap / testing utility — build a RelKlInput directly from a full table
// ---------------------------------------------------------------------------

/// Build a [`RelKlInput`] (in [`MuPools::PerGen`] form) for a set of full-table
/// elements, mirroring exactly what [`CellGraph::to_relkl`] (`wgraphtoklmat`)
/// would produce for that set of vertices.
///
/// This is the **bootstrap bridge** used to validate `relklpols` (Task P4)
/// without depending on the not-yet-built `klcells` driver: given a group's full
/// [`KlTable`](crate::kl::KlTable) and a subset of element indices (typically a
/// left cell of [`CellData::lcells`](crate::kl::CellData)), it produces the same
/// `RelKlInput` that the cell's W-graph would yield.
///
/// The elements are ordered by `(length, canonical word)` — increasing length —
/// matching the `RelKlInput::elms` contract.  For each Bruhat-ordered pair
/// `(j, i)` (`i < j` in this order) with a non-zero table mu, the slot stores the
/// per-generator pool indices of `eps · μ^s_{lo,hi}` where
/// `eps = −(−1)^(ℓ_i+ℓ_j)` — exactly the value `wgraphtoklmat` stores, so
/// `from_relkl(input)` recovers a [`CellGraph`] whose `mpols` m-values equal the
/// raw table mu values.
pub fn relkl_input_from_table(
    g: &CoxeterGroup,
    t: &crate::kl::KlTable,
    cell: &[crate::element::ElmIdx],
) -> RelKlInput {
    use crate::element::ElmIdx;

    let rank = g.rank;
    // Order cell elements by (length, canonical word) — increasing length.
    let mut elems: Vec<ElmIdx> = cell.to_vec();
    elems.sort_by(|&a, &b| {
        let wa = &t.elms.elms[a as usize];
        let wb = &t.elms.elms[b as usize];
        wa.len().cmp(&wb.len()).then_with(|| wa.cmp(wb))
    });
    let words: Vec<Word> = elems
        .iter()
        .map(|&e| t.elms.elms[e as usize].clone())
        .collect();
    let lens: Vec<usize> = words.iter().map(|w| w.len()).collect();

    let n = elems.len();
    let mut mues: Vec<Vec<Laurent>> = (0..rank)
        .map(|_| vec![Laurent::zero(), Laurent::one()])
        .collect();
    let mut klmat: Vec<Vec<KlSlot>> = (0..n).map(|j| vec![None; j]).collect();

    for j in 0..n {
        for i in 0..j {
            // Bruhat-ordered (lo < hi) by canonical index.
            let (lo, hi) = {
                let a = elems[i];
                let b = elems[j];
                if a < b {
                    (a, b)
                } else {
                    (b, a)
                }
            };
            if !t.bruhat_leq(lo, hi) {
                continue;
            }
            let eps_neg = (lens[i] + lens[j]) % 2 == 0; // eps = -(-1)^p
            let mut mu = vec![0u32; rank];
            let mut any = false;
            for s in 0..rank {
                let muval = t.mu(s, lo, hi);
                if muval.is_zero() {
                    continue;
                }
                let m = if eps_neg { -&muval } else { muval };
                mu[s] = intern(&mut mues[s], m);
                any = true;
            }
            if any {
                klmat[j][i] = Some(SlotData { mu });
            }
        }
    }

    RelKlInput {
        elms: words,
        klmat,
        mpols: MuPools::PerGen(mues),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        element::ElmIdx,
        kl::{cells::CellData, klpolynomials_seq, KlOpts, KlTable},
    };
    use std::collections::BTreeSet;

    fn build_table(ty: &str) -> (CoxeterGroup, KlTable) {
        let group = CoxeterGroup::from_type(ty).unwrap();
        let opts = KlOpts::equal(group.rank);
        let table = klpolynomials_seq(&group, &opts).unwrap();
        (group, table)
    }

    /// Helper: the canonical word-set of every left cell of a full table.
    fn lcell_word_sets(t: &KlTable) -> BTreeSet<Vec<Word>> {
        let cd = CellData::from_table(t);
        cd.lcells
            .iter()
            .map(|cell| {
                let mut ws: Vec<Word> = cell
                    .iter()
                    .map(|&e| t.elms.elms[e as usize].clone())
                    .collect();
                ws.sort();
                ws
            })
            .collect()
    }

    fn graph_word_sets(graphs: &[CellGraph]) -> BTreeSet<Vec<Word>> {
        graphs.iter().map(|gph| gph.word_set()).collect()
    }

    // -----------------------------------------------------------------------
    // Test 1: round-trip idempotence
    // -----------------------------------------------------------------------
    /// Build a CellGraph from one A3 cell, then `to_relkl → from_relkl` again;
    /// the second CellGraph must be identical (same X, Xrep, Isets, mmat, mpols)
    /// to the first.  After one normalising pass the round-trip is the identity.
    #[test]
    fn from_relkl_roundtrip() {
        let (g, t) = build_table("A3");
        let cd = CellData::from_table(&t);
        let weights = vec![1u32; g.rank];

        for (ci, cell) in cd.lcells.iter().enumerate() {
            let input = relkl_input_from_table(&g, &t, cell);
            let cg1 = CellGraph::from_relkl(&g, &weights, &input);

            // Round-trip: to_relkl then from_relkl.
            let back = cg1.to_relkl(&g);
            let cg2 = CellGraph::from_relkl(&g, &weights, &back);

            assert_eq!(cg1.x, cg2.x, "cell {ci}: X mismatch after round-trip");
            assert_eq!(cg1.xrep, cg2.xrep, "cell {ci}: Xrep mismatch");
            assert_eq!(cg1.isets, cg2.isets, "cell {ci}: Isets mismatch");
            assert_eq!(cg1.mpols, cg2.mpols, "cell {ci}: mpols mismatch");
            assert_eq!(cg1.mmat, cg2.mmat, "cell {ci}: mmat mismatch");
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: decompose the full group → left cells (A3 + B3)
    // -----------------------------------------------------------------------
    /// Build one CellGraph over ALL of A3 (resp. B3) from the full table mu
    /// data; decompose() vertex-word-sets must equal the full-table lcells.
    #[test]
    fn decompose_full_group_a3() {
        check_decompose_full_group("A3");
    }

    #[test]
    fn decompose_full_group_b3() {
        check_decompose_full_group("B3");
    }

    fn check_decompose_full_group(ty: &str) {
        let (g, t) = build_table(ty);
        let weights = vec![1u32; g.rank];
        let all: Vec<ElmIdx> = (0..t.n() as u32).collect();

        let input = relkl_input_from_table(&g, &t, &all);
        let cg = CellGraph::from_relkl(&g, &weights, &input);
        let comps = cg.decompose(&g);

        let got = graph_word_sets(&comps);
        let want = lcell_word_sets(&t);

        assert_eq!(
            got,
            want,
            "{ty}: decompose() word-sets must equal full-table lcells \
             ({} comps vs {} cells)",
            comps.len(),
            want.len()
        );
        // Element conservation.
        let total: usize = comps.iter().map(|c| c.len()).sum();
        assert_eq!(total, t.n(), "{ty}: components must cover all elements");
    }

    // -----------------------------------------------------------------------
    // Test 3: cell_w0 — multiply by w0 (A3)
    // -----------------------------------------------------------------------
    /// For each A3 full-table cell graph, `cell_w0` returns a graph whose
    /// element set equals the full-table cell containing (first element)·w0.
    /// w0-stable cells return self (element set unchanged).
    #[test]
    fn cell_w0_a3() {
        let (g, t) = build_table("A3");
        let cd = CellData::from_table(&t);
        let weights = vec![1u32; g.rank];
        let w0 = g.longest_perm().clone();

        // Map coxelm → which lcell word-set it belongs to.
        let cell_sets = lcell_word_sets(&t);

        for (ci, cell) in cd.lcells.iter().enumerate() {
            let input = relkl_input_from_table(&g, &t, cell);
            let cg = CellGraph::from_relkl(&g, &weights, &input);

            let cg0 = cg.cell_w0(&g);
            let got = cg0.word_set();

            // The expected set: each element of the original cell, right-mult
            // by w0, canonicalised.
            let mut expected: Vec<Word> =
                cg.x.iter()
                    .map(|w| {
                        let p = g.word_to_perm(w).then(&w0);
                        g.perm_to_word(&p)
                    })
                    .collect();
            expected.sort();

            assert_eq!(
                got, expected,
                "cell {ci}: cell_w0 element set must be cell·w0"
            );
            // The result must itself be a full-table left cell.
            assert!(
                cell_sets.contains(&got),
                "cell {ci}: cell_w0 result is not a full-table left cell"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: star_orbit consistency (A3)
    // -----------------------------------------------------------------------
    /// `star_orbit` of a cell graph: element sets match `star_orbit_right` of the
    /// cell perms; all members share `mmat`/`mpols` (graph-isomorphic data).
    #[test]
    fn star_orbit_consistency_a3() {
        let (g, t) = build_table("A3");
        let cd = CellData::from_table(&t);
        let weights = vec![1u32; g.rank];

        for (ci, cell) in cd.lcells.iter().enumerate() {
            let input = relkl_input_from_table(&g, &t, cell);
            let cg = CellGraph::from_relkl(&g, &weights, &input);

            let cg_orbit = cg.star_orbit(&g);

            // Reference: star_orbit_right on the cell perms.
            let perms: Vec<_> = cg.x.iter().map(|w| g.word_to_perm(w)).collect();
            let ref_orbit = crate::star::star_orbit_right(&g, &perms);

            assert_eq!(
                cg_orbit.len(),
                ref_orbit.len(),
                "cell {ci}: orbit size mismatch"
            );

            // Compare element sets (as coxelm sets) between the two orbits.
            let cg_keys: BTreeSet<Vec<Vec<u32>>> = cg_orbit
                .iter()
                .map(|m| {
                    let mut k: Vec<Vec<u32>> = m.xrep.iter().map(|ce| ce.0.to_vec()).collect();
                    k.sort();
                    k
                })
                .collect();
            let ref_keys: BTreeSet<Vec<Vec<u32>>> = ref_orbit
                .iter()
                .map(|members| {
                    let mut k: Vec<Vec<u32>> = members
                        .iter()
                        .map(|p| p.coxelm_sr(&g.simple_root).0.to_vec())
                        .collect();
                    k.sort();
                    k
                })
                .collect();
            assert_eq!(
                cg_keys, ref_keys,
                "cell {ci}: star_orbit element sets differ from star_orbit_right"
            );

            // All members carry the SAME graph data: `mpols` verbatim, and an
            // isomorphic `mmat`.  PyCox passes Isets/mmat/mpols unchanged into
            // each orbit member, then `.normalise()` re-sorts the (relabelled)
            // base set — so the mmat *keys* may be permuted, but the multiset of
            // mmat *values* and the count of edges are invariant.
            let base_values = mmat_value_multiset(&cg);
            for (mi, m) in cg_orbit.iter().enumerate() {
                assert_eq!(m.mpols, cg.mpols, "cell {ci} member {mi}: mpols differ");
                assert_eq!(
                    m.mmat.len(),
                    cg.mmat.len(),
                    "cell {ci} member {mi}: mmat edge count differs"
                );
                assert_eq!(
                    mmat_value_multiset(m),
                    base_values,
                    "cell {ci} member {mi}: mmat value multiset differs"
                );
            }
        }
    }

    /// The sorted multiset of `mmat` values (slot index vectors).  Invariant
    /// under the key-relabelling performed by `normalise`.
    fn mmat_value_multiset(cg: &CellGraph) -> Vec<Vec<u32>> {
        let mut v: Vec<Vec<u32>> = cg.mmat.values().cloned().collect();
        v.sort();
        v
    }

    // -----------------------------------------------------------------------
    // Test 5: sign-flip pinning (B3) — BOTH parities, explicit values
    // -----------------------------------------------------------------------
    /// Pin the length-parity sign flip `m = −(−1)^(ℓy+ℓx)·pool` for BOTH parities.
    ///
    /// In *genuine* equal-parameter cell W-graphs every real-mu pair has odd
    /// `ℓy+ℓx` (a degree/parity fact — verified against the PyCox oracle for B3:
    /// all mu-block entries have parity 1), so the even-parity branch never fires
    /// on real data.  To exercise both signs we hand-build a `RelKlInput` over a
    /// chosen set of real B3 elements and inject a known pool value into one
    /// even-parity slot and one odd-parity slot.
    ///
    /// ## Chosen B3 pairs (documented)
    ///
    /// We search the B3 element table for two ordered pairs `(y, x)` with
    /// `x` Bruhat-below `y`, some generator `s ∈ I(x) \ I(y)`, and:
    /// - an **odd** pair: `ℓ(y)+ℓ(x)` odd  ⇒ flip `+1` ⇒ stored m = `+pool`;
    /// - an **even** pair: `ℓ(y)+ℓ(x)` even ⇒ flip `−1` ⇒ stored m = `−pool`.
    ///
    /// The injected pool value is `POOL = v⁻¹ + v` (a non-trivial Laurent), so a
    /// sign error is unambiguous.
    #[test]
    fn sign_flip_pinning_b3() {
        let (g, t) = build_table("B3");
        let weights = vec![1u32; g.rank];
        let rank = g.rank;

        // A distinctive pool value: v^{-1} + v.
        let pool_val = Laurent::from_coeffs(-1, vec![1, 0, 1]);

        // Find an even-parity and an odd-parity (y, x, s) triple from the B3
        // element table with s ∈ I(x)\I(y) and x ≤_Bruhat y.
        let n = t.n();
        let mut even_triple: Option<(ElmIdx, ElmIdx, usize)> = None;
        let mut odd_triple: Option<(ElmIdx, ElmIdx, usize)> = None;
        'outer: for y in 0..n as u32 {
            let py = g.word_to_perm(&t.elms.elms[y as usize]);
            let iy = g.left_descents(&py);
            let ly = t.elms.lengths[y as usize] as usize;
            for x in 0..y {
                if !t.bruhat_leq(x, y) {
                    continue;
                }
                let px = g.word_to_perm(&t.elms.elms[x as usize]);
                let ix = g.left_descents(&px);
                let lx = t.elms.lengths[x as usize] as usize;
                // pick s ∈ I(x) \ I(y).
                let s = (0..rank).find(|&s| {
                    let sg = s as Gen;
                    ix.contains(&sg) && !iy.contains(&sg)
                });
                let Some(s) = s else { continue };
                let parity = (ly + lx) % 2;
                if parity == 0 && even_triple.is_none() {
                    even_triple = Some((y, x, s));
                } else if parity == 1 && odd_triple.is_none() {
                    odd_triple = Some((y, x, s));
                }
                if even_triple.is_some() && odd_triple.is_some() {
                    break 'outer;
                }
            }
        }
        let (ye, xe, se) = even_triple.expect("B3: no even-parity (y,x,s) found");
        let (yo, xo, so) = odd_triple.expect("B3: no odd-parity (y,x,s) found");

        // Build a CellGraph for each pair as a 2-vertex graph [x, y] (x shorter
        // or equal; from_relkl iterates x<y by position with x = position 0).
        let check = |hi: ElmIdx, lo: ElmIdx, s: usize, want_neg: bool| {
            // Vertices in increasing length: position 0 = lo, position 1 = hi.
            let elms = vec![
                t.elms.elms[lo as usize].clone(),
                t.elms.elms[hi as usize].clone(),
            ];
            // One global pool [0, 1, POOL]; a Global slot carries a SINGLE index
            // that applies to whichever generator s ∈ I(x)\I(y) is active.
            let global = vec![Laurent::zero(), Laurent::one(), pool_val.clone()];
            // Position (j=1, i=0) carries the injected slot (single global index).
            let klmat: Vec<Vec<KlSlot>> = vec![vec![], vec![Some(SlotData { mu: vec![2] })]];
            let input = RelKlInput {
                elms,
                klmat,
                mpols: MuPools::Global(global),
            };
            let cg = CellGraph::from_relkl(&g, &weights, &input);
            // The stored m for generator s on key (1, 0).
            let row = cg
                .mmat
                .get(&(1, 0))
                .expect("expected mmat entry for the injected pair");
            let idx = row[s];
            assert_ne!(idx, NO_SLOT, "generator {s} slot should be filled");
            let stored = &cg.mpols[s][idx as usize];
            let expected = if want_neg {
                -&pool_val
            } else {
                pool_val.clone()
            };
            assert_eq!(
                *stored, expected,
                "sign flip: want_neg={want_neg}, stored={stored:?}, expected={expected:?}"
            );
        };

        // Even parity ⇒ flip = -1 ⇒ stored = -POOL.
        check(ye, xe, se, true);
        // Odd parity ⇒ flip = +1 ⇒ stored = +POOL.
        check(yo, xo, so, false);
    }
}
