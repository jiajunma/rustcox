//! Relative Kazhdan–Lusztig polynomials for parabolic induction (Task P4).
//!
//! Exact port of PyCox `relklpols` (`pycox-ref/pycox_ref.py` 10496–10773) and
//! `relmue` (10483–10494), **equal parameters only**.  On any discrepancy the
//! Python source wins.  The normative extraction is
//! `docs/superpowers/plans/2026-06-11-pycox-relklpols-notes.md`.
//!
//! Given a Coxeter group `W`, a parabolic subgroup `W1 = W_J ⊂ W`, and a left
//! cell (or union of left cells) `C` of `W1` described by a [`RelKlInput`]
//! (`cell1`), [`relklpols`] computes the relative KL polynomials of the induced
//! set `X1·C`, where `X1` is the set of minimal-length left coset
//! representatives of `W1` in `W`.  By Geck's induction theorem, `X1·C` is a
//! union of left cells of `W`.  The output [`RelKlOutput::input`] is a
//! [`RelKlInput`] in [`MuPools::Global`] form, ready to feed into
//! [`CellGraph::from_relkl`](crate::cellgraph::CellGraph::from_relkl).
//!
//! # The five index spaces
//!
//! The recursion juggles five distinct index spaces.  We keep them straight with
//! named type aliases and a disciplined naming convention:
//!
//! | space            | meaning                                   | alias / var |
//! |------------------|-------------------------------------------|-------------|
//! | W-generator      | a simple generator of `W`                 | `s` (`Gen`) |
//! | coset index      | position in `X1` (coset reps)             | `x`, `y` (`Cx`) |
//! | cell index       | position in `cell1.elms` (elements of `C`)| `u`, `v` (`Cu`) |
//! | flat `ap` index  | position in the induced set `X·C`         | `u32` |
//! | W1 element       | a perm of `W1` in `W1`'s own root system  | `p1[u]` |
//!
//! The W1-local generator space appears only transiently in the `lft1` lookups;
//! we resolve it to W-generator indices at the boundary (see below).
//!
//! # The `lft` / `lft1` keying convention (the subtle part)
//!
//! [`Lft`] encodes left-multiplication of a coset rep by a W-generator:
//! - [`Lft::In(x)`] — `s·X1[x]` stays in `X1` at coset index `x`;
//! - [`Lft::Out(t)`] — `s·X1[x]` leaves `X1`; it equals `X1[x]·t'` for a unique
//!   W1-generator `t'`, and `t` is the **W-generator index** `J[t']` (i.e. the
//!   *global* generator `gen_map[t']`).
//!
//! PyCox encodes this case as the integer `-t-1` and keys the `lft1` dictionary
//! by `J[t]` (the W-index).  We adopt that same W-index convention: `lft1` is a
//! `Vec` indexed by W-generator (length `W.rank`), with only the `J`-entries
//! populated, so `lft1[t]` for a [`Lft::Out(t)`] payload works directly with no
//! local/global conversion.  This matches the PyCox lookups `lft1[-1-sx]` and
//! `lft1[t]` (where `t = -1-sx`) verbatim.

use std::collections::HashMap;

use crate::{
    bruhat,
    cellgraph::{KlSlot, MuPools, RelKlInput, SlotData},
    element::{Gen, Perm, Word},
    group::CoxeterGroup,
    laurent::Laurent,
    parabolic::{red_left_coset_reps, Parabolic},
};

use super::relkl_recur::{
    compute_h, diag_block_mu, intern, relmue, CaseBCtx, Cu, Cx, Lft, SlotState,
};

// ---------------------------------------------------------------------------
// Public options + output
// ---------------------------------------------------------------------------

/// Options for [`relklpols`].  A placeholder for future parallelism knobs
/// (Task P6); only [`Default`] is meaningful today.
#[derive(Clone, Debug, Default)]
pub struct RelKlOpts {
    // Reserved for future use (e.g. `threads: usize`).  Kept private so adding
    // fields is non-breaking.
    _private: (),
}

/// Output of [`relklpols`].
///
/// `input` is the [`RelKlInput`] contract consumed by
/// [`CellGraph::from_relkl`](crate::cellgraph::CellGraph::from_relkl): its
/// `elms` are the induced-set canonical words sorted by **length** (stable),
/// `klmat` is the flat strict-lower-triangular matrix with single-`Global`-index
/// [`SlotData`]s, and `mpols` is [`MuPools::Global`] (the `mues` pool, seeded
/// `[zero, one]`).
///
/// `perms` matches `input.elms` order.  `rklpols` is the relative-KL polynomial
/// pool, seeded `[zero, one]`.
///
/// # Why only `input` + `perms`
///
/// Task P5 (`klcells`) consumes the induced graph by building a
/// [`CellGraph`](crate::cellgraph::CellGraph) via `from_relkl(output.input)` and
/// decomposing it; it constructs its own `pairs`/involution data independently
/// from the group, so it needs only `input` (the graph) and `perms` (the
/// element identities).  Per YAGNI we expose exactly that, plus the two pools
/// for inspection/testing.  `elmsX` (coset-rep words) and the `(y, v) → flat`
/// bijection are internal to the recursion and not re-exposed.
#[derive(Clone, Debug)]
pub struct RelKlOutput {
    /// The induced-cell W-graph in `RelKlInput`/`Global` form.
    pub input: RelKlInput,
    /// Perms of `input.elms`, in the same order.
    pub perms: Vec<Perm>,
    /// The relative-KL polynomial pool, seeded `[zero, one]`.
    pub rklpols: Vec<Laurent>,
    /// The global mu pool (`mues`), seeded `[zero, one]`.
    pub mues: Vec<Laurent>,
}

// ---------------------------------------------------------------------------
// relklpols
// ---------------------------------------------------------------------------

/// Relative KL polynomials of the induced set `X1·C`.
///
/// `cell1` describes the left cell (or union of cells) `C` of `W1` as a
/// [`RelKlInput`] in [`MuPools::PerGen`] form (the output of
/// [`CellGraph::to_relkl`](crate::cellgraph::CellGraph::to_relkl)).  Its `elms`
/// are reduced words in `W1`'s **own** generator labels.
///
/// Equal parameters only: all generator weights are implicitly `1` (PyCox's
/// `weightL = 1` branch).  No weights parameter is taken; the `Lw`/`Lw1` length
/// sums below are plain word lengths.
// The recursion is a faithful matrix port of PyCox `relklpols`: the positional
// `(y, x, v, u)` index loops mirror the source line-for-line and index multiple
// parallel structures (`mat`, `bruhatX`, `bij`, the coset/cell tables), so
// iterator rewrites would obscure the correspondence rather than clarify it.
#[allow(clippy::needless_range_loop)]
pub fn relklpols(
    w: &CoxeterGroup,
    w1: &Parabolic,
    cell1: &RelKlInput,
    _opts: &RelKlOpts,
) -> RelKlOutput {
    debug_assert!(
        matches!(cell1.mpols, MuPools::PerGen(_)),
        "relklpols expects cell1 in PerGen form (from CellGraph::to_relkl)"
    );

    let rank = w.rank;
    let j = w1.gen_map(); // J[t'] = W-generator index of W1-local generator t'.

    // --- Setup: coset reps X1 ------------------------------------------------
    let x1w: Vec<Word> = red_left_coset_reps(w, &w1.sub_j);
    let x1: Vec<Perm> = x1w.iter().map(|word| w.word_to_perm(word)).collect();
    let nx = x1.len();
    // Lw[x] = ℓ_W(X1w[x]) (= word length, equal params).
    let lw: Vec<u32> = x1w.iter().map(|word| word.len() as u32).collect();

    // Index X1 by coxelm for the `s·X1[x] ∈ X1?` membership test.
    let x1_pos: HashMap<_, Cx> = x1
        .iter()
        .enumerate()
        .map(|(i, p)| (p.coxelm_sr(&w.simple_root), i))
        .collect();

    // lft[s][x]: see `Lft`.  s over W.rank, x over X1.
    let lft: Vec<Vec<Lft>> = (0..rank)
        .map(|s| {
            (0..nx)
                .map(|x| {
                    // sw = s · X1[x] (LEFT multiply): PyCox `[w[i] for i in
                    // permgens[s]]` = then(permgens[s], X1[x]).
                    let sw = w.permgens[s].then(&x1[x]);
                    let sw_ce = sw.coxelm_sr(&w.simple_root);
                    if let Some(&xi) = x1_pos.get(&sw_ce) {
                        Lft::In(xi)
                    } else {
                        // Leaves X: sw = X1[x]·t for a unique W-generator t (in J).
                        // PyCox `[permgens[t][i] for i in w]` = then(X1[x],
                        // permgens[t]) = X1[x]·t (RIGHT multiply) — NOT t·X1[x].
                        let t = (0..rank)
                            .find(|&t| {
                                x1[x].then(&w.permgens[t]).coxelm_sr(&w.simple_root) == sw_ce
                            })
                            .expect("s·X1[x] leaves X yet no W-generator realises it");
                        debug_assert!(
                            j.contains(&(t as Gen)),
                            "lft Out(t): t={t} not in J={j:?} (s={s}, x={x})"
                        );
                        Lft::Out(t as Gen)
                    }
                })
                .collect()
        })
        .collect();

    // --- Setup: cell C in W1 -------------------------------------------------
    let nc = cell1.elms.len();
    // Lw1[u] = Σ poids[J[s]] over cell1.elms[u]; equal params ⇒ word length.
    let lw1: Vec<u32> = cell1.elms.iter().map(|word| word.len() as u32).collect();
    // p1[u] = W1.wordtoperm(local word) — perms in W1's OWN root system.
    let p1: Vec<Perm> = cell1
        .elms
        .iter()
        .map(|word| w1.group.word_to_perm(word))
        .collect();
    let p1_pos: HashMap<_, Cu> = p1
        .iter()
        .enumerate()
        .map(|(i, p)| (p.coxelm_sr(&w1.group.simple_root), i))
        .collect();

    // lft1[J[t']][u]: left-mult of p1[u] by W1-local generator t'.
    //   in p1 → its index; else descends-out (-1 → encode `nc` sentinel? no:
    //   PyCox uses -1 and len(p1)).  We store the raw signed-ish result.
    // PyCox stores -1 (descends out) and len(p1) (ascends out).  We mirror with
    // an enum-free i64 so the `u < lft1[...][u]` and `lft1[t][w] > w` comparisons
    // port verbatim.
    const DESC_OUT: i64 = -1; // PyCox -1: t'·u < u, leaves the cell downward.
    let asc_out: i64 = nc as i64; // PyCox len(p1): t'·u > u, leaves the cell upward.
    let n1 = w1.group.n_pos as usize; // W1.N
                                      // lft1 keyed by W-generator (length rank); only J-entries are filled.
    let mut lft1: Vec<Vec<i64>> = vec![Vec::new(); rank];
    for (t_local, &t_w) in j.iter().enumerate() {
        let col: Vec<i64> = (0..nc)
            .map(|u| {
                // w1elt = t'·p1[u] (left multiply) in W1's root system.
                let w1elt = w1.group.permgens[t_local].then(&p1[u]);
                let ce = w1elt.coxelm_sr(&w1.group.simple_root);
                if let Some(&ui) = p1_pos.get(&ce) {
                    ui as i64
                } else if (p1[u].0[w1.group.simple_root[t_local]] as usize) >= n1 {
                    // p1[u][t'] >= W1.N  ⇒  t'·p1[u] < p1[u]  ⇒  descends out.
                    DESC_OUT
                } else {
                    asc_out
                }
            })
            .collect();
        lft1[t_w as usize] = col;
    }

    // --- bruhatX[y][x] = Bruhat(X1[x] <= X1[y]) for x <= y -------------------
    // Stored as a full nx×nx symmetric-by-construction lower table: bx[y][x].
    let bruhat_x: Vec<Vec<bool>> = (0..nx)
        .map(|y| (0..=y).map(|x| bruhat::leq(w, &x1[x], &x1[y])).collect())
        .collect();
    // Helper closure to query bruhatX with x <= y (only valid then).
    // bruhat_x[y] has length y+1 (indices x = 0..=y).  The recursion only ever
    // queries pairs with x <= y (a mathematical invariant — `s·x` ascending
    // stays ≤ y for s ∈ ldy), but we guard defensively so an unexpected x > y
    // returns `false` rather than panicking.
    let bx = |y: Cx, x: Cx| -> bool { x <= y && bruhat_x[y][x] };

    // --- Matrix init ---------------------------------------------------------
    // mat[(y,x)] = nc×nc grid of SlotState; only (y,x) with bruhatX present.
    // Diagonal blocks (y,y) always present.
    let mut mues: Vec<Laurent> = vec![Laurent::zero(), Laurent::one()];
    let mut mat: HashMap<(Cx, Cx), Vec<Vec<SlotState>>> = HashMap::new();

    for y in 0..nx {
        for x in 0..y {
            if bx(y, x) {
                let mut grid = vec![vec![SlotState::Absent; nc]; nc];
                for v in 0..nc {
                    for u in 0..nc {
                        // PyCox: (x==y and u==v) or Lw[x]+Lw1[u] < Lw[y]+Lw1[v].
                        // Here x<y so the first disjunct is false.
                        if lw[x] + lw1[u] < lw[y] + lw1[v] {
                            grid[v][u] = SlotState::Pending;
                        }
                    }
                }
                mat.insert((y, x), grid);
            }
        }
        // Diagonal block (y,y): copied from cell1.
        let mut diag = vec![vec![SlotState::Absent; nc]; nc];
        for i in 0..nc {
            for jj in 0..i {
                if let Some(slot) = cell1.klmat[i][jj].as_ref() {
                    // PyCox: mat[y,y][i][j]='c0'; then read the FIRST generator r
                    // with a non-''/'0' slot index; intern that mu into mues.
                    let mu_idx = diag_block_mu(slot, &cell1.mpols, &mut mues);
                    diag[i][jj] = SlotState::Done { rk: 0, mu: mu_idx };
                }
            }
            // Diagonal of diagonal: 'c1c0' → rk=1 (one), mu=0 (zero).
            diag[i][i] = SlotState::Done { rk: 1, mu: 0 };
        }
        mat.insert((y, y), diag);
    }

    // --- Main recursion ------------------------------------------------------
    let mut rklpols: Vec<Laurent> = vec![Laurent::zero(), Laurent::one()];

    for y in 0..nx {
        let ldy = w.left_descents(&x1[y]); // W-generators.
        for x in (0..y).rev() {
            if !bx(y, x) {
                continue;
            }
            // fs  = [s in ldy : lft[s][x] is In(xi) with xi > x] (s·x ascends, stays in X)
            // fs1 = [s in ldy : lft[s][x] is In(xi) with 0 <= xi < x]
            let fs: Vec<Gen> = ldy
                .iter()
                .copied()
                .filter(|&s| matches!(lft[s as usize][x], Lft::In(xi) if xi > x))
                .collect();
            let fs1: Vec<Gen> = ldy
                .iter()
                .copied()
                .filter(|&s| matches!(lft[s as usize][x], Lft::In(xi) if xi < x))
                .collect();

            if !fs.is_empty() {
                // Case A: s·x ascends and stays in X.
                let s = fs[0] as usize;
                let sx = match lft[s][x] {
                    Lft::In(xi) => xi,
                    Lft::Out(_) => unreachable!("fs guarantees In"),
                };
                // We need to read mat[(y, sx)] and write mat[(y, x)] — split borrow.
                for v in 0..nc {
                    for u in 0..nc {
                        if !mat[&(y, x)][v][u].is_marked() {
                            continue;
                        }
                        // Source slot mat[y,sx][v][u].
                        let src = if bx(y, sx) {
                            mat.get(&(y, sx)).map(|g| g[v][u])
                        } else {
                            None
                        };
                        let new_state = match src {
                            Some(s_state) if s_state.is_marked() => {
                                // Copy the rk verbatim; compute mu via relmue.
                                let rk = s_state.rk().unwrap_or(0);
                                let mu = if rk != 0 {
                                    let m = relmue(
                                        lw[y] + lw1[v],
                                        lw[x] + lw1[u],
                                        &rklpols[rk as usize],
                                    );
                                    intern(&mut mues, m)
                                } else {
                                    0
                                };
                                SlotState::Done { rk, mu }
                            }
                            _ => SlotState::Done { rk: 0, mu: 0 }, // '0c0'
                        };
                        mat.get_mut(&(y, x)).unwrap()[v][u] = new_state;
                    }
                }
            } else {
                // Case B: fs empty.
                for u in 0..nc {
                    // Vanishing test: ∃ s ∈ ldy with lft[s][x] = Out(t) and
                    // u < lft1[t][u]  (s exits X at W1-gen t and t ascends u).
                    let vanishes = ldy.iter().any(|&s| {
                        if let Lft::Out(t) = lft[s as usize][x] {
                            (u as i64) < lft1[t as usize][u]
                        } else {
                            false
                        }
                    });
                    if vanishes {
                        for v in 0..nc {
                            if mat[&(y, x)][v][u].is_marked() {
                                mat.get_mut(&(y, x)).unwrap()[v][u] =
                                    SlotState::Done { rk: 0, mu: 0 };
                            }
                        }
                        continue;
                    }
                    // s = fs1[0] if fs1 nonempty else ldy[0].
                    let s = if let Some(&s1) = fs1.first() {
                        s1 as usize
                    } else {
                        ldy[0] as usize
                    };
                    let sx = lft[s][x]; // Lft
                    let sy = match lft[s][y] {
                        Lft::In(yi) => yi,
                        // s ∈ ldy(y) means s is a left descent of y, so s·y < y
                        // stays in X (descending in X keeps you in X).  PyCox
                        // always reads sy as an index.
                        Lft::Out(_) => unreachable!("s ∈ ldy ⇒ s·y descends, stays in X"),
                    };
                    for v in 0..nc {
                        if !mat[&(y, x)][v][u].is_marked() {
                            continue;
                        }
                        let h = compute_h(CaseBCtx {
                            y,
                            x,
                            u,
                            v,
                            s,
                            sx,
                            sy,
                            lw: &lw,
                            lw1: &lw1,
                            nc,
                            bx: &bruhat_x,
                            lft: &lft,
                            lft1: &lft1,
                            mat: &mat,
                            mues: &mues,
                            rklpols: &rklpols,
                            cell1,
                        });
                        let new_state = if h.is_zero() {
                            SlotState::Done { rk: 0, mu: 0 } // '0c0'
                        } else {
                            let rk = intern(&mut rklpols, h.clone());
                            let m = relmue(lw[y] + lw1[v], lw[x] + lw1[u], &h);
                            let mu = intern(&mut mues, m);
                            SlotState::Done { rk, mu }
                        };
                        mat.get_mut(&(y, x)).unwrap()[v][u] = new_state;
                    }
                }
            }
        }
    }

    // --- Relabel: ap = X·C words sorted by length (stable) -------------------
    // ap-word for (y, v): reduce(X1w[y] ++ [J[s'] for s' in cell1.elms[v]]).
    let mut ap_pairs: Vec<((Cx, Cu), Word)> = Vec::with_capacity(nx * nc);
    for y in 0..nx {
        for v in 0..nc {
            let mut full = x1w[y].clone();
            full.extend(w1.word_to_w(&cell1.elms[v]));
            let reduced = w.perm_to_word(&w.word_to_perm(&full));
            ap_pairs.push(((y, v), reduced));
        }
    }
    // PyCox: ap.sort(key=len) — stable sort by length only.
    let mut order: Vec<usize> = (0..ap_pairs.len()).collect();
    order.sort_by_key(|&i| ap_pairs[i].1.len());
    let ap: Vec<Word> = order.iter().map(|&i| ap_pairs[i].1.clone()).collect();
    let ap_perms: Vec<Perm> = ap.iter().map(|word| w.word_to_perm(word)).collect();

    // bij[(y,v)] = ap1.index(permmult(X1[y], elms1[v])) where elms1[v] is the
    // W-perm of the cell element.  We find it by perm equality.
    let elms1: Vec<Perm> = cell1
        .elms
        .iter()
        .map(|word| w.word_to_perm(&w1.word_to_w(word)))
        .collect();
    // Index ap perms for fast lookup.
    let ap_pos: HashMap<_, u32> = ap_perms
        .iter()
        .enumerate()
        .map(|(i, p)| (p.coxelm_sr(&w.simple_root), i as u32))
        .collect();
    let mut bij: HashMap<(Cx, Cu), u32> = HashMap::new();
    for y in 0..nx {
        for v in 0..nc {
            // permmult(X1[y], elms1[v]) = then(X1[y], elms1[v])? PyCox permmult
            // is then().  But X1[y]·(cell elt) as a W element: word X1w[y]++cellword.
            let prod = x1[y].then(&elms1[v]);
            let ce = prod.coxelm_sr(&w.simple_root);
            let flat = *ap_pos
                .get(&ce)
                .expect("induced product perm must appear in ap");
            bij.insert((y, v), flat);
        }
    }

    // nmat: flat strict-lower-triangular (klmat[fy] of length fy) with single
    // Global-index SlotData.  PyCox copies every slot != 'f' (including '0c0'
    // zero slots) where bij[x,u] <= bij[y,v].
    let n_flat = ap.len();
    let mut klmat: Vec<Vec<KlSlot>> = (0..n_flat).map(|fy| vec![None; fy]).collect();
    for y in 0..nx {
        for x in 0..=y {
            if !bx(y, x) {
                continue;
            }
            let grid = match mat.get(&(y, x)) {
                Some(g) => g,
                None => continue,
            };
            for v in 0..nc {
                for u in 0..nc {
                    let fy = bij[&(y, v)] as usize;
                    let fx = bij[&(x, u)] as usize;
                    if fx <= fy && grid[v][u].is_marked() {
                        // Completed slot → store its mu index (single Global).
                        // Pending should not survive; treat defensively as zero.
                        let mu = grid[v][u].mu().unwrap_or(0);
                        // PyCox stores 'c<rk>c<mu>'; from_relkl reads mu[0].
                        if fx < fy {
                            klmat[fy][fx] = Some(SlotData { mu: vec![mu] });
                        }
                        // fx == fy is the diagonal (dropped — klmat[fy] has no fy).
                    }
                }
            }
        }
    }

    let input = RelKlInput {
        elms: ap,
        klmat,
        mpols: MuPools::Global(mues.clone()),
    };

    RelKlOutput {
        input,
        perms: ap_perms,
        rklpols,
        mues,
    }
}
