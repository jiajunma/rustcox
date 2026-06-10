#!/usr/bin/env python3
"""Golden-data generator for rustcox.

Runs PyCox (vendored as pycox_ref.py, GPL-3) and writes canonical JSON
files that the Rust test-suite compares against bit-for-bit.

Canonicalisation rules (MUST match rustcox's `io::canonical` exactly):

* Elements are identified by their canonical reduced word: the word
  produced by repeatedly stripping the smallest left descent
  (PyCox `permtoword`).  Elements are sorted by (length, word) with
  words compared lexicographically.  All matrices/cells/indices are
  remapped through this ordering.
* A Laurent polynomial is serialised as {"v": val, "c": [c0, c1, ...]}
  where c0 is the coefficient of x^val.  Zero is {"v": 0, "c": []}.
  Plain integers n are {"v": 0, "c": [n]} (or zero as above).
* Polynomial pools ("pols", and "mues" per generator) are deduplicated
  and sorted by the key (val, coeffs) with coeffs compared
  lexicographically; matrices store indices into the sorted pools.
* klmat / mumat use -1 as the sentinel for "not Bruhat-comparable"
  ('f' in PyCox) resp. "no mu-coefficient for this generator".
* Cells are lists of canonical element indices, each cell sorted
  ascending; the list of cells is sorted lexicographically.  duflo
  rows [d, a, n] follow the same cell order as "lcells"; "lorder" is
  permuted consistently with that order.

Usage:
  python3 gen_golden.py basics A3 B2 H3 ...
  python3 gen_golden.py kl A3:1 B2:2,1 G2:1,3 ...
      (TYPE:w0,w1,...  weights per generator; "TYPE:1" = all weights 1)
  python3 gen_golden.py suite          # everything committed to golden/
  python3 gen_golden.py suite-big      # + A5, F4 (slow, large files)
"""
import json
import os
import sys
import gzip
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pycox_ref import (coxeter, klpolynomials, allwords, longestperm, v)

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'golden')


# ---------------------------------------------------------------------------
# polynomial helpers
# ---------------------------------------------------------------------------

def poljson(f):
    """PyCox lpol-or-int -> {"v": val, "c": coeffs} canonical form."""
    if isinstance(f, int):
        if f == 0:
            return {"v": 0, "c": []}
        return {"v": 0, "c": [f]}
    if not f.coeffs:
        return {"v": 0, "c": []}
    return {"v": f.val, "c": list(f.coeffs)}


def polkey(pj):
    return (pj["v"], tuple(pj["c"]))


def canonical_pool(pol_list):
    """Dedup + sort a list of poljson dicts.

    Returns (sorted_pool, remap) where remap[i] is the new index of the
    polynomial that had index i in pol_list."""
    keys = [polkey(poljson(p)) for p in pol_list]
    uniq = sorted(set(keys))
    pos = {k: i for i, k in enumerate(uniq)}
    remap = [pos[k] for k in keys]
    pool = [{"v": k[0], "c": list(k[1])} for k in uniq]
    return pool, remap


# ---------------------------------------------------------------------------
# group construction
# ---------------------------------------------------------------------------

def parse_type(spec):
    """'A3' -> ('A', 3); 'I7' -> ('I7', 2) (PyCox dihedral convention)."""
    series = spec[0]
    num = int(spec[1:])
    if series == 'I':
        return ('I%d' % num, 2)
    return (series, num)


def make_group(spec):
    typ, rank = parse_type(spec)
    return coxeter(typ, rank)


def type_json(spec):
    series = spec[0]
    num = int(spec[1:])
    if series == 'I':
        return [{"series": "I", "rank": 2, "m": num}]
    return [{"series": series, "rank": num}]


# ---------------------------------------------------------------------------
# canonical element order
# ---------------------------------------------------------------------------

def canonical_order(W, elms):
    """elms: list of reduced words from PyCox (klpolynomials order).

    Returns (canwords, sigma) where canwords[k] is the canonical word of
    the element with canonical index k, and sigma[i] = canonical index
    of PyCox element i."""
    can = [W.permtoword(W.wordtoperm(w)) for w in elms]
    order = sorted(range(len(can)), key=lambda i: (len(can[i]), can[i]))
    sigma = [0] * len(can)
    for newi, oldi in enumerate(order):
        sigma[oldi] = newi
    return [can[i] for i in order], sigma


# ---------------------------------------------------------------------------
# basics golden
# ---------------------------------------------------------------------------

def gen_basics(spec):
    W = make_group(spec)
    data = {
        "schema": "rustcox-golden-v1",
        "kind": "basics",
        "type": type_json(spec),
        "rank": len(W.rank),
        "order": W.order,
        "N": W.N,
        "degrees": sorted(W.degrees),
        "coxetermat": [list(r) for r in W.coxetermat],
    }
    # root coordinates only when they are plain integers (crystallographic)
    if all(isinstance(c, int) for r in W.roots for c in r):
        data["roots"] = [list(r) for r in W.roots]
    # longest element as canonical word + length histogram, when affordable
    if W.order <= 10000:
        wds = allwords(W)
        can = sorted((W.permtoword(W.wordtoperm(w)) for w in wds),
                     key=lambda w: (len(w), w))
        hist = [0] * (W.N + 1)
        for w in can:
            hist[len(w)] += 1
        data["length_histogram"] = hist
        data["longest_word"] = W.permtoword(longestperm(W))
    return data


# ---------------------------------------------------------------------------
# kl golden
# ---------------------------------------------------------------------------

def parse_weights(W, wspec):
    parts = [int(x) for x in wspec.split(',')]
    if len(parts) == 1:
        return len(W.rank) * parts
    assert len(parts) == len(W.rank), "need one weight per generator"
    return parts


def gen_kl(spec, wspec):
    W = make_group(spec)
    weights = parse_weights(W, wspec)
    kl = klpolynomials(W, weights, v)
    n = len(kl['elms'])
    rank = len(W.rank)

    canwords, sigma = canonical_order(W, kl['elms'])
    inv = [0] * n                       # inv[new] = old
    for old, new in enumerate(sigma):
        inv[new] = old

    pol_pool, pol_remap = canonical_pool(kl['klpols'])
    mu_pools, mu_remaps = [], []
    for s in range(rank):
        pool, remap = canonical_pool(kl['mpols'][s])
        mu_pools.append(pool)
        mu_remaps.append(remap)

    # parse PyCox 'c<p>c<i0>c<i1>...' strings into canonical matrices
    lengths = [len(w) for w in canwords]
    klmat, mumat = [], []
    for wn in range(n):
        wo = inv[wn]
        prow, mrow = [], []
        for yn in range(wn + 1):
            yo = inv[yn]
            if yo > wo or (lengths[yn] == lengths[wn] and yn != wn):
                prow.append(-1)
                mrow.append([-1] * rank)
                continue
            ent = kl['klmat'][wo][yo]
            if ent[0] != 'c':
                prow.append(-1)
                mrow.append([-1] * rank)
                continue
            tok = ent.split('c')        # ['', '<p>', m_0, ..., m_{rank-1}]
            prow.append(pol_remap[int(tok[1])])
            mrow.append([mu_remaps[s][int(tok[2 + s])] if tok[2 + s] != ''
                         else -1 for s in range(rank)])
        klmat.append(prow)
        mumat.append(mrow)

    def canon_cells(cells):
        cs = [sorted(sigma[x] for x in c) for c in cells]
        order = sorted(range(len(cs)), key=lambda i: cs[i])
        return [cs[i] for i in order], order

    lcells, lperm = canon_cells(kl['lcells'])
    rcells, _ = canon_cells(kl['rcells'])
    tcells, _ = canon_cells(kl['tcells'])
    duflo = [[sigma[kl['duflo'][i][0]], kl['duflo'][i][1], kl['duflo'][i][2]]
             for i in lperm]
    lorder = [[1 if kl['lorder'][i][j] else 0 for j in lperm] for i in lperm]
    arrows = sorted([sigma[a], sigma[b]] for (a, b) in kl['arrows'])

    return {
        "schema": "rustcox-golden-v1",
        "kind": "kl",
        "type": type_json(spec),
        "weights": weights,
        "rank": rank,
        "order": W.order,
        "N": W.N,
        "elms": canwords,
        "pols": pol_pool,
        "mues": mu_pools,
        "klmat": klmat,
        "mumat": mumat,
        "arrows": arrows,
        "lcells": lcells,
        "duflo": duflo,
        "lorder": lorder,
        "rcells": rcells,
        "tcells": tcells,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def write_json(name, data, compress=False):
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    txt = json.dumps(data, separators=(',', ':'), sort_keys=True)
    if compress:
        path = os.path.join(GOLDEN_DIR, name + '.json.gz')
        with gzip.open(path, 'wt', encoding='utf-8') as fh:
            fh.write(txt)
    else:
        path = os.path.join(GOLDEN_DIR, name + '.json')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(txt)
    sys.stderr.write('wrote %s (%d bytes)\n' % (path, os.path.getsize(path)))


BASICS_SUITE = ['A1', 'A2', 'A3', 'A4', 'A5', 'B2', 'B3', 'B4', 'C3',
                'D4', 'D5', 'G2', 'F4', 'H3', 'H4', 'I5', 'I7', 'I8', 'E6']
KL_SUITE = [('A1', '1'), ('A2', '1'), ('A3', '1'), ('A4', '1'),
            ('B2', '1'), ('B3', '1'), ('B4', '1'), ('C3', '1'),
            ('D4', '1'), ('G2', '1'), ('H3', '1'), ('I5', '1'), ('I7', '1'),
            ('B2', '2,1'), ('B2', '1,2'), ('B3', '2,1,1'), ('G2', '1,3'),
            ('G2', '3,1'), ('I8', '1,2')]
KL_BIG = [('A5', '1'), ('F4', '1')]


def kl_name(spec, wspec):
    w = wspec.replace(',', '_')
    return 'kl_%s_w%s' % (spec, w)


def main(argv):
    if not argv:
        sys.stderr.write(__doc__)
        return 1
    mode, args = argv[0], argv[1:]
    t0 = time.time()
    if mode == 'basics':
        for spec in args:
            write_json('basics_%s' % spec, gen_basics(spec))
    elif mode == 'kl':
        for a in args:
            spec, _, wspec = a.partition(':')
            wspec = wspec or '1'
            write_json(kl_name(spec, wspec), gen_kl(spec, wspec))
    elif mode in ('suite', 'suite-big'):
        for spec in BASICS_SUITE:
            write_json('basics_%s' % spec, gen_basics(spec))
        for spec, wspec in KL_SUITE:
            write_json(kl_name(spec, wspec), gen_kl(spec, wspec))
        if mode == 'suite-big':
            for spec, wspec in KL_BIG:
                write_json(kl_name(spec, wspec), gen_kl(spec, wspec),
                           compress=True)
    else:
        sys.stderr.write('unknown mode %r\n' % mode)
        return 1
    sys.stderr.write('done in %.1fs\n' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
