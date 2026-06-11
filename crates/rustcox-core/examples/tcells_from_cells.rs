//! Compute the number of two-sided cells from a `cells_*.json[.gz]` document
//! (output of `rustcox cells`), without any KL recomputation.
//!
//! Method: right cells are the element-wise inverses of left cells; two-sided
//! cells are the connected components of "same left cell OR same right cell"
//! (union-find), exactly as in `kl/cells.rs` but driven by the cells file.
//! The inverse of an element given by a word is the reversed word; elements
//! are identified by coxelm fingerprints (no full-group enumeration).
//!
//! Usage: tcells_from_cells <TYPE> <cells.json[.gz]>
//! e.g.:  tcells_from_cells E7 results/cells_E7.json.gz

use std::collections::HashMap;
use std::io::Read;

use rustcox_core::element::CoxElm;
use rustcox_core::group::CoxeterGroup;

fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn main() {
    let mut args = std::env::args().skip(1);
    let typ = args.next().expect("usage: tcells_from_cells <TYPE> <FILE>");
    let path = args.next().expect("usage: tcells_from_cells <TYPE> <FILE>");

    let raw = std::fs::read(&path).expect("read cells file");
    let text = if path.ends_with(".gz") {
        let mut s = String::new();
        flate2::read::GzDecoder::new(&raw[..])
            .read_to_string(&mut s)
            .expect("gunzip");
        s
    } else {
        String::from_utf8(raw).expect("utf8")
    };
    let doc: serde_json::Value = serde_json::from_str(&text).expect("json");
    let cells = doc["cells"].as_array().expect("cells key");

    let g = CoxeterGroup::from_type(&typ).expect("group");
    eprintln!("group {typ}: order={} ncells={}", g.order, cells.len());

    // element coxelm -> left-cell index
    let mut lcell_of: HashMap<CoxElm, u32> = HashMap::new();
    let mut words: Vec<(Vec<u8>, u32)> = Vec::new();
    for (ci, cell) in cells.iter().enumerate() {
        for w in cell.as_array().expect("cell array") {
            let word: Vec<u8> = w
                .as_array()
                .expect("word")
                .iter()
                .map(|x| x.as_u64().expect("gen") as u8)
                .collect();
            let ce = g.word_to_perm(&word).coxelm_sr(&g.simple_root);
            lcell_of.insert(ce, ci as u32);
            words.push((word, ci as u32));
        }
    }
    eprintln!("indexed {} elements", words.len());
    assert_eq!(words.len() as u128, g.order, "cells must partition W");

    // union-find over left cells: join lcell(w) with lcell(w^{-1}) viewed as
    // the right-cell relation (w and its inverse link their left cells'
    // two-sided classes through rcell membership).
    let n = cells.len();
    let mut parent: Vec<usize> = (0..n).collect();
    for (word, ci) in &words {
        let inv: Vec<u8> = word.iter().rev().copied().collect();
        let ice = g.word_to_perm(&inv).coxelm_sr(&g.simple_root);
        let cj = *lcell_of.get(&ice).expect("inverse element must be in some cell");
        // w in lcell ci and w^{-1} in lcell cj  =>  ci and cj share a right
        // cell (the inverse of lcell cj), hence same two-sided cell.
        let (a, b) = (find(&mut parent, *ci as usize), find(&mut parent, cj as usize));
        if a != b {
            parent[a] = b;
        }
    }
    let mut roots: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        roots.insert(r);
    }
    println!("TCELLS {} (from {} left cells, {})", roots.len(), n, typ);
}
