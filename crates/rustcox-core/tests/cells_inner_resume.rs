//! Q4 end-to-end test: `klcells` streaming survives a SLURM-style kill that
//! lands *inside* a monster rep's inner `relklpols` call.
//!
//! The driver-level checkpoint (Q1) recovers between reps, but a single rep that
//! exceeds the time box would livelock without the layer-granular inner log
//! (Q4).  This test forces a kill mid-inner-call via the test-only inner-stop
//! hook, then resumes and asserts:
//!   * the final canonical partition equals an uninterrupted `klcells` run;
//!   * the inner block-log was actually USED on resume (the driver's
//!     `relkl_inner_resumes` probe is ≥ 1);
//!   * the inner log files are deleted once the rep completes (bounded disk).

use std::cell::RefCell;
use std::io;
use std::path::{Path, PathBuf};

use rustcox_core::{
    element::Word,
    group::CoxeterGroup,
    kl::{
        klcells, klcells_streaming_test_inner_stop, CellRecord, CellsOpts, CheckpointCfg,
        KlCellsSummary,
    },
};

/// Canonicalize a flat list of cells the way `klcells` does at the top level.
fn canonicalize(g: &CoxeterGroup, cells: &[Vec<Word>]) -> Vec<Vec<Word>> {
    let mut out: Vec<Vec<Word>> = cells
        .iter()
        .map(|c| {
            let mut can: Vec<Word> = c
                .iter()
                .map(|w| g.perm_to_word(&g.word_to_perm(w)))
                .collect();
            can.sort_by(|a, b| (a.len(), a).cmp(&(b.len(), b)));
            can
        })
        .collect();
    out.sort();
    out
}

fn scratch(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rcx_inner_resume_{}_{label}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Number of `*.blkhdr` files under `dir/relkl/` (in-flight inner logs).
fn inner_log_headers(dir: &Path) -> usize {
    let relkl = dir.join("relkl");
    match std::fs::read_dir(&relkl) {
        Ok(rd) => rd
            .flatten()
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|n| n.ends_with(".blkhdr"))
                    .unwrap_or(false)
            })
            .count(),
        Err(_) => 0,
    }
}

/// Run streaming with an inner-stop hook for `(rep, layer)`; collect records.
/// Returns `(result, records, summary_opt)`.
#[allow(clippy::type_complexity)]
fn run_with_inner_stop(
    g: &CoxeterGroup,
    opts: &CellsOpts,
    cfg: &CheckpointCfg,
    rep_stop: Option<(usize, usize)>,
    tier_direct: usize,
    tier_tau: usize,
) -> (Result<KlCellsSummary, String>, Vec<CellRecord>) {
    let records = RefCell::new(Vec::new());
    let mut sink = |rec: CellRecord| -> io::Result<()> {
        records.borrow_mut().push(rec);
        Ok(())
    };
    let res = klcells_streaming_test_inner_stop(
        g,
        opts,
        &mut sink,
        None,
        Some(cfg),
        tier_direct,
        tier_tau,
        rep_stop,
    )
    .map_err(|e| e.to_string());
    (res, records.into_inner())
}

#[test]
fn klcells_resume_with_inner_log() {
    // B4 with tiny tiers fragments each induced set into many components, so a
    // rep emits many cells across multiple inner wavefront layers — the closest
    // local analogue of an E8 monster rep.
    let g = CoxeterGroup::from_type("B4").unwrap();
    let opts = CellsOpts::default();
    let (tier_direct, tier_tau) = (1usize, 3usize);

    let full = klcells(&g, &opts).unwrap();

    // Find the first rep whose inner call actually runs multiple layers, so the
    // stop hook (stop after layer 1) leaves a durable inner log.  We probe by
    // trying increasing rep indices with a fresh checkpoint dir each time.
    let mut chosen: Option<usize> = None;
    for rep_i in 0..16usize {
        let cfg = CheckpointCfg::new(scratch(&format!("probe{rep_i}")));
        let (res, _recs) =
            run_with_inner_stop(&g, &opts, &cfg, Some((rep_i, 1)), tier_direct, tier_tau);
        // The inner stop surfaces as an Err and leaves a header behind.
        if res.is_err() && inner_log_headers(&cfg.dir) >= 1 {
            chosen = Some(rep_i);
            let _ = std::fs::remove_dir_all(&cfg.dir);
            break;
        }
        let _ = std::fs::remove_dir_all(&cfg.dir);
    }
    let rep_i = chosen.expect("some B4 rep must run ≥2 inner layers under tiny tiers");

    // Phase 1: kill INSIDE rep `rep_i`'s inner call, after layer 1.
    let cfg = CheckpointCfg::new(scratch("run"));
    let (res1, disk) =
        run_with_inner_stop(&g, &opts, &cfg, Some((rep_i, 1)), tier_direct, tier_tau);
    assert!(res1.is_err(), "phase 1 must abort on the inner stop");
    assert!(
        inner_log_headers(&cfg.dir) >= 1,
        "an inner block-log must be durable after the kill"
    );

    // Phase 2: resume with NO inner stop → the inner log resumes the rep, the
    // run completes.  `relkl_inner_resumes` proves the inner log was used.
    let (res2, resumed) = run_with_inner_stop(&g, &opts, &cfg, None, tier_direct, tier_tau);
    let summary = res2.expect("phase 2 must complete");
    assert!(
        summary.relkl_inner_resumes >= 1,
        "the inner block-log must be USED on resume (got {} resumes)",
        summary.relkl_inner_resumes
    );
    assert_eq!(
        summary.total_elements, g.order,
        "resumed run must cover all of W"
    );

    // After the rep completes, its inner log is deleted (bounded disk).  At the
    // clean finish no in-flight inner logs remain.
    assert_eq!(
        inner_log_headers(&cfg.dir),
        0,
        "inner logs must be deleted once their rep completes"
    );

    // Reconstruct the stream the CLI would have (kept disk prefix ++ resumed)
    // and assert it canonicalizes to the full partition.
    let keep = summary.records_kept as usize;
    assert!(
        keep <= disk.len(),
        "records_kept must not exceed disk records"
    );
    let mut all: Vec<Vec<Word>> = disk[..keep].iter().map(|r| r.words.clone()).collect();
    all.extend(resumed.iter().map(|r| r.words.clone()));
    let got = canonicalize(&g, &all);
    assert_eq!(
        got, full.cells,
        "reconstructed partition after inner-log resume must equal the full run"
    );

    let _ = std::fs::remove_dir_all(&cfg.dir);
}
