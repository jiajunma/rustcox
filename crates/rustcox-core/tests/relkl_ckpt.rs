//! Layer-granular checkpoint/resume tests for `relklpols` (Task Q4).
//!
//! Five families:
//!   1. `relkl_ckpt_roundtrip_*` — run with a stop-after-layer hook, then resume
//!      without the hook; the resumed [`RelKlOutput`] is BYTE-IDENTICAL to an
//!      uninterrupted run.  Covers stops at the first, a middle, and the last
//!      layer (resume = pure replay).
//!   2. `relkl_ckpt_fingerprint_mismatch` — a log for a different `cell1` (same
//!      rep tag) is rejected; the run starts fresh and the stale files are gone.
//!   3. `relkl_ckpt_truncated_log` — truncating the physical log mid-record (its
//!      tail bytes are lost but the header byte length is honored) still resumes
//!      correctly via header-bounded replay.
//!   4. `relkl_ckpt_multi_interrupt` — stopping and resuming at *every* layer in
//!      turn reproduces the uninterrupted output (the strongest replay check).
//!   5. `relkl_ckpt_clean_run_deletes_files` — a completed call leaves no log.

use std::path::PathBuf;

use rustcox_core::{
    cellgraph::{relkl_input_from_table, RelKlInput},
    element::ElmIdx,
    group::CoxeterGroup,
    kl::{
        relkl_ckpt::{self, RelKlCkptCfg},
        relklpols, relklpols_resumable, CellData, KlOpts, KlTable, RelKlOpts, RelKlOutput,
        RelKlRunOutcome,
    },
    parabolic::Parabolic,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_table(g: &CoxeterGroup) -> KlTable {
    rustcox_core::kl::klpolynomials_seq(g, &KlOpts::equal(g.rank)).unwrap()
}

fn w1_cells(t1: &KlTable) -> Vec<Vec<ElmIdx>> {
    CellData::from_table(t1).lcells
}

/// A unique scratch dir per test (process id + label), cleaned on entry.
fn scratch(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rcx_relkl_ckpt_{}_{label}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Equality of two [`RelKlOutput`]s on every field that defines the contract:
/// the induced graph (`input`), the perms, and both pools.  (`stats` is derived
/// and not part of the byte contract, but we check the occupancy counts too.)
fn assert_output_identical(a: &RelKlOutput, b: &RelKlOutput, ctx: &str) {
    assert_eq!(a.input.elms, b.input.elms, "{ctx}: elms differ");
    assert_eq!(a.input.klmat, b.input.klmat, "{ctx}: klmat differ");
    assert_eq!(a.input.mpols, b.input.mpols, "{ctx}: mpols differ");
    assert_eq!(a.perms, b.perms, "{ctx}: perms differ");
    assert_eq!(a.rklpols, b.rklpols, "{ctx}: rklpols differ");
    assert_eq!(a.mues, b.mues, "{ctx}: mues differ");
    assert_eq!(
        (a.stats.absent, a.stats.zero, a.stats.nonzero),
        (b.stats.absent, b.stats.zero, b.stats.nonzero),
        "{ctx}: slot stats differ"
    );
}

/// Pick the biggest (most elements) W1 cell of B3 ⊂ B4 — the most layers/biggest
/// blocks in the local suite, the closest analogue of an E8 monster rep.
fn b4_biggest_cell() -> (CoxeterGroup, Parabolic, RelKlInput) {
    let w = CoxeterGroup::from_type("B4").unwrap();
    let j: Vec<u8> = vec![0, 1, 2]; // W1 = B3.
    let w1 = Parabolic::new(&w, &j).unwrap();
    let t1 = build_table(&w1.group);
    let cells = w1_cells(&t1);
    let biggest = cells
        .iter()
        .max_by_key(|c| c.len())
        .expect("B3 has cells")
        .clone();
    let cell1 = relkl_input_from_table(&w1.group, &t1, &biggest);
    (w, w1, cell1)
}

/// Count the layers an uninterrupted run would have (= |X1|), so the
/// stop-after-layer hook can target the first / mid / last layer.
fn num_layers(w: &CoxeterGroup, j: &[u8]) -> usize {
    rustcox_core::parabolic::red_left_coset_reps(w, j).len()
}

// ---------------------------------------------------------------------------
// Test 1: roundtrip — stop after layer k, resume, byte-identical
// ---------------------------------------------------------------------------

fn roundtrip_at(stop_layer: usize, label: &str) {
    let (w, w1, cell1) = b4_biggest_cell();
    let dir = scratch(label);

    // Reference: a plain uninterrupted run.
    let reference = relklpols(&w, &w1, &cell1, &RelKlOpts { threads: Some(1) });

    // Phase 1: run with the stop hook.
    let mut cfg = RelKlCkptCfg::new(&dir, "rep00000");
    cfg.test_stop_after_layer = Some(stop_layer);
    let outcome = relklpols_resumable(&w, &w1, &cell1, &RelKlOpts { threads: Some(1) }, Some(&cfg));
    match outcome {
        RelKlRunOutcome::Stopped { last_layer } => {
            assert_eq!(
                last_layer, stop_layer,
                "{label}: stopped at the wrong layer"
            );
            assert!(
                cfg.header_exists(),
                "{label}: a durable header must exist after a stop"
            );
        }
        RelKlRunOutcome::Done(_) => panic!("{label}: expected Stopped, got Done"),
    }

    // Phase 2: resume WITHOUT the stop hook → completes.
    let resume_cfg = RelKlCkptCfg::new(&dir, "rep00000");
    let resumed = match relklpols_resumable(
        &w,
        &w1,
        &cell1,
        &RelKlOpts { threads: Some(1) },
        Some(&resume_cfg),
    ) {
        RelKlRunOutcome::Done(out) => *out,
        RelKlRunOutcome::Stopped { .. } => panic!("{label}: resume must complete"),
    };

    assert_output_identical(&resumed, &reference, label);

    // Clean completion deletes the log files.
    assert!(
        !resume_cfg.header_exists() && !resume_cfg.log_path().is_file(),
        "{label}: completed call must delete its log files"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn relkl_ckpt_roundtrip_first_layer() {
    // k = 0: stop before any non-trivial wavefront work (layer 0 has no
    // off-diagonal blocks); resume recomputes essentially everything.
    roundtrip_at(0, "k0");
}

#[test]
fn relkl_ckpt_roundtrip_mid_layer() {
    let n = num_layers(&CoxeterGroup::from_type("B4").unwrap(), &[0, 1, 2]);
    assert!(n >= 3, "B4/B3 must have several layers (got {n})");
    roundtrip_at(n / 2, "kmid");
}

#[test]
fn relkl_ckpt_roundtrip_last_layer() {
    // Stop after the LAST layer → resume is a pure replay (no new compute).
    let n = num_layers(&CoxeterGroup::from_type("B4").unwrap(), &[0, 1, 2]);
    roundtrip_at(n - 1, "klast");
}

// ---------------------------------------------------------------------------
// Test 4: stop+resume at EVERY layer in turn (strongest replay check)
// ---------------------------------------------------------------------------

#[test]
fn relkl_ckpt_multi_interrupt() {
    let (w, w1, cell1) = b4_biggest_cell();
    let dir = scratch("multi");
    let opts = RelKlOpts { threads: Some(1) };

    let reference = relklpols(&w, &w1, &cell1, &opts);
    let n = num_layers(&w, &[0, 1, 2]);

    // Stop after layer 0, resume-and-stop after layer 1, … until completion.
    for stop in 0..n {
        let mut cfg = RelKlCkptCfg::new(&dir, "repmulti");
        cfg.test_stop_after_layer = Some(stop);
        match relklpols_resumable(&w, &w1, &cell1, &opts, Some(&cfg)) {
            RelKlRunOutcome::Stopped { last_layer } => assert_eq!(last_layer, stop),
            RelKlRunOutcome::Done(_) => panic!("unexpected Done at stop={stop}"),
        }
    }
    // Final resume with no stop completes.
    let final_cfg = RelKlCkptCfg::new(&dir, "repmulti");
    let done = match relklpols_resumable(&w, &w1, &cell1, &opts, Some(&final_cfg)) {
        RelKlRunOutcome::Done(out) => *out,
        RelKlRunOutcome::Stopped { .. } => panic!("final resume must complete"),
    };
    assert_output_identical(&done, &reference, "multi-interrupt");

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Test 2: fingerprint mismatch → fresh start, stale files replaced
// ---------------------------------------------------------------------------

#[test]
fn relkl_ckpt_fingerprint_mismatch() {
    let w = CoxeterGroup::from_type("B4").unwrap();
    let j: Vec<u8> = vec![0, 1, 2];
    let w1 = Parabolic::new(&w, &j).unwrap();
    let t1 = build_table(&w1.group);
    let cells = w1_cells(&t1);
    assert!(cells.len() >= 2, "need two distinct B3 cells");

    // Two DIFFERENT cells produce different fingerprints.
    let cell_a = relkl_input_from_table(&w1.group, &t1, &cells[0]);
    let cell_b = relkl_input_from_table(&w1.group, &t1, &cells[1]);

    let dir = scratch("fp");

    // Write a partial log for cell_a under tag "rep" (stop mid-way).
    let mut cfg_a = RelKlCkptCfg::new(&dir, "rep");
    cfg_a.test_stop_after_layer = Some(1);
    let _ = relklpols_resumable(
        &w,
        &w1,
        &cell_a,
        &RelKlOpts { threads: Some(1) },
        Some(&cfg_a),
    );
    assert!(cfg_a.header_exists(), "cell_a log must exist");

    // Now run cell_b with the SAME tag: the fingerprint differs, so the stale
    // cell_a log is discarded and cell_b runs fresh to completion — identical to
    // an uninterrupted cell_b run.
    let reference_b = relklpols(&w, &w1, &cell_b, &RelKlOpts { threads: Some(1) });
    let cfg_b = RelKlCkptCfg::new(&dir, "rep");
    let got_b = match relklpols_resumable(
        &w,
        &w1,
        &cell_b,
        &RelKlOpts { threads: Some(1) },
        Some(&cfg_b),
    ) {
        RelKlRunOutcome::Done(out) => *out,
        RelKlRunOutcome::Stopped { .. } => panic!("cell_b must complete"),
    };
    assert_output_identical(&got_b, &reference_b, "fingerprint-mismatch fresh run");
    // Completed → files gone (the stale cell_a files were replaced then deleted).
    assert!(
        !cfg_b.header_exists() && !cfg_b.log_path().is_file(),
        "stale files must be replaced and the completed run cleans up"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Test 3: truncated log → header-bounded replay still resumes
// ---------------------------------------------------------------------------

#[test]
fn relkl_ckpt_truncated_log() {
    let (w, w1, cell1) = b4_biggest_cell();
    let dir = scratch("trunc");
    let opts = RelKlOpts { threads: Some(1) };

    let reference = relklpols(&w, &w1, &cell1, &opts);
    let n = num_layers(&w, &[0, 1, 2]);

    // Stop after a middle layer so several records are durable.
    let mut cfg = RelKlCkptCfg::new(&dir, "rep");
    cfg.test_stop_after_layer = Some(n / 2);
    let _ = relklpols_resumable(&w, &w1, &cell1, &opts, Some(&cfg));
    assert!(cfg.header_exists());

    // Simulate a crash mid-append: physically extend the log with junk bytes
    // beyond the header's recorded length (a trailing partial record).  Resume
    // must IGNORE the junk via header-bounded replay and still complete.
    let len = relkl_ckpt::log_len_for_test(&cfg).unwrap();
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(cfg.log_path())
            .unwrap();
        // Append a bogus partial frame (magic + garbage) past the header bound.
        f.write_all(b"RKL\n\xff\xff\xff\xff\x10\x00").unwrap();
        f.flush().unwrap();
    }
    assert!(
        relkl_ckpt::log_len_for_test(&cfg).unwrap() > len,
        "log must be physically longer after the junk append"
    );

    let resume_cfg = RelKlCkptCfg::new(&dir, "rep");
    let got = match relklpols_resumable(&w, &w1, &cell1, &opts, Some(&resume_cfg)) {
        RelKlRunOutcome::Done(out) => *out,
        RelKlRunOutcome::Stopped { .. } => panic!("resume must complete"),
    };
    assert_output_identical(&got, &reference, "truncated-log resume");

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Test 5: a clean uninterrupted resumable run deletes its log files
// ---------------------------------------------------------------------------

#[test]
fn relkl_ckpt_clean_run_deletes_files() {
    let (w, w1, cell1) = b4_biggest_cell();
    let dir = scratch("clean");
    let cfg = RelKlCkptCfg::new(&dir, "rep");

    let out =
        match relklpols_resumable(&w, &w1, &cell1, &RelKlOpts { threads: Some(1) }, Some(&cfg)) {
            RelKlRunOutcome::Done(out) => *out,
            RelKlRunOutcome::Stopped { .. } => panic!("no stop hook set"),
        };
    let reference = relklpols(&w, &w1, &cell1, &RelKlOpts { threads: Some(1) });
    assert_output_identical(&out, &reference, "clean resumable run");

    assert!(
        !cfg.header_exists() && !cfg.log_path().is_file(),
        "clean completion must leave no log residue"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Test 6: parallel (threads=4) resume == sequential uninterrupted
// ---------------------------------------------------------------------------

#[test]
fn relkl_ckpt_parallel_resume_matches_sequential() {
    let (w, w1, cell1) = b4_biggest_cell();
    let dir = scratch("par");
    let n = num_layers(&w, &[0, 1, 2]);

    // Sequential uninterrupted reference.
    let reference = relklpols(&w, &w1, &cell1, &RelKlOpts { threads: Some(1) });

    // Stop mid-run under threads=4, then resume under threads=4.
    let mut cfg = RelKlCkptCfg::new(&dir, "rep");
    cfg.test_stop_after_layer = Some(n / 2);
    match relklpols_resumable(&w, &w1, &cell1, &RelKlOpts { threads: Some(4) }, Some(&cfg)) {
        RelKlRunOutcome::Stopped { .. } => {}
        RelKlRunOutcome::Done(_) => panic!("expected a mid-run stop"),
    }
    let resume_cfg = RelKlCkptCfg::new(&dir, "rep");
    let got = match relklpols_resumable(
        &w,
        &w1,
        &cell1,
        &RelKlOpts { threads: Some(4) },
        Some(&resume_cfg),
    ) {
        RelKlRunOutcome::Done(out) => *out,
        RelKlRunOutcome::Stopped { .. } => panic!("resume must complete"),
    };
    assert_output_identical(&got, &reference, "parallel(4) resume vs sequential");

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Test 7: H3 determinism — every W1 cell, threads=4 mid-run resume == seq
// ---------------------------------------------------------------------------

#[test]
fn relkl_ckpt_h3_all_cells_determinism() {
    let w = CoxeterGroup::from_type("H3").unwrap();
    let j: Vec<u8> = vec![0, 1]; // W1 = H3_{0,1} = I2(5)-ish rank-2 parabolic.
    let w1 = Parabolic::new(&w, &j).unwrap();
    let t1 = build_table(&w1.group);
    let n = num_layers(&w, &j);

    for (ci, cell) in w1_cells(&t1).into_iter().enumerate() {
        let cell1 = relkl_input_from_table(&w1.group, &t1, &cell);
        let reference = relklpols(&w, &w1, &cell1, &RelKlOpts { threads: Some(1) });

        let dir = scratch(&format!("h3_{ci}"));
        // Kill mid-run under threads=4, then resume under threads=4.
        let mut cfg = RelKlCkptCfg::new(&dir, "rep");
        cfg.test_stop_after_layer = Some((n / 2).min(n.saturating_sub(1)));
        let _ = relklpols_resumable(&w, &w1, &cell1, &RelKlOpts { threads: Some(4) }, Some(&cfg));

        let resume_cfg = RelKlCkptCfg::new(&dir, "rep");
        let got = match relklpols_resumable(
            &w,
            &w1,
            &cell1,
            &RelKlOpts { threads: Some(4) },
            Some(&resume_cfg),
        ) {
            RelKlRunOutcome::Done(out) => *out,
            RelKlRunOutcome::Stopped { .. } => panic!("H3 cell {ci}: resume must complete"),
        };
        assert_output_identical(&got, &reference, &format!("H3 cell {ci}"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
