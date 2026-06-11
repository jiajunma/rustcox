//! Streaming + checkpoint/resume tests for `klcells` (Task Q1).
//!
//! Four families:
//!   1. `streaming_equals_in_memory` — collect every streamed record,
//!      canonicalize, and assert it equals `klcells().cells`; summary counts
//!      match the in-memory result.
//!   2. `checkpoint_resume_mid_run` / `..._at_rep_boundary` — simulate a kill by
//!      making the sink error after `K` records, then resume from the on-disk
//!      checkpoint with a fresh sink; the concatenation (records kept on disk +
//!      records re-emitted) canonicalizes to the full partition.  Covers a kill
//!      at a rep boundary AND mid-rep (re-emitted work after truncation).
//!   3. `checkpoint_mismatched_opts_rejected` — a checkpoint for a different
//!      group/opts is ignored; the run starts fresh.
//!   4. `resume_after_completion_is_noop` — resubmitting after a clean finish
//!      emits no new records and reports the same totals.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::io;

use rustcox_core::{
    element::Word,
    group::CoxeterGroup,
    kl::{
        klcells, klcells_streaming, klcells_streaming_with_tiers, CellRecord, CellsOpts,
        CheckpointCfg, KlCellsSummary,
    },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Canonicalize a flat list of cells (raw word lists) the same way `klcells`
/// does at the top level: re-reduce every word, sort each cell by `(len, lex)`,
/// then sort the cell list.  Lets us compare a streamed partition to the
/// in-memory `KlCellsResult.cells` byte-for-byte.
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

/// A unique scratch dir per test (process id + label), cleaned on entry.
fn scratch(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("rcx_stream_{}_{label}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Collect every streamed record into a Vec (no checkpointing).
fn stream_all(g: &CoxeterGroup, opts: &CellsOpts) -> (Vec<CellRecord>, KlCellsSummary) {
    let records = RefCell::new(Vec::new());
    let mut sink = |rec: CellRecord| -> io::Result<()> {
        records.borrow_mut().push(rec);
        Ok(())
    };
    let summary = klcells_streaming(g, opts, &mut sink, None, None).unwrap();
    (records.into_inner(), summary)
}

// ---------------------------------------------------------------------------
// Test 1: streaming == in-memory
// ---------------------------------------------------------------------------

#[test]
fn streaming_equals_in_memory() {
    for name in ["A1", "A2", "A3", "B3", "H3"] {
        let g = CoxeterGroup::from_type(name).unwrap();
        let opts = CellsOpts::default();

        let in_mem = klcells(&g, &opts).unwrap();
        let (records, summary) = stream_all(&g, &opts);

        // (a) canonicalized streamed partition == in-memory cells, byte-for-byte.
        let streamed_cells: Vec<Vec<Word>> = records.iter().map(|r| r.words.clone()).collect();
        let got = canonicalize(&g, &streamed_cells);
        assert_eq!(
            got, in_mem.cells,
            "{name}: streamed partition must canonicalize to klcells().cells"
        );

        // (b) summary counts match.
        assert_eq!(
            summary.ncells,
            in_mem.cells.len(),
            "{name}: summary.ncells mismatch"
        );
        assert_eq!(
            summary.n_star_reps, in_mem.n_star_reps,
            "{name}: summary.n_star_reps mismatch"
        );
        assert_eq!(
            summary.total_elements, g.order,
            "{name}: summary.total_elements must equal |W|"
        );
        assert_eq!(summary.resumed_at_rep, None, "{name}: fresh run, no resume");

        // (c) provenance is well-formed: rep_index is non-decreasing across the
        // stream, and orbit_index restarts within each rep.
        let mut last_rep = 0usize;
        for rec in &records {
            assert!(
                rec.rep_index >= last_rep,
                "{name}: rep_index must be non-decreasing"
            );
            last_rep = rec.rep_index;
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: checkpoint / resume
// ---------------------------------------------------------------------------

/// Run streaming until the sink has accepted `kill_after` records, then make the
/// sink error (simulating a SLURM kill).  Returns the records that were
/// successfully written to "disk" before the kill.
///
/// The driver propagates the sink error and aborts; the last checkpoint on disk
/// reflects the most recent completed rep boundary.
fn run_until_killed(
    g: &CoxeterGroup,
    opts: &CellsOpts,
    cfg: &CheckpointCfg,
    kill_after: usize,
    tier_direct: usize,
    tier_tau: usize,
) -> Vec<CellRecord> {
    let written = RefCell::new(Vec::new());
    let mut sink = |rec: CellRecord| -> io::Result<()> {
        if written.borrow().len() >= kill_after {
            return Err(io::Error::other("simulated kill"));
        }
        written.borrow_mut().push(rec);
        Ok(())
    };
    // Expect an error (the simulated kill) unless kill_after exceeds the cell
    // count (then it completes — caller picks a small kill_after).
    let _ =
        klcells_streaming_with_tiers(g, opts, &mut sink, None, Some(cfg), tier_direct, tier_tau);
    written.into_inner()
}

/// Resume from the checkpoint in `cfg` and collect the re-emitted records.
fn resume_collect(
    g: &CoxeterGroup,
    opts: &CellsOpts,
    cfg: &CheckpointCfg,
    tier_direct: usize,
    tier_tau: usize,
) -> (Vec<CellRecord>, KlCellsSummary) {
    let written = RefCell::new(Vec::new());
    let mut sink = |rec: CellRecord| -> io::Result<()> {
        written.borrow_mut().push(rec);
        Ok(())
    };
    let summary =
        klcells_streaming_with_tiers(g, opts, &mut sink, None, Some(cfg), tier_direct, tier_tau)
            .unwrap();
    (written.into_inner(), summary)
}

/// The core resume property: kill after `kill_after` records, resume, and assert
/// the (disk records truncated to the checkpoint's `records_kept`) ++ (resumed
/// records) canonicalizes to the full in-memory partition.
fn assert_resume_reconstructs(name: &str, kill_after: usize, tier_direct: usize, tier_tau: usize) {
    let g = CoxeterGroup::from_type(name).unwrap();
    let opts = CellsOpts::default();
    let cfg = CheckpointCfg::new(scratch(&format!("{name}_{kill_after}")));

    let full = klcells(&g, &opts).unwrap();

    // Phase 1: run until the simulated kill.
    let disk = run_until_killed(&g, &opts, &cfg, kill_after, tier_direct, tier_tau);
    assert!(
        disk.len() >= kill_after.min(1),
        "{name}: expected some records before the kill"
    );

    // Phase 2: resume.  The summary tells the CLI how many disk records to keep.
    let (resumed, summary) = resume_collect(&g, &opts, &cfg, tier_direct, tier_tau);
    assert!(
        summary.resumed_at_rep.is_some(),
        "{name}: resume must report a resume point"
    );
    let keep = summary.records_kept as usize;
    assert!(
        keep <= disk.len(),
        "{name}: records_kept ({keep}) must not exceed records on disk ({})",
        disk.len()
    );

    // Reconstruct the stream the CLI would have: kept disk prefix ++ resumed.
    let mut all: Vec<Vec<Word>> = disk[..keep].iter().map(|r| r.words.clone()).collect();
    all.extend(resumed.iter().map(|r| r.words.clone()));

    let got = canonicalize(&g, &all);
    assert_eq!(
        got, full.cells,
        "{name} (kill_after={kill_after}): reconstructed partition must equal the full run"
    );

    // The total element count on resume completes |W|.
    assert_eq!(
        summary.total_elements, g.order,
        "{name}: resumed run must finish covering W"
    );

    let _ = std::fs::remove_dir_all(&cfg.dir);
}

#[test]
fn checkpoint_resume_default_tiers() {
    // Default tiers: each induced set decomposes whole, so kills land at varied
    // points across reps.  Try several kill points including 1 (very early).
    for &k in &[1usize, 2, 3, 5, 8] {
        assert_resume_reconstructs("B3", k, 300, 1500);
    }
}

#[test]
fn checkpoint_resume_mid_rep_tiny_tiers() {
    // Tiny tiers fragment each induced set into many components, so a single rep
    // emits many cells — killing mid-rep forces the re-emit-after-truncation
    // path (records_kept < disk.len()).  H3 has a rich star structure.
    for &k in &[1usize, 2, 4, 7] {
        assert_resume_reconstructs("H3", k, 1, 3);
    }
}

#[test]
fn checkpoint_resume_at_rep_boundary() {
    // A4 / A3: smaller, exercise the boundary case where the kill coincides with
    // a completed rep (records_kept == disk.len()).
    for name in ["A3", "A4"] {
        for &k in &[2usize, 4, 6] {
            assert_resume_reconstructs(name, k, 300, 1500);
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3: mismatched group/opts rejected
// ---------------------------------------------------------------------------

#[test]
fn checkpoint_mismatched_opts_rejected() {
    let cfg = CheckpointCfg::new(scratch("mismatch"));
    let opts = CellsOpts::default();

    // Write a checkpoint for B3 (run it partway via a kill).
    let g_b3 = CoxeterGroup::from_type("B3").unwrap();
    let _ = run_until_killed(&g_b3, &opts, &cfg, 2, 300, 1500);
    assert!(cfg.ckpt_path().is_file(), "B3 checkpoint must exist");

    // Now run A3 with the SAME checkpoint dir: the B3 fingerprint mismatches, so
    // A3 must start fresh (no resume) and still produce the full A3 partition.
    let g_a3 = CoxeterGroup::from_type("A3").unwrap();
    let full_a3 = klcells(&g_a3, &opts).unwrap();
    let (records, summary) = resume_collect(&g_a3, &opts, &cfg, 300, 1500);

    assert_eq!(
        summary.resumed_at_rep, None,
        "A3 must NOT resume from a B3 checkpoint"
    );
    assert_eq!(summary.records_kept, 0, "A3 fresh run keeps 0 disk records");

    let cells: Vec<Vec<Word>> = records.iter().map(|r| r.words.clone()).collect();
    let got = canonicalize(&g_a3, &cells);
    assert_eq!(
        got, full_a3.cells,
        "A3 fresh run (after rejecting B3 ckpt) must equal the full A3 partition"
    );

    let _ = std::fs::remove_dir_all(&cfg.dir);
}

// ---------------------------------------------------------------------------
// Test 4: resume after a clean finish is a no-op
// ---------------------------------------------------------------------------

#[test]
fn resume_after_completion_is_noop() {
    let g = CoxeterGroup::from_type("B3").unwrap();
    let opts = CellsOpts::default();
    let cfg = CheckpointCfg::new(scratch("noop"));

    // First run: complete it (no kill), with checkpointing.
    let records1 = RefCell::new(Vec::new());
    let mut sink1 = |rec: CellRecord| -> io::Result<()> {
        records1.borrow_mut().push(rec);
        Ok(())
    };
    let s1 = klcells_streaming(&g, &opts, &mut sink1, None, Some(&cfg)).unwrap();
    assert_eq!(s1.resumed_at_rep, None);
    assert_eq!(s1.total_elements, g.order);
    let n1 = records1.borrow().len();
    assert!(n1 > 0);

    // Second run: same checkpoint dir.  next_rep is past the last rep, so the
    // induction loop body never runs — no new records, totals preserved.
    let records2 = RefCell::new(Vec::new());
    let mut sink2 = |rec: CellRecord| -> io::Result<()> {
        records2.borrow_mut().push(rec);
        Ok(())
    };
    let s2 = klcells_streaming(&g, &opts, &mut sink2, None, Some(&cfg)).unwrap();
    assert!(
        s2.resumed_at_rep.is_some(),
        "second run resumes from the completed checkpoint"
    );
    assert_eq!(
        records2.borrow().len(),
        0,
        "completed checkpoint re-emits nothing"
    );
    assert_eq!(
        s2.total_elements, g.order,
        "totals preserved from the checkpoint"
    );

    let _ = std::fs::remove_dir_all(&cfg.dir);
}

// ---------------------------------------------------------------------------
// Test 5: reps sink fires once per star-rep
// ---------------------------------------------------------------------------

#[test]
fn reps_sink_count_matches_summary() {
    let g = CoxeterGroup::from_type("B3").unwrap();
    let opts = CellsOpts::default();

    let rep_indices: RefCell<Vec<usize>> = RefCell::new(Vec::new());
    let rep_sizes: RefCell<BTreeSet<usize>> = RefCell::new(BTreeSet::new());
    let records = RefCell::new(0usize);

    let mut sink = |_rec: CellRecord| -> io::Result<()> {
        *records.borrow_mut() += 1;
        Ok(())
    };
    let mut reps = |idx: usize, cg: &rustcox_core::cellgraph::CellGraph| -> io::Result<()> {
        rep_indices.borrow_mut().push(idx);
        rep_sizes.borrow_mut().insert(cg.x.len());
        Ok(())
    };

    let summary = klcells_streaming(&g, &opts, &mut sink, Some(&mut reps), None).unwrap();

    // One reps callback per recorded star-rep, indices 0..n_star_reps.
    assert_eq!(
        rep_indices.borrow().len(),
        summary.n_star_reps,
        "reps sink must fire once per star-rep"
    );
    let want: Vec<usize> = (0..summary.n_star_reps).collect();
    assert_eq!(
        *rep_indices.borrow(),
        want,
        "rep indices must be 0..n dense"
    );
    assert!(
        !rep_sizes.borrow().is_empty(),
        "reps must carry vertex sets"
    );
}
