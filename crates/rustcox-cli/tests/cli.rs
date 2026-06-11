//! Integration tests for the rustcox CLI binary.
//!
//! Uses `std::process::Command` to drive the compiled binary.

use std::{fs, path::PathBuf, process::Command};

/// Return the path to the compiled `rustcox` binary.
fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_rustcox"))
}

/// Return the workspace root (two levels up from the cli crate manifest).
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn golden_dir() -> PathBuf {
    workspace_root().join("golden")
}

// ---------------------------------------------------------------------------
// Test 1: info B4 — output contains "order" and "384"
// ---------------------------------------------------------------------------

#[test]
fn info_b4() {
    let out = Command::new(bin())
        .args(["info", "B4"])
        .output()
        .expect("failed to run rustcox info B4");

    assert!(
        out.status.success(),
        "info B4 failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("order"),
        "stdout should contain 'order': {stdout}"
    );
    assert!(
        stdout.contains("384"),
        "stdout should contain '384' (order of B4): {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: kl B2 --weights 2,1 --summary — exit 0, npols=3, lcells=6
// ---------------------------------------------------------------------------

#[test]
fn kl_summary_b2_uneq() {
    let out = Command::new(bin())
        .args(["kl", "B2", "--weights", "2,1", "--summary"])
        .output()
        .expect("failed to run rustcox kl B2 --weights 2,1 --summary");

    assert!(
        out.status.success(),
        "kl B2 --weights 2,1 --summary failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("npols=3"),
        "stdout should contain 'npols=3': {stdout}"
    );
    assert!(
        stdout.contains("lcells=6"),
        "stdout should contain 'lcells=6': {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: kl B3 -o tmpdir/b3.json; verify against golden; corrupt → exit 1
// ---------------------------------------------------------------------------

#[test]
fn kl_export_then_verify() {
    let tmp = tempdir();

    // Compute and export B3 KL table
    let b3_path = tmp.join("b3.json");
    let out = Command::new(bin())
        .args(["kl", "B3", "-o", b3_path.to_str().unwrap()])
        .output()
        .expect("failed to run rustcox kl B3 -o b3.json");
    assert!(
        out.status.success(),
        "kl B3 export failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(b3_path.exists(), "b3.json should have been created");

    // Verify against the golden file
    let golden_b3 = golden_dir().join("kl_B3_w1.json");
    let out = Command::new(bin())
        .args([
            "verify",
            b3_path.to_str().unwrap(),
            "--against",
            golden_b3.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run verify");
    assert!(
        out.status.success(),
        "verify against golden should exit 0: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Corrupt the JSON: replace first duflo entry's first number slightly
    let content = fs::read_to_string(&b3_path).expect("read b3.json");
    // The duflo key contains arrays like [0,0,1],[1,...] — corrupt one digit
    let corrupted = content.replacen(r#""duflo":[[0,"#, r#""duflo":[[99,"#, 1);
    assert_ne!(
        content, corrupted,
        "corruption should have changed the file"
    );

    let corrupt_path = tmp.join("b3_corrupt.json");
    fs::write(&corrupt_path, corrupted).expect("write corrupt file");

    let out = Command::new(bin())
        .args([
            "verify",
            corrupt_path.to_str().unwrap(),
            "--against",
            golden_b3.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run verify on corrupt file");
    assert_eq!(
        out.status.code(),
        Some(1),
        "verify of corrupted file should exit 1: stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

// ---------------------------------------------------------------------------
// Test 4: kl B2 --weights 2,2,1 → exit nonzero, stderr mentions weights
// ---------------------------------------------------------------------------

#[test]
fn kl_weights_len_error() {
    let out = Command::new(bin())
        .args(["kl", "B2", "--weights", "2,2,1"])
        .output()
        .expect("failed to run rustcox kl B2 --weights 2,2,1");

    assert!(!out.status.success(), "kl B2 --weights 2,2,1 should fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.to_lowercase().contains("weight"),
        "stderr should mention 'weight': {stderr}"
    );
}

// ---------------------------------------------------------------------------
// Test 5: selftest --golden-dir golden — exit 0; every file PASS, no SKIPs.
// Task 18 enabled the previously-skipped I7/I8 cyclotomic dihedral goldens.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow: A5+F4 each take several seconds even in debug"]
fn selftest_passes() {
    let gdir = golden_dir();
    let out = Command::new(bin())
        .args(["selftest", "--golden-dir", gdir.to_str().unwrap()])
        .output()
        .expect("failed to run rustcox selftest");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(
        out.status.success(),
        "selftest should exit 0: stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("PASS"),
        "stdout should contain 'PASS': {stdout}"
    );
    assert!(
        !stdout.contains("FAIL"),
        "stdout should not contain 'FAIL': {stdout}"
    );
    // With CycInt support there is nothing left to skip — every golden computes.
    assert!(
        !stdout.contains("SKIP"),
        "stdout should contain no 'SKIP' lines: {stdout}"
    );
    // The cyclotomic dihedral goldens must be present and passing.
    for needle in [
        "PASS  basics_I7",
        "PASS  basics_I8",
        "PASS  kl_I7",
        "PASS  kl_I8",
    ] {
        assert!(
            stdout.contains(needle),
            "selftest output missing '{needle}': {stdout}"
        );
    }
    // The Phase-2 cells goldens (kind: "cells") must be present and passing.
    for needle in [
        "PASS  cells_B3",
        "PASS  cells_B4",
        "PASS  cells_F4",
        "PASS  cells_H3",
    ] {
        assert!(
            stdout.contains(needle),
            "selftest output missing cells golden '{needle}': {stdout}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5b: cells F4 --summary — stdout contains ncells=72, nstarreps=29
// ---------------------------------------------------------------------------

#[test]
fn cells_summary_f4() {
    let out = Command::new(bin())
        .args(["cells", "F4", "--summary"])
        .output()
        .expect("failed to run rustcox cells F4 --summary");

    assert!(
        out.status.success(),
        "cells F4 --summary failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ncells=72"),
        "stdout should contain 'ncells=72': {stdout}"
    );
    assert!(
        stdout.contains("nstarreps=29"),
        "stdout should contain 'nstarreps=29': {stdout}"
    );
    assert!(
        stdout.contains("order=1152"),
        "stdout should contain 'order=1152': {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 5c: cells B3 -o tmp; verify against golden/cells_B3.json — exit 0
// ---------------------------------------------------------------------------

#[test]
fn cells_export_verify() {
    let tmp = tempdir();

    // Compute and export the B3 cells document.
    let b3_path = tmp.join("cells_b3.json");
    let out = Command::new(bin())
        .args(["cells", "B3", "-o", b3_path.to_str().unwrap()])
        .output()
        .expect("failed to run rustcox cells B3 -o");
    assert!(
        out.status.success(),
        "cells B3 export failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(b3_path.exists(), "cells_b3.json should have been created");

    // Verify against the golden cells file — must match byte-for-byte (exit 0).
    let golden_b3 = golden_dir().join("cells_B3.json");
    let out = Command::new(bin())
        .args([
            "verify",
            b3_path.to_str().unwrap(),
            "--against",
            golden_b3.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run verify on cells");
    assert!(
        out.status.success(),
        "verify cells B3 against golden should exit 0: stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

// ---------------------------------------------------------------------------
// Test 6: bench-kl B3 --threads 1,2 — exit 0
// ---------------------------------------------------------------------------

#[test]
fn bench_smoke() {
    let out = Command::new(bin())
        .args(["bench-kl", "B3", "--threads", "1,2"])
        .output()
        .expect("failed to run rustcox bench-kl B3 --threads 1,2");

    assert!(
        out.status.success(),
        "bench-kl B3 --threads 1,2 failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a temporary directory and return its path.
/// The caller is responsible for cleanup (we rely on test isolation — the OS
/// will clean up /tmp on reboot, and tests each get a unique subdir).
fn tempdir() -> PathBuf {
    let base = std::env::temp_dir().join(format!(
        "rustcox_cli_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&base).expect("create temp dir");
    base
}
