//! Criterion benchmarks for the KL engine.
//!
//! Groups benchmarked (rank ≤ 4–6 only, per project constraints):
//!   B4  (|W| = 384)   — medium equal-parameter
//!   H3  (|W| = 120)   — medium equal-parameter
//!   A5  (|W| = 720)   — larger equal-parameter  [sample_size = 10]
//!   F4  (|W| = 1152)  — large equal-parameter   [sample_size = 10]
//!
//! Parallel benches are F4-only (t = 2, 4, 8) — large enough to show speedup.
//! Unequal-parameter bench: B3 with weights [2, 1, 1].
//! Cell bench: CellData::from_table on a precomputed F4 table.
//!
//! Run with:
//!   cargo bench -p rustcox-core

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rustcox_core::{
    group::CoxeterGroup,
    kl::{klcells, klpolynomials, klpolynomials_seq, CellData, CellsOpts, KlOpts},
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn equal_opts(rank: usize, threads: Option<usize>) -> KlOpts {
    KlOpts {
        weights: vec![1; rank],
        threads,
        layer_chunk: None,
    }
}

// ---------------------------------------------------------------------------
// Sequential benches: B4, H3, A5, F4
// ---------------------------------------------------------------------------

fn bench_kl_seq(c: &mut Criterion) {
    let mut group = c.benchmark_group("kl_seq");

    // B4  |W| = 384
    {
        let g = CoxeterGroup::from_type("B4").expect("B4");
        let opts = equal_opts(g.rank, None);
        group.bench_function("B4", |b| {
            b.iter(|| klpolynomials_seq(&g, &opts).expect("B4 seq"))
        });
    }

    // H3  |W| = 120
    {
        let g = CoxeterGroup::from_type("H3").expect("H3");
        let opts = equal_opts(g.rank, None);
        group.bench_function("H3", |b| {
            b.iter(|| klpolynomials_seq(&g, &opts).expect("H3 seq"))
        });
    }

    // A5  |W| = 720  — keep sample_size small
    {
        let g = CoxeterGroup::from_type("A5").expect("A5");
        let opts = equal_opts(g.rank, None);
        group.sample_size(10).bench_function("A5", |b| {
            b.iter(|| klpolynomials_seq(&g, &opts).expect("A5 seq"))
        });
    }

    // F4  |W| = 1152 — keep sample_size small
    {
        let g = CoxeterGroup::from_type("F4").expect("F4");
        let opts = equal_opts(g.rank, None);
        group.sample_size(10).bench_function("F4", |b| {
            b.iter(|| klpolynomials_seq(&g, &opts).expect("F4 seq"))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Parallel benches: F4 × {t=2, t=4, t=8}
// ---------------------------------------------------------------------------

fn bench_kl_par(c: &mut Criterion) {
    let g = CoxeterGroup::from_type("F4").expect("F4");

    let mut group = c.benchmark_group("kl_par_F4");
    group.sample_size(10);

    for &t in &[2usize, 4, 8] {
        let opts = equal_opts(g.rank, Some(t));
        group.bench_with_input(BenchmarkId::new("threads", t), &t, |b, _| {
            b.iter(|| klpolynomials(&g, &opts).expect("F4 par"))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Unequal-parameter bench: B3 with weights [2, 1, 1]
// ---------------------------------------------------------------------------

fn bench_kl_uneq(c: &mut Criterion) {
    let mut group = c.benchmark_group("kl_uneq");

    let g = CoxeterGroup::from_type("B3").expect("B3");
    let opts = KlOpts {
        weights: vec![2, 1, 1],
        threads: None,
        layer_chunk: None,
    };
    group.bench_function("B3_w211", |b| {
        b.iter(|| klpolynomials_seq(&g, &opts).expect("B3 uneq seq"))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Cell bench: CellData::from_table on a precomputed F4 table
// ---------------------------------------------------------------------------

fn bench_cells(c: &mut Criterion) {
    let mut group = c.benchmark_group("cells");
    group.sample_size(10);

    // Compute the table once outside the benchmark loop.
    let g = CoxeterGroup::from_type("F4").expect("F4");
    let opts = equal_opts(g.rank, None);
    let table = klpolynomials_seq(&g, &opts).expect("F4 seq for cells bench");

    group.bench_function("F4", |b| b.iter(|| CellData::from_table(&table)));

    group.finish();
}

// ---------------------------------------------------------------------------
// klcells benches: B4, H3, F4 (sequential) + F4 on 4 threads (Task P6)
// ---------------------------------------------------------------------------

/// Benchmark the parabolic-induction `klcells` driver.  `threads = Some(1)` is
/// the sequential relative-KL wavefront; `cells_F4_t4` exercises the parallel
/// wavefront (4 threads) to show the speedup on the largest local group.
fn bench_klcells(c: &mut Criterion) {
    let mut group = c.benchmark_group("klcells");
    group.sample_size(10);

    let seq = |threads: Option<usize>| CellsOpts {
        all_cells: true,
        threads,
    };

    // B4 |W| = 384 (sequential).
    {
        let g = CoxeterGroup::from_type("B4").expect("B4");
        group.bench_function("cells_B4", |b| {
            b.iter(|| klcells(&g, &seq(Some(1))).expect("B4 cells"))
        });
    }

    // H3 |W| = 120 (sequential).
    {
        let g = CoxeterGroup::from_type("H3").expect("H3");
        group.bench_function("cells_H3", |b| {
            b.iter(|| klcells(&g, &seq(Some(1))).expect("H3 cells"))
        });
    }

    // F4 |W| = 1152 (sequential), and the 4-thread parallel wavefront.
    {
        let g = CoxeterGroup::from_type("F4").expect("F4");
        group.bench_function("cells_F4", |b| {
            b.iter(|| klcells(&g, &seq(Some(1))).expect("F4 cells seq"))
        });
        group.bench_function("cells_F4_t4", |b| {
            b.iter(|| klcells(&g, &seq(Some(4))).expect("F4 cells t4"))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion entry points
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_kl_seq,
    bench_kl_par,
    bench_kl_uneq,
    bench_cells,
    bench_klcells,
);
criterion_main!(benches);
