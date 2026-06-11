//! `rustcox cells <TYPE> [options]` — left cells by parabolic induction.
//!
//! Computes the left-cell partition of a finite Coxeter group with the
//! `klcells` driver (parabolic induction; equal parameters).  Prints a one-line
//! summary and/or writes a canonical `kind: "cells"` JSON document matching the
//! golden `cells_*` format, so `rustcox verify` can compare it against a golden.

use std::time::{Duration, Instant};

use anyhow::Context;
use rustcox_core::{
    group::CoxeterGroup,
    io::cells_json_doc,
    kl::{klcells, CellsOpts, KlCellsResult},
};

use super::kl::write_json_doc;

pub struct CellsArgs {
    pub type_str: String,
    pub threads: Option<usize>,
    pub summary: bool,
    pub output: Option<String>,
}

pub fn run(args: CellsArgs) -> anyhow::Result<()> {
    let group = CoxeterGroup::from_type(&args.type_str)
        .with_context(|| format!("invalid type '{}'", args.type_str))?;

    let opts = CellsOpts {
        all_cells: true,
        threads: args.threads,
    };

    let t0 = Instant::now();
    let res = klcells(&group, &opts).with_context(|| "klcells computation failed")?;
    let elapsed = t0.elapsed();

    // Print the summary when explicitly asked, or by default when not writing a
    // file (mirrors the `kl` subcommand's behaviour).
    let show_summary = args.summary || args.output.is_none();
    if show_summary {
        print_summary(&group, &res, elapsed);
    }

    if let Some(ref out_path) = args.output {
        let doc = cells_json_doc(&group, &res);
        write_json_doc(out_path, &doc)?;
    }

    Ok(())
}

/// Print the one-line cells summary to stdout.
///
/// Format (space-separated `key=value`; `time` last and not stable for
/// scripting):
///
/// ```text
/// ncells=<n> nstarreps=<n> order=<n> time=<seconds>s
/// ```
fn print_summary(group: &CoxeterGroup, res: &KlCellsResult, elapsed: Duration) {
    println!(
        "ncells={} nstarreps={} order={} time={:.3}s",
        res.cells.len(),
        res.n_star_reps,
        group.order,
        elapsed.as_secs_f64()
    );
}
