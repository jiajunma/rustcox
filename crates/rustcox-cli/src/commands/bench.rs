//! `rustcox bench-kl <TYPE> [--threads list,of,counts] [--weights ...]` —
//! simple wall-time benchmark table.

use std::time::Instant;

use anyhow::Context;
use rustcox_core::{
    group::CoxeterGroup,
    kl::{klpolynomials, KlOpts},
};

use super::kl::WeightSpec;

pub struct BenchArgs {
    pub type_str: String,
    pub threads: Vec<usize>,
    pub weight_spec: Option<WeightSpec>,
}

pub fn run(args: BenchArgs) -> anyhow::Result<()> {
    let group = CoxeterGroup::from_type(&args.type_str)
        .with_context(|| format!("invalid type '{}'", args.type_str))?;

    let weights = match &args.weight_spec {
        Some(spec) => spec.resolve(group.rank),
        None => vec![1u32; group.rank],
    };

    // Use default thread counts if none specified
    let thread_counts = if args.threads.is_empty() {
        vec![1, 2, 4]
    } else {
        args.threads.clone()
    };

    println!(
        "bench-kl {} weights=[{}]",
        args.type_str,
        weights
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("{:<10} {:>12} {:>10}", "threads", "time(s)", "speedup");
    println!("{}", "-".repeat(35));

    let mut baseline: Option<f64> = None;

    for &t in &thread_counts {
        let opts = KlOpts {
            weights: weights.clone(),
            threads: Some(t),
            layer_chunk: None,
        };

        let t0 = Instant::now();
        klpolynomials(&group, &opts)
            .with_context(|| format!("klpolynomials failed for {} threads={t}", args.type_str))?;
        let elapsed = t0.elapsed().as_secs_f64();

        let speedup = match baseline {
            None => {
                baseline = Some(elapsed);
                1.0
            }
            Some(base) => base / elapsed,
        };

        println!("{:<10} {:>12.4} {:>10.2}x", t, elapsed, speedup);
    }

    Ok(())
}
