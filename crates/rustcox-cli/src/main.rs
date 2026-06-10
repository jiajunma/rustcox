//! rustcox CLI — Kazhdan-Lusztig polynomials and cells for finite Coxeter groups.

mod commands;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use commands::{
    bench::{run as bench_run, BenchArgs},
    info::run as info_run,
    kl::{run as kl_run, KlArgs, WeightSpec},
    selftest::run as selftest_run,
    verify::run as verify_run,
};

#[derive(Parser)]
#[command(
    name = "rustcox",
    about = "Kazhdan-Lusztig polynomials for finite Coxeter groups"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Print group information: type, rank, order, N, degrees, Coxeter matrix.
    Info {
        /// Coxeter group type string, e.g. "B4", "A2xA1", "I5".
        #[arg(name = "TYPE")]
        type_str: String,
    },

    /// Compute the full KL table and cells.
    Kl {
        /// Coxeter group type string.
        #[arg(name = "TYPE")]
        type_str: String,

        /// Generator weights: a single integer K (all equal) or comma-separated
        /// list w0,w1,... (e.g. "2,1" or "K").  Default: all ones.
        #[arg(long, value_name = "WEIGHTS")]
        weights: Option<String>,

        /// Number of threads for parallel computation.
        #[arg(long, value_name = "N")]
        threads: Option<usize>,

        /// Layer chunk size for parallel computation.
        #[arg(long, value_name = "K")]
        layer_chunk: Option<usize>,

        /// Print a one-line summary of space-separated key=value pairs.
        ///
        /// Stable keys (in order): npols mues lcells rcells tcells duflo arrows checks_ok.
        /// The time= field is always last and is NOT stable for scripting (hardware-dependent).
        #[arg(long)]
        summary: bool,

        /// Write canonical JSON to FILE (gz if filename ends .gz).
        #[arg(short = 'o', value_name = "FILE")]
        output: Option<String>,

        /// Use the sequential reference driver instead of the parallel one.
        #[arg(long)]
        seq: bool,
    },

    /// Compare two JSON documents (gz-transparent).
    ///
    /// Exit code 0 = match, 1 = mismatch, 2 = argument/IO error.
    Verify {
        /// Path to the file to verify.
        file: String,

        /// Path to the golden file to verify against.
        #[arg(long)]
        against: String,
    },

    /// Run golden-file self-tests.
    ///
    /// Reads every kl_*.json[.gz] and basics_*.json[.gz] from the golden
    /// directory, rebuilds the group and computations, and compares.
    /// Exit code 0 iff all tests pass.
    ///
    /// Golden files whose type contains an I2(m) component with m not in
    /// {3, 4, 5, 6} are reported as SKIP (pending CyclotomicInteger support).
    Selftest {
        /// Directory containing golden files.  Default: ./golden
        #[arg(long, value_name = "DIR", default_value = "golden")]
        golden_dir: PathBuf,
    },

    /// Quick wall-time benchmark table.
    BenchKl {
        /// Coxeter group type string.
        #[arg(name = "TYPE")]
        type_str: String,

        /// Comma-separated list of thread counts to benchmark.  Default: 1,2,4
        #[arg(long, value_name = "COUNTS")]
        threads: Option<String>,

        /// Generator weights (same format as `kl --weights`).
        #[arg(long, value_name = "WEIGHTS")]
        weights: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Info { type_str } => info_run(&type_str),

        Commands::Kl {
            type_str,
            weights,
            threads,
            layer_chunk,
            summary,
            output,
            seq,
        } => {
            let weight_spec = match weights.as_deref() {
                None => None,
                Some(s) => match WeightSpec::parse(s) {
                    Ok(ws) => Some(ws),
                    Err(e) => {
                        eprintln!("error: {e}");
                        std::process::exit(2);
                    }
                },
            };

            kl_run(KlArgs {
                type_str,
                weight_spec,
                threads,
                layer_chunk,
                summary,
                output,
                seq,
            })
        }

        Commands::Verify { file, against } => match verify_run(&file, &against) {
            Ok(true) => std::process::exit(0),
            Ok(false) => std::process::exit(1),
            Err(e) => {
                eprintln!("error: {e:#}");
                std::process::exit(2);
            }
        },

        Commands::Selftest { golden_dir } => match selftest_run(&golden_dir) {
            Ok(true) => std::process::exit(0),
            Ok(false) => std::process::exit(1),
            Err(e) => {
                eprintln!("error: {e:#}");
                std::process::exit(2);
            }
        },

        Commands::BenchKl {
            type_str,
            threads,
            weights,
        } => {
            let thread_counts = match threads.as_deref() {
                None => vec![],
                Some(s) => {
                    let result: Result<Vec<usize>, _> =
                        s.split(',').map(|p| p.trim().parse::<usize>()).collect();
                    match result {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("error: invalid --threads value '{s}': {e}");
                            std::process::exit(2);
                        }
                    }
                }
            };

            let weight_spec = match weights.as_deref() {
                None => None,
                Some(s) => match WeightSpec::parse(s) {
                    Ok(ws) => Some(ws),
                    Err(e) => {
                        eprintln!("error: {e}");
                        std::process::exit(2);
                    }
                },
            };

            bench_run(BenchArgs {
                type_str,
                threads: thread_counts,
                weight_spec,
            })
        }
    };

    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
