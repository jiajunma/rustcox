#!/usr/bin/env bash
#SBATCH --job-name=rustcox-big
#SBATCH --partition=cpu
#SBATCH --account=yushilingroup
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=results/%x-%j.out
#
# Rank-6 full KL tables: D6 (23040), B6 (46080), E6 (51840).
# Summary-only (full JSON export at these sizes is impractical); D6 also runs
# sequentially so the summary lines can be cross-checked seq-vs-parallel.
# Submit from the repo root:  sbatch hpc/big_groups.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results
BIN=./target/release/rustcox
T=$SLURM_CPUS_PER_TASK

echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$T date=$(date)"

echo "== D6 parallel t=$T =="
/usr/bin/time -v $BIN kl D6 --threads "$T" --summary
echo "== D6 sequential (summary cross-check) =="
/usr/bin/time -v $BIN kl D6 --seq --summary

echo "== B6 parallel t=$T =="
/usr/bin/time -v $BIN kl B6 --threads "$T" --summary

echo "== E6 parallel t=$T =="
/usr/bin/time -v $BIN kl E6 --threads "$T" --summary

echo "DONE $(date)"
