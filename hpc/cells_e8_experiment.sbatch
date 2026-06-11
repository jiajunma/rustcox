#!/usr/bin/env bash
#SBATCH --job-name=rustcox-cells-e8x
#SBATCH --partition=cpu
#SBATCH --account=yushilingroup
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=03:00:00
#SBATCH --output=results/%x-%j.out
#
# EXPERIMENT (time-boxed, expected to fail): E8 left cells by induction.
# PyCox documentation states klcells E8 is out of reach; the structural
# blocker is the relative-KL matrix for large E7 cells (240 coset reps ×
# |C|² slots per comparable pair → TB-scale for |C| ≳ 5000). This run
# documents where and how the limit is hit. Gate passed per plan P7:
# E7 finished in 61 s / 6.5 GB. Whatever happens, the outcome is recorded.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results
echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK date=$(date)"
echo "EXPERIMENT: expected outcome is OOM/timeout at the first large induced set."
/usr/bin/time -v ./target/release/rustcox cells E8 \
  --threads "$SLURM_CPUS_PER_TASK" --summary -o results/cells_E8.json.gz \
  || echo "E8_EXPERIMENT_FAILED (exit $?) — expected; see analysis in hpc/ comments"
echo "DONE $(date)"
