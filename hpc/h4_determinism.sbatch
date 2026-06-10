#!/usr/bin/env bash
#SBATCH --job-name=rustcox-h4
#SBATCH --partition=cpu
#SBATCH --account=yushilingroup
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=results/%x-%j.out
#
# H4 full KL table (|W| = 14400) — first run at this scale.
# Proves the determinism guarantee end-to-end: the sequential and the
# 64-thread parallel drivers must produce byte-identical canonical JSON.
# Submit from the repo root:  sbatch hpc/h4_determinism.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results
BIN=./target/release/rustcox

echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK date=$(date)"

echo "== selftest =="
$BIN selftest

echo "== H4 sequential =="
/usr/bin/time -v $BIN kl H4 --seq --summary -o results/h4_seq.json

echo "== H4 parallel t=$SLURM_CPUS_PER_TASK =="
/usr/bin/time -v $BIN kl H4 --threads "$SLURM_CPUS_PER_TASK" --summary -o results/h4_par.json

echo "== byte-level determinism check =="
cmp results/h4_seq.json results/h4_par.json && echo "DETERMINISM_OK (byte-identical)"

ls -l results/h4_seq.json
gzip -9 -f results/h4_par.json
rm -f results/h4_seq.json
echo "DONE $(date)"
