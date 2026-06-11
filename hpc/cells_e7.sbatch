#!/usr/bin/env bash
#SBATCH --job-name=rustcox-cells-e7
#SBATCH --partition=cpu
#SBATCH --account=yushilingroup
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=results/%x-%j.out
#
# E7 left cells by parabolic induction (|W| = 2,903,040).
# Expected (PyCox documentation, Geck): 6364 left cells, 56 star-class reps.
# PyCox needed ~4 hours; rustcox target: well under 1 hour.
# Submit from the repo root AFTER cells_medium passes:
#   sbatch hpc/cells_e7.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results
BIN=./target/release/rustcox
T=$SLURM_CPUS_PER_TASK

echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$T date=$(date)"

echo "== E7 cells (t=$T) =="
/usr/bin/time -v $BIN cells E7 --threads "$T" --summary -o results/cells_E7.json.gz

echo "== expected-count assertion =="
python3 - <<'PY'
import gzip, json, sys
d = json.load(gzip.open("results/cells_E7.json.gz", "rt"))
print(f"E7: ncells={d['ncells']} nstarreps={d['nstarreps']} "
      f"(expected 6364 / 56)")
sys.exit(0 if (d["ncells"] == 6364 and d["nstarreps"] == 56) else 1)
PY

echo "DONE $(date)"
