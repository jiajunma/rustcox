#!/usr/bin/env bash
#SBATCH --job-name=rustcox-cells-med
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
# Cells-by-induction validation ladder: H4, D6, B6, E6.
# Expected left-cell counts (pinned by the Phase-1 full-table HPC runs):
#   H4 = 206, D6 = 578, B6 = 752, E6 = 652.
# H4 additionally gets an exact partition cross-check against the archived
# full-table document results/h4_par.json.gz.
# Submit from the repo root:  sbatch hpc/cells_medium.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results
BIN=./target/release/rustcox
T=$SLURM_CPUS_PER_TASK

echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$T date=$(date)"

echo "== selftest (51 golden files incl. cells) =="
$BIN selftest

for G in H4 D6 B6 E6; do
  echo "== cells $G (t=$T) =="
  /usr/bin/time -v $BIN cells "$G" --threads "$T" --summary -o "results/cells_$G.json.gz" 2>&1 \
    | grep -E "ncells=|Maximum resident|Elapsed \(wall"
done

echo "== expected-count assertions =="
python3 - <<'PY'
import gzip, json, sys
expect = {"H4": 206, "D6": 578, "B6": 752, "E6": 652}
ok = True
for g, n in expect.items():
    d = json.load(gzip.open(f"results/cells_{g}.json.gz", "rt"))
    got = d["ncells"]
    print(f"{g}: ncells={got} expected={n} {'OK' if got==n else 'MISMATCH'}")
    ok &= (got == n)
sys.exit(0 if ok else 1)
PY

echo "== H4 exact partition cross-check vs full-table archive =="
python3 - <<'PY'
import gzip, json, sys
cells = json.load(gzip.open("results/cells_H4.json.gz", "rt"))["cells"]
full = json.load(gzip.open("results/h4_par.json.gz", "rt"))
elms, lcells = full["elms"], full["lcells"]
full_cells = sorted(sorted([elms[i] for i in c], key=lambda w: (len(w), w))
                    for c in lcells)
ind_cells = sorted(sorted(c, key=lambda w: (len(w), w)) for c in cells)
if full_cells == ind_cells:
    print("H4_PARTITION_OK (206 cells byte-identical to full-table run)")
else:
    print("H4_PARTITION_MISMATCH"); sys.exit(1)
PY

echo "DONE $(date)"
