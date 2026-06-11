#!/usr/bin/env bash
#SBATCH --job-name=rustcox-killtest
#SBATCH --partition=cpu
#SBATCH --account=yushilingroup
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --output=results/%x-%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
BIN=./target/release/rustcox
rm -rf results/e7_kr results/e7_kr.jsonl.gz; mkdir -p results/e7_kr
echo "== phase 1: start E7 streaming, kill -9 after 25s =="
$BIN cells E7 --threads 64 --stream results/e7_kr.jsonl.gz --checkpoint-dir results/e7_kr --summary &
PID=$!
sleep 25
kill -9 $PID || true
wait $PID || echo "killed as planned (exit $?)"
ls -la results/e7_kr/ results/e7_kr.jsonl.gz || true
echo "== phase 2: resume =="
$BIN cells E7 --threads 64 --stream results/e7_kr.jsonl.gz --checkpoint-dir results/e7_kr --summary
echo "== phase 3: verify vs archived E7 =="
python3 - <<'PY'
import gzip, json, sys
stream = [json.loads(l) for l in gzip.open("results/e7_kr.jsonl.gz", "rt")]
cells = sorted(sorted(r["cell"], key=lambda w: (len(w), w)) for r in stream)
ref = json.load(gzip.open("results/cells_E7.json.gz", "rt"))["cells"]
ref = sorted(sorted(c, key=lambda w: (len(w), w)) for c in ref)
print("stream cells:", len(cells), "ref:", len(ref))
sys.exit(0 if cells == ref else 1)
PY
echo "KILL_RESUME_OK $(date)"
