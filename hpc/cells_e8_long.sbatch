#!/usr/bin/env bash
#SBATCH --job-name=rustcox-cells-e8
#SBATCH --partition=cpu
#SBATCH --account=yushilingroup
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=4-00:00:00
#SBATCH --output=results/%x-%j.out
#
# ===========================================================================
# E8 left cells by parabolic induction — multi-day, checkpoint/resume run.
# ===========================================================================
#
# |W(E8)| = 696,729,600.  Expected (Geck/PyCox literature): 101796 left cells,
# 106 star-class reps.  This was historically out of reach for PyCox; the
# rustcox streaming driver (Task Q1) makes it feasible by (a) never holding all
# cells in RAM — each cell is streamed to disk as a JSON line — and (b)
# checkpointing the compact persistent loop state after every W1-rep, so a
# SLURM timeout or node preemption loses at most one rep of work.
#
# ---------------------------------------------------------------------------
# Resume protocol (THE KEY POINT): just resubmit this same script.
# ---------------------------------------------------------------------------
#   sbatch hpc/cells_e8_long.sbatch          # first submission
#   sbatch hpc/cells_e8_long.sbatch          # after a timeout — auto-resumes
#
# On resubmit, `rustcox cells E8 --checkpoint-dir results/e8_ckpt ...` finds the
# matching checkpoint, recomputes the cheap E7 recursion (~60 s) to rebuild the
# W1 star-reps, restores celms/tot/registry, truncates the stream file to the
# checkpointed record count, and replays from the checkpointed rep.  No manual
# state surgery is ever needed.  A clean re-run after completion is a no-op
# (next_rep is past the last rep) and finishes in seconds.
#
# SIGTERM/SIGINT are NOT trapped (no unsafe, no extra crates): kill-safety comes
# entirely from checkpoint-after-every-rep.  Killing the process at ANY instant
# is safe — the next resubmit recovers from the last per-rep checkpoint.
#
# ---------------------------------------------------------------------------
# Storage analysis (provision the Lustre quota before submitting).
# ---------------------------------------------------------------------------
#   results/e8_cells.jsonl.gz   the cell stream, one JSON line per left cell:
#                               {"rep":i,"orbit":j,"cell":[[words...]]}.
#                               101796 cells covering 696,729,600 elements;
#                               element words average ~30-60 generators.
#                               gzipped estimate: ~2.5-5 GB.
#   results/e8_reps/reps/       the 106 star-rep W-graphs, NNNNNN.json.gz each.
#                               These are the mathematical payload (the W-graph
#                               data).  A few are very large (E7-induced cells);
#                               upper bound ~50 GB total, typically much less.
#   results/e8_ckpt/            the checkpoint (klcells.ckpt): version tag +
#                               group/opts fingerprint, next_rep, tot, the
#                               involution skip-set celms (≤ ~280k coxelms ≈
#                               9 MB as u32 arrays for E8), the star-rep
#                               registry fingerprints, and the stream record
#                               count.  ~10-20 MB; rewritten atomically (tmp +
#                               rename) after every rep.
#   Total headroom to provision: ~60 GB is comfortable.
#
# ---------------------------------------------------------------------------
# Build first on the login node (compute nodes have no internet):
#   export PATH=$HOME/.cargo/bin:$PATH && cargo build --release
# ===========================================================================
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results results/e8_ckpt results/e8_reps
BIN=./target/release/rustcox
T=$SLURM_CPUS_PER_TASK

echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$T date=$(date)"
if [ -f results/e8_ckpt/klcells.ckpt ]; then
  echo "found existing checkpoint — this run will AUTO-RESUME"
else
  echo "no checkpoint — fresh E8 run"
fi

echo "== E8 cells (streaming + checkpoint + save-reps, t=$T) =="
/usr/bin/time -v "$BIN" cells E8 \
  --threads "$T" \
  --stream results/e8_cells.jsonl.gz \
  --checkpoint-dir results/e8_ckpt \
  --save-reps results/e8_reps \
  --summary

# The summary line (ncells / nstarreps / order) prints on completion.  When the
# job is killed by the time limit before finishing, the above command is
# terminated; the last per-rep checkpoint is intact, and resubmitting this
# script resumes.  Only assert the expected counts once the run completes.
echo "== expected-count check (only meaningful after a COMPLETED run) =="
python3 - <<'PY' || echo "stream not yet complete — resubmit to continue"
import gzip, json, sys
n = 0
with gzip.open("results/e8_cells.jsonl.gz", "rt") as f:
    for _ in f:
        n += 1
print(f"E8: cell records on disk = {n} (expected 101796 when complete)")
sys.exit(0 if n == 101796 else 1)
PY

echo "DONE $(date)"
