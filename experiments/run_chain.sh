#!/usr/bin/env bash
# Runs after 03_run_full_whisper.py finishes:
#   04 (Indic comparator) -> 05 (case study) -> 06 (figures)
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

# Wait for any in-flight 03_run_full_whisper.py to finish.
while pgrep -fa "03_run_full_whisper" > /dev/null; do
  sleep 5
done

mkdir -p results
LOG=results/chain.log
{
  echo "=== chain start: $(date) ==="
  echo "--- 04 Indic comparator ---"
  python -u experiments/04_run_indic_comparator.py
  echo "--- 05 case study ---"
  python -u experiments/05_run_case_study.py
  echo "--- 06 figures ---"
  python -u experiments/06_make_figures_full.py
  echo "=== chain done: $(date) ==="
} > "$LOG" 2>&1
