#!/bin/bash
# The evaluation matrix: baseline (main) vs branch, on both sessions. Run after the
# rebuilds (bench/private/*/{base,branch,branch-geminiverify}.db) finish.
#
#   bench/run_matrix.sh            # everything not already scored
#   FORCE=1 bench/run_matrix.sh    # rescore everything
#
# Branch databases are dreamed on a copy (dream changes memory), so the rebuilt
# databases stay reusable. Results: bench/private/results/<name>.json; summary at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
P=bench/private
PY=.venv/bin/python
MAIN=../exp_discrete_main

score() {  # name db cases code
  local name=$1 db=$2 cases=$3 code=$4
  if [ -z "${FORCE:-}" ] && [ -f "$P/results/$name.json" ]; then echo "= $name (already scored)"; return; fi
  echo "= scoring $name"
  $PY bench/run_g46.py --name "$name" --db "$db" --cases "$cases" --code "$code" > "$P/results/$name.log" 2>&1
  tail -16 "$P/results/$name.log"
}

dreamed() {  # src -> copy with a dream run on it
  local src=$1 out=${1%.db}-dream.db
  if [ -z "${FORCE:-}" ] && [ -f "$out" ]; then echo "$out"; return; fi
  rm -f "$out"; $PY -c "import sqlite3,sys; a=sqlite3.connect(sys.argv[1]); b=sqlite3.connect(sys.argv[2]); a.backup(b)" "$src" "$out"
  DISCRETE_MEMORY_EXPORT_DIR=off $PY -m discrete.cli --db "$out" dream --force > "${out%.db}.log" 2>&1
  echo "$out"
}

mkdir -p $P/results
G=$P/g46/cases.yaml
L=$P/live1006/cases.yaml
score g46-base "$P/g46/base.db" "$G" "$MAIN"
score g46-branch "$(dreamed $P/g46/branch.db)" "$G" .
score g46-branch-geminiverify "$(dreamed $P/g46/branch-geminiverify.db)" "$G" .
score g46-base+branchretrieval "$(dreamed $P/g46/base.db)" "$G" .     # drafting held fixed: retrieval/dream effect
score live-base "$P/live1006/base.db" "$L" "$MAIN"
score live-branch "$(dreamed $P/live1006/branch.db)" "$L" .
echo; $PY bench/run_g46.py --compare g46-base g46-branch g46-branch-geminiverify g46-base+branchretrieval
echo; $PY bench/run_g46.py --compare live-base live-branch
