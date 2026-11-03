#!/usr/bin/env bash
# One-command run (Linux / macOS / Git-Bash):  ./run_all.sh [--quick]
set -e
cd "$(dirname "$0")"
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -r requirements.txt
python run_pipeline.py "$@"                       # train + all experiments + figures
python -m unittest discover -s tests -v           # self-tests
python report/build_report.py                     # Word report
echo "Starting web app on http://127.0.0.1:8080  (Ctrl+C to stop)"
python webapp/app.py
