"""Entry point — run the design-time benchmark.

Reproduces the design-time evaluation reported in the thesis: 240 activity
cases from 30 BPMN/CMMN process models, 7 methods (S, A–F) on Clean-ORD.
Design-time runs on Clean-ORD only — the enrichment ablation lives at
runtime, where the same landscape is exercised in both states.

Usage:
    .venv/bin/python run_dt.py                       # all methods, Clean-ORD
    .venv/bin/python run_dt.py --methods A B         # subset
    .venv/bin/python run_dt.py --limit 5             # smoke test
    .venv/bin/python run_dt.py --processes machine_breakdown.bpmn
    .venv/bin/python run_dt.py --no-wipe             # append to existing results
"""

import sys

from src.design_time.dt_benchmark import main

if __name__ == "__main__":
    # Design-time evaluation runs on Clean-ORD only. Pin it here so users
    # do not accidentally reproduce with a different state.
    if not any(a.startswith("--state") for a in sys.argv[1:]):
        sys.argv.extend(["--state", "clean"])
    main()
