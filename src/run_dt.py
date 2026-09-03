"""Entry point — run the design-time benchmark.

Reproduces the design-time evaluation reported in the thesis: 240 activity
cases from 30 BPMN/CMMN process models, 7 methods (S, A–F) on Clean-ORD.
Design-time runs on Clean-ORD only — the enrichment ablation lives at
runtime, where the same landscape is exercised in both states.

Usage (from the repository root):
    python3 -m src.run_dt                       # all methods, Clean-ORD
    python3 -m src.run_dt --methods A B         # subset
    python3 -m src.run_dt --limit 5             # smoke test
    python3 -m src.run_dt --processes proc_001.xml
    python3 -m src.run_dt --no-wipe             # append to existing results
"""

import sys

from src.eval.design_time import main

if __name__ == "__main__":
    # Design-time evaluation runs on Clean-ORD only. Pin it here so users
    # do not accidentally reproduce with a different state.
    if not any(a.startswith("--state") for a in sys.argv[1:]):
        sys.argv.extend(["--state", "clean"])
    main()
