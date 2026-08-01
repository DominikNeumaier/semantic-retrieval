"""Entry point — run the runtime benchmark.

Reproduces the runtime evaluation reported in the thesis: 110 test cases
split across four expected modes (30 Skill-Guided, 20 Skill-Adjusted,
40 Dynamic, 20 Out-of-Scope), 7 methods (S, A–F), 2 ORD states
(clean, enriched). Skill-Guided cases measure routing accuracy only —
they do not run retrieval.

The evaluation reported in the thesis used three passes over the runtime
cases:

  1) Baseline pass (this file, default flags): resolver + planner run
     for every case. Records land with condition A0, A1, ..., F1.

  2) Fair-retrieval pass for Dynamic (--force-mode dynamic): resolver
     and planner are bypassed so every Dynamic case reaches retrieval
     regardless of routing errors. Records get an `f` suffix
     (A0f, A1f, ...) and are analysed alongside the baseline.

  3) Multi-hint pass for Dynamic multi-intent cases (run_mh_benchmark):
     the pre-decomposed sub-queries of dy-21..dy-40 are fed to retrieval
     one at a time. Records get an `mh` suffix (A0mh, A1mh, ...).

Usage:
    .venv/bin/python run_rt.py                              # baseline pass
    .venv/bin/python run_rt.py --force-mode dynamic --mode dynamic
    .venv/bin/python -m src.runtime.run_mh_benchmark        # multi-hint pass

    .venv/bin/python run_rt.py --methods A                  # single method
    .venv/bin/python run_rt.py --states enriched --limit 6  # smoke test
    .venv/bin/python run_rt.py --mode skill_guided          # single mode
    .venv/bin/python run_rt.py --no-wipe                    # resume after interruption
"""

from src.runtime.rt_benchmark import main

if __name__ == "__main__":
    main()
