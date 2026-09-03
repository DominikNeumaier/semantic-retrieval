# Reproducing Paper II numbers

Run all commands from the repository root. They read committed traces under `results/retrieval/` and `results/orchestration/` and do not call external models.

```bash
python3 analysis/numbers.py
python3 analysis/statistics.py
python3 analysis/figures.py
python3 analysis/ambiguity_profile.py
```

- `numbers.py` verifies the headline runtime, Out-of-Scope, routing, and design-time values and fails if required traces are missing.
- `statistics.py` reproduces Dynamic bootstrap confidence intervals and paired McNemar p-values.
- `figures.py` prints the cost/precision and token-efficiency coordinates.
- `ambiguity_profile.py` reproduces the distractor-ambiguity table using the ORD-Bench ambiguity report.

## Published result layout

```text
results/retrieval/design-time/                 design-time traces and summaries
results/retrieval/runtime/skill_adjusted/      component-level gap retrieval
results/retrieval/runtime/dynamic/             forced singles + Multi-Hint subtraces
results/retrieval/runtime/out_of_scope/         refusal evaluation
results/retrieval/runtime/skill_guided/         skill-selection summary
results/orchestration/                          routing and planning traces
```

Condition labels use `<method><state>` where state `0` is Clean-ORD and state `1` is Enriched-ORD. Method letters are `S=Baseline`, `A=Embedding`, `B=Progressive`, `C=Graph`, `D=Agentic Tools`, `E=Agentic Raw`, and `F=Agentic Hybrid`.

The Dynamic paper score gives each of the 40 cases one unit: forced single-intent outcomes for `dy-01` to `dy-20`, and the mean of the pre-decomposed sub-query outcomes for `dy-21` to `dy-40`.
