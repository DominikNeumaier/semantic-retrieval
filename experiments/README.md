# Reproducing the paper numbers

Every number and figure in both papers is computed from the committed data in
`results/` and `benchmark/` — nothing is hand-entered. Run scripts from the repo root.

## Paper II — Semantic Retrieval
```
python3 experiments/paper2/numbers.py
```
Prints every value in Paper II's tables, figures, and appendix (runtime R@1/R@5,
out-of-scope refusal, funnel, failure modes, bootstrap CIs + McNemar p-values,
design-time). Deterministic (seed 42). Reads `results/runtime/` and `results/design-time/`.

## Paper I — ORD-Bench
```
python3 experiments/paper1/disambiguation/run_disambiguation.py         # -24.5% / -28.3% structural, +27.7% embedding
python3 experiments/paper1/embedding_analysis/scripts/embedding_by_tier.py    # r=0.63, per-tier cosine
```
Figure PDFs are (re)built by the `make_*.py` scripts next to each analysis.

## Data
`results/runtime/<mode>/traces/<condition>/` and `results/design-time/traces/<method>/`.
Condition = `<method><state>[variant]` (e.g. `A1f`). Method letters:
S=Baseline, A=Embedding, B=Progressive, C=Graph, D=Agentic Tools, E=Agentic Raw, F=Agentic Hybrid.
