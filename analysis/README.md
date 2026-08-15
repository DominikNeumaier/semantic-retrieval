# Reproducing the paper numbers

Every number and figure in the Semantic Retrieval paper is computed from the
committed traces under `results/` — nothing is hand-entered. Run scripts from
the repo root.

## Numbers
```
python3 analysis/numbers.py
```
Prints every value in the paper's tables, figures, and appendix (runtime
R@1/R@5, out-of-scope refusal, funnel, failure modes, design-time). Deterministic
(seed 42). Reads `results/runtime/` and `results/design-time/`.

## Statistical reliability
```
python3 analysis/statistics.py
```
Reproduces the bootstrap confidence intervals and paired McNemar p-values for
the Dynamic mode, enriched state (the author-scored combination of single-intent
and multi-intent cases that `numbers.py` does not recompute). Deterministic
(seed 42).

## Figures 7 and 8 (cost--precision, token efficiency)
```
python3 analysis/figures.py
```
Prints the plotted coordinates of `fig:pareto` (tokens per case vs R@1) and
`fig:token-eff` (R@1 per 1k tokens) on Dynamic, enriched. Tokens per case = the
retrieval method's own token spend (single: `trace.tokens`; multi: sum over
sub-queries). Embedding is the one pinned value (~1k per query): its traces
record the one-time corpus-embedding cost, not the per-query cost, so charging
that amortised index build to every query would misrepresent its runtime cost.
Reads `results/retrieval/runtime/dynamic/traces/`.

> `numbers.py` predates the `results/retrieval/` + `results/orchestration/`
> restructure and still reads the old `results/runtime/` layout;
> `statistics.py` and `figures.py` are the up-to-date reproduction scripts.

## Data
`results/runtime/<mode>/traces/<condition>/` and `results/design-time/traces/<method>/`.
Condition = `<method><state>[variant]` (e.g. `A1f`). Method letters.
S=Baseline, A=Embedding, B=Progressive, C=Graph, D=Agentic Tools, E=Agentic Raw, F=Agentic Hybrid.
