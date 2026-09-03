# Semantic Retrieval

**A process-aware orchestration architecture and retrieval comparison for LLM-based agent orchestration.**

This repository contains the architecture, seven retrieval strategies, committed evaluation traces, and paper analyses. It consumes ORD-Bench as an explicit one-way data dependency.

## Contributions

1. Four-layer process-aware reference architecture with Skill-Guided, Skill-Adjusted, and Dynamic modes
2. Seven retrieval strategies over a typed ORD description layer
3. Paired Clean-ORD/Enriched-ORD evaluation on 110 ORD-Bench runtime cases

## Repository boundary

- `ord-bench` owns benchmark data, case generation, validation, and benchmark analyses.
- `semantic-retrieval` owns orchestration, retrieval methods, evaluation runs, and result analyses.

The paper artefacts use the ORD-Bench data from commit:

```text
c1fadf8c242ca2779c856524aa8629e505128ee4
```

Set the benchmark checkout explicitly; a sibling `../ord-bench` checkout remains the backwards-compatible default:

```bash
export ORD_BENCH_DIR=/absolute/path/to/ord-bench
```

## Structure

```text
src/methods/                    seven retrieval strategies
src/runtime/                    Intent Resolver, Planner, Mode Selection
src/eval/                       design-time and runtime evaluation harnesses
results/retrieval/              committed retrieval traces used by the paper
results/orchestration/          committed routing/planning traces
analysis/                       read-only paper verification scripts
web/                            static results browser
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

LLM reruns require an OpenAI-compatible endpoint configured through `.env` or environment variables:

```bash
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=anthropic--claude-4.5-haiku
EMBEDDING_BASE_URL=...
EMBEDDING_MODEL=text-embedding-3-large
```

## Verify the published paper values

These commands are read-only and use the committed traces; they do not call an external model:

```bash
python3 -m unittest discover -s tests
python3 analysis/numbers.py
python3 analysis/statistics.py
python3 analysis/figures.py
python3 analysis/ambiguity_profile.py
```

`numbers.py` verifies the headline accuracy values from the runtime retrieval table, Out-of-Scope table, routing figure, and design-time appendix table. `statistics.py` reproduces bootstrap confidence intervals and McNemar p-values. `figures.py` prints the cost and token-efficiency coordinates.

## Rerun experiments

Reruns call the configured models and write results. Model-provider changes can produce different traces even with temperature 0.

```bash
python3 -m src.eval.eval_sg_routing
python3 -m src.run_dt
python3 -m src.run_rt --mode skill_adjusted
python3 -m src.run_rt --mode dynamic --force-mode dynamic
python3 -m src.eval.run_mh_benchmark
python3 -m src.run_rt --mode out_of_scope
```

The published Dynamic retrieval metric combines the 20 forced single-intent cases with per-sub-query Multi-Hint results for the 20 multi-intent cases. Skill-Adjusted retrieval is scored against the expected gap resources, separately from routing.

Reruns write to `results/design-time/` and `results/runtime/`; the committed paper traces under `results/retrieval/` and `results/orchestration/` remain unchanged. Within the rerun tree, a mode directory may be cleared unless `--no-wipe` is supplied.

## Paper

*Semantic Retrieval: A Process-Aware Architecture and Retrieval Comparison for Agent Orchestration*, Neumaier, 2026. The paper source is maintained with the thesis and is not part of this repository checkout.
