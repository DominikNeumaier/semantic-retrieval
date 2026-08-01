# Semantic Retrieval

**A process-aware orchestration architecture and retrieval comparison for LLM-based agent orchestration.**

LLM-based agents operating over enterprise landscapes must select the right resource among many semantically similar alternatives. Existing approaches wrap resources in free-text descriptions and rely on the LLM to reason over them — providing limited grounding for disambiguation. This work asks whether a typed description layer and semantic enrichment improve selection, and which retrieval mechanisms benefit.

## Contributions

1. **Process-aware reference architecture** — a four-layer design (Resource, Description, Semantic, Orchestration) with three orchestration modes (Skill-Guided, Skill-Adjusted, Dynamic) spanning the full process-structuredness spectrum.
2. **Seven retrieval strategies** over a typed ORD layer — covering embedding-based, LLM-guided, graph-based, and agentic retrieval families, each with a fixed prompt and no tuning so differences are attributable to the description layer.
3. **Systematic comparison with and without semantic enrichment** — evaluated on 110 runtime cases from ORD-Bench across both ORD states, with bootstrap confidence intervals and McNemar significance tests.

**Key finding:** Semantic enrichment is conditional — it improves selection by up to +15 pp for methods that consume typed fields (Agentic Tools, Graph), yet leaves free-text methods flat or degrades them. Enrichment and retrieval are one coupled design decision.

## Structure

```
src/
  methods/    Seven retrieval strategies (Baseline, Embedding, Progressive, Graph, Tools, Raw, Hybrid)
  runtime/    Orchestration pipeline (Intent Resolver, Planner, Mode Selection)
  eval/       Evaluation harnesses (design-time, runtime)
results/      Runtime and design-time evaluation traces and summaries
experiments/  Reproduction scripts (numbers.py, statistics.py)
paper/        Conference paper LaTeX source and PDF
docs/         Design documentation
web/          Interactive results browser (traces, R@1/R@5, enrichment delta, cost)
```

## Reproducing the numbers

```bash
python experiments/numbers.py     # all paper tables and figures
python experiments/statistics.py  # bootstrap CIs and McNemar p-values
```

## Dependencies

Requires the ORD-Bench landscape and test cases (`ord-bench` repo) for the evaluation data.

## Paper

*Semantic Retrieval: A Process-Aware Architecture and Retrieval Comparison for Agent Orchestration*, Neumaier, 2026.
