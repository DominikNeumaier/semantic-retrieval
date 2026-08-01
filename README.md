# Semantic Retrieval

Process-aware orchestration architecture and retrieval comparison for LLM-based agent orchestration.

## Structure

```
semretr/        Python package
  methods/      Seven retrieval strategies (Baseline, Embedding, Progressive, Graph, Tools, Raw, Hybrid)
  runtime/      Orchestration pipeline (Intent Resolver, Planner, Mode Selection)
  eval/         Evaluation harnesses (design-time, runtime)
paper/          LaTeX source for the Semantic Retrieval conference paper
experiments/    Reproduction scripts (from experiments/paper2/)
results/        Runtime and design-time evaluation results
docs/           Documentation
```

## Dependencies

Requires `ord-bench` for the landscape and test cases.

## Migration status

Populated from MasterThesis repo. See migration plan in MasterThesis/.claude/plans/.
