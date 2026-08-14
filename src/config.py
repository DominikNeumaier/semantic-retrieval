"""Central configuration. Reads .env from the project root."""

import os
from pathlib import Path

# config.py is at src/config.py -> project root is one level up
ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# LLM
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:6655/litellm/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "no-key")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic--claude-4.5-haiku")

# Embeddings
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "http://localhost:6655/litellm/v1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")

# Determinism
LLM_TEMPERATURE = 0.0
LLM_SEED = 42
TOP_K = 5

# Paths — landscape data lives in the sibling ord-bench repo
BENCHMARK_DIR        = ROOT.parent / "ord-bench" / "data"
LANDSCAPE_DIR        = BENCHMARK_DIR / "landscape" / "systems"
LANDSCAPE_ENRICHED_DIR = BENCHMARK_DIR / "landscape" / "systems_enriched"
DT_OUTPUT_DIR        = BENCHMARK_DIR / "test_cases" / "design_time" / "output"
RT_OUTPUT_DIR        = BENCHMARK_DIR / "test_cases" / "runtime" / "output"
CACHE_DIR            = ROOT / "cache"
RESULTS_DT           = ROOT / "results" / "design-time"
RESULTS_RT           = ROOT / "results" / "runtime"

# ORD state for DT runs: "clean" loads ord.json, "enriched" loads ord_enriched.json
ORD_STATE = os.environ.get("ORD_STATE", "enriched")
