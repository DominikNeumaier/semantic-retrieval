"""Thin, deterministic LLM + embedding clients with file-based cache.

Cache key = hash of (model, payload).
Latency is the cold-call wall time (cached calls return latency=0).
Every call returns (response, meta) where meta carries tokens + latency.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from src import config


_llm_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
_emb_client = OpenAI(base_url=config.EMBEDDING_BASE_URL, api_key=config.LLM_API_KEY)


def _refresh_token() -> bool:
    """Reload .env from disk and rebuild the OpenAI clients with the
    fresh credential. Returns True if the token actually changed."""
    global _llm_client, _emb_client
    old_key = config.LLM_API_KEY
    try:
        from dotenv import load_dotenv
        load_dotenv(config.ROOT / ".env", override=True)
        import os
        new_key = os.environ.get("LLM_API_KEY", old_key)
        if new_key != old_key:
            config.LLM_API_KEY = new_key
            _llm_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=new_key)
            _emb_client = OpenAI(base_url=config.EMBEDDING_BASE_URL, api_key=new_key)
            return True
    except Exception as e:
        print(f"  [llm] token refresh failed: {e}")
    return False


def _call_with_retry(fn, what: str):
    """Run `fn()` once; on auth error, reload .env and retry once.
    `what` is just for logging."""
    try:
        return fn()
    except Exception as e:
        msg = str(e).lower()
        if "jwt" in msg or "401" in msg or "auth" in msg or "expired" in msg:
            print(f"  [llm] auth error on {what}; refreshing token …")
            if _refresh_token():
                print(f"  [llm] retrying {what} with fresh token")
                return fn()
            raise
        raise


def get_clients():
    """Public accessor so method_tools / method_raw can use the current
    refreshable client instead of importing OpenAI directly."""
    return _llm_client, _emb_client


def _hash(payload: Any) -> str:
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()


def _cache_path(kind: str, key: str) -> Path:
    d = config.CACHE_DIR / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def chat(prompt: str, system: str = "", model: str = "") -> tuple[str, dict]:
    """Single-turn chat. Returns (text, meta). Cached by (model, prompt, system)."""
    model = model or config.LLM_MODEL
    key = _hash({"model": model, "system": system, "prompt": prompt})
    cp = _cache_path("chat", key)
    if cp.exists():
        cached = json.loads(cp.read_text())
        return cached["text"], {"tokens": cached["tokens"], "latency": 0.0, "cached": True}

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.time()
    resp = _call_with_retry(
        lambda: _llm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            seed=config.LLM_SEED,
        ),
        what=f"chat({key[:8]})",
    )
    latency = time.time() - t0
    text = resp.choices[0].message.content or ""
    tokens = (resp.usage.total_tokens if resp.usage else 0)

    cp.write_text(json.dumps({"text": text, "tokens": tokens}))
    return text, {"tokens": tokens, "latency": latency, "cached": False}


def embed(text: str) -> tuple[list[float], dict]:
    """Embed a single string. Cached by (model, text)."""
    key = _hash({"model": config.EMBEDDING_MODEL, "text": text})
    cp = _cache_path("embed", key)
    if cp.exists():
        cached = json.loads(cp.read_text())
        return cached["vec"], {"tokens": cached["tokens"], "latency": 0.0, "cached": True}

    t0 = time.time()
    # encoding_format='float' is required: the Hyperspace proxy does not
    # decode the OpenAI SDK's base64 default ('PROXY_READ_ERROR').
    resp = _call_with_retry(
        lambda: _emb_client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=text,
            encoding_format="float",
        ),
        what=f"embed({key[:8]})",
    )
    latency = time.time() - t0
    vec = resp.data[0].embedding
    tokens = resp.usage.total_tokens if resp.usage else 0

    cp.write_text(json.dumps({"vec": vec, "tokens": tokens}))
    return vec, {"tokens": tokens, "latency": latency, "cached": False}
