#!/usr/bin/env python
"""An Ollama-compatible stub upstream.

Benchmarking a gateway against a real model measures the model, not the gateway: a
7B model on CPU takes seconds per request and buries the microseconds the gateway
spends routing, hashing and accounting. This stub answers instantly (or after a fixed
delay) so the numbers describe the proxy.

    python scripts/stub_upstream.py --port 11500 --delay-ms 0 --tokens 32
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

EMBEDDING_DIMENSIONS = 64
WORD = "token"


def build_app(delay_ms: float, tokens: int) -> FastAPI:
    app = FastAPI(title="stub-upstream", docs_url=None, openapi_url=None)
    delay = delay_ms / 1000.0

    @app.get("/api/tags")
    async def tags() -> JSONResponse:
        return JSONResponse({"models": [{"name": "stub-model"}]})

    @app.post("/api/embed")
    async def embed(request: Request) -> JSONResponse:
        payload = await request.json()
        text = str(payload.get("input", ""))
        return JSONResponse({"model": payload.get("model", "stub"), "embeddings": [_vector(text)]})

    @app.post("/api/chat")
    async def chat(request: Request) -> Any:
        payload = await request.json()
        model = str(payload.get("model", "stub-model"))
        prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in payload["messages"])
        if delay:
            await asyncio.sleep(delay)
        if payload.get("stream"):
            return StreamingResponse(
                _ndjson(model, prompt_tokens, tokens),
                media_type="application/x-ndjson",
            )
        return JSONResponse(
            {
                "model": model,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": " ".join([WORD] * tokens)},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": prompt_tokens,
                "eval_count": tokens,
            }
        )

    return app


async def _ndjson(model: str, prompt_tokens: int, tokens: int) -> AsyncIterator[bytes]:
    for _ in range(tokens):
        yield json.dumps({"message": {"content": WORD + " "}, "done": False}).encode() + b"\n"
    final = {
        "model": model,
        "message": {"content": ""},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": prompt_tokens,
        "eval_count": tokens,
    }
    yield json.dumps(final).encode() + b"\n"


def _vector(text: str) -> list[float]:
    """A deterministic pseudo-embedding.

    Identical prompts get identical vectors, different prompts get uncorrelated ones.
    That is enough to drive the cache's hit and miss paths; it says nothing about
    semantics and is not meant to.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (digest * (EMBEDDING_DIMENSIONS // len(digest) + 1))[:EMBEDDING_DIMENSIONS]
    return [(byte - 127.5) / 127.5 for byte in raw]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11500)
    parser.add_argument("--delay-ms", type=float, default=0.0, help="simulated model latency")
    parser.add_argument("--tokens", type=int, default=32, help="completion tokens per answer")
    args = parser.parse_args()

    uvicorn.run(
        build_app(args.delay_ms, args.tokens),
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
