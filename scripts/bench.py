#!/usr/bin/env python
"""Load generator for the gateway.

Sends N chat completions at a fixed concurrency and reports the latency distribution,
throughput and the cache outcome mix reported by the ``X-Gateway-Cache`` header.

Point it at a gateway whose upstream is ``scripts/stub_upstream.py`` to measure the
gateway itself; point it at a gateway in front of a real model to measure the model.

    python scripts/bench.py --requests 500 --concurrency 20 --unique 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import Counter
from dataclasses import dataclass

import httpx


@dataclass(slots=True)
class Sample:
    latency: float
    status: int
    cache: str


async def _one(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    bypass: bool,
) -> Sample:
    headers = {"Authorization": f"Bearer {api_key}"}
    if bypass:
        headers["X-Gateway-Cache-Bypass"] = "true"
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{url.rstrip('/')}/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            headers=headers,
        )
    except httpx.HTTPError as exc:
        print(f"request failed: {exc}")
        return Sample(time.perf_counter() - started, 0, "error")
    return Sample(
        latency=time.perf_counter() - started,
        status=response.status_code,
        cache=response.headers.get("X-Gateway-Cache", "-"),
    )


async def run(args: argparse.Namespace) -> list[Sample]:
    unique_prompts = max(1, int(args.requests * args.unique))
    prompts = [f"benchmark prompt number {index}" for index in range(unique_prompts)]
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency * 2)

    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        async def worker(index: int) -> Sample:
            async with semaphore:
                return await _one(
                    client,
                    args.url,
                    args.api_key,
                    args.model,
                    prompts[index % unique_prompts],
                    args.no_cache,
                )

        # Warm-up requests are excluded from the report: the first call per prompt pays
        # for a cold connection pool and an empty cache index.
        if args.warmup:
            await asyncio.gather(*(worker(i) for i in range(min(args.warmup, args.requests))))
        return list(await asyncio.gather(*(worker(i) for i in range(args.requests))))


def report(samples: list[Sample], elapsed: float, concurrency: int) -> None:
    latencies = sorted(sample.latency for sample in samples)
    statuses = Counter(sample.status for sample in samples)
    caches = Counter(sample.cache for sample in samples)
    successful = [s for s in samples if s.status == 200]

    def percentile(fraction: float) -> float:
        if not latencies:
            return 0.0
        index = min(len(latencies) - 1, round(fraction * (len(latencies) - 1)))
        return latencies[index]

    print()
    print(f"requests      {len(samples)} at concurrency {concurrency}")
    print(f"wall clock    {elapsed:.2f} s")
    print(f"throughput    {len(samples) / elapsed:.1f} req/s")
    print(f"success       {len(successful)} / {len(samples)}")
    print(f"statuses      {dict(sorted(statuses.items()))}")
    print(f"cache         {dict(sorted(caches.items()))}")
    if latencies:
        print(f"mean          {statistics.fmean(latencies) * 1000:.2f} ms")
        print(f"p50           {percentile(0.50) * 1000:.2f} ms")
        print(f"p95           {percentile(0.95) * 1000:.2f} ms")
        print(f"p99           {percentile(0.99) * 1000:.2f} ms")
        print(f"max           {latencies[-1] * 1000:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--api-key", default="sk-local-dev")
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--unique",
        type=float,
        default=1.0,
        help="fraction of distinct prompts; 0.1 means every prompt repeats ten times",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--no-cache", action="store_true", help="send X-Gateway-Cache-Bypass")
    args = parser.parse_args()

    started = time.perf_counter()
    samples = asyncio.run(run(args))
    report(samples, time.perf_counter() - started, args.concurrency)


if __name__ == "__main__":
    main()
