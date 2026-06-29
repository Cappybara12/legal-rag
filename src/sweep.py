"""Sweeps the prefetch limit (how many candidates Stage 1 hands to Stage 2)
to show the speed/accuracy tradeoff the brief asks for in section 5.
"""

import time

from collection import get_client
from query import dense_only_search, two_stage_search

PREFETCH_LIMITS = [10, 25, 50, 100, 250, 500]

QUERIES = [
    "What are the limits on indemnification for third-party IP infringement claims?",
    "Who owns the intellectual property in the deliverables created under the agreement?",
    "What is the cap on liability for damages under this agreement?",
    "Under what circumstances can the service provider terminate the agreement?",
    "What happens to confidentiality obligations after the agreement ends?",
]


def run_sweep(top_k: int = 3):
    client = get_client()

    # Warm up models and connections to avoid counting startup latency in the sweep
    dense_only_search("warmup", client)
    two_stage_search("warmup", client)

    baseline_latency = []
    for query in QUERIES:
        t0 = time.perf_counter()
        dense_only_search(query, client, top_k=top_k)
        baseline_latency.append(time.perf_counter() - t0)
    baseline_avg = sum(baseline_latency) / len(baseline_latency) * 1000

    print(f"Standard dense-only search (no rescore): {baseline_avg:.1f} ms avg\n")
    print(f"{'prefetch_limit':<16}{'avg latency (ms)':<20}{'Top-1 Match Rate':<22}")

    # Compute reference results at the highest prefetch limit (which represents the accuracy ceiling)
    reference_limit = PREFETCH_LIMITS[-1]
    reference_top_ids = []
    for query in QUERIES:
        results = two_stage_search(query, client, prefetch_limit=reference_limit, top_k=top_k)
        reference_top_ids.append(results[0].chunk_id if results else None)

    for limit in PREFETCH_LIMITS:
        latencies = []
        top_ids = []
        for query in QUERIES:
            t0 = time.perf_counter()
            results = two_stage_search(query, client, prefetch_limit=limit, top_k=top_k)
            latencies.append(time.perf_counter() - t0)
            top_ids.append(results[0].chunk_id if results else None)

        avg_latency = sum(latencies) / len(latencies) * 1000
        
        matches = sum(1 for a, b in zip(top_ids, reference_top_ids) if a == b)
        match_pct = (matches / len(QUERIES)) * 100
        if limit == reference_limit:
            match_status = "100% (Reference)"
        else:
            match_status = f"{match_pct:.0f}% ({matches}/{len(QUERIES)})"

        print(f"{limit:<16}{avg_latency:<20.1f}{match_status:<22}")


if __name__ == "__main__":
    run_sweep()

