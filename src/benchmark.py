"""Compares standard dense RAG against the two-stage BQ + ColBERT pipeline
on token payload size and query latency, using tiktoken for token counts
since that's what actually gets billed downstream.
"""

import time

import tiktoken
from fastembed import LateInteractionTextEmbedding

from collection import get_client
from isolate import isolate_sentences
from query import dense_only_search, two_stage_search

ENCODER = tiktoken.get_encoding("cl100k_base")

QUERIES = [
    "What are the limits on indemnification for third-party IP infringement claims?",
    "Who owns the intellectual property in the deliverables created under the agreement?",
    "What is the cap on liability for damages under this agreement?",
    "Under what circumstances can the service provider terminate the agreement?",
    "What happens to confidentiality obligations after the agreement ends?",
]


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def run_benchmark(top_k: int = 3, prefetch_limit: int = 100):
    client = get_client()
    colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

    dense_tokens, two_stage_tokens, isolated_tokens = [], [], []
    dense_latencies, two_stage_latencies = [], []

    # Warm up models and connections to avoid counting startup latency in the benchmark
    dense_only_search("warmup", client)
    two_stage_search("warmup", client)

    for query in QUERIES:
        t0 = time.perf_counter()
        dense_results = dense_only_search(query, client, top_k=top_k)
        dense_latencies.append(time.perf_counter() - t0)
        dense_tokens.append(sum(count_tokens(r.text) for r in dense_results))

        t0 = time.perf_counter()
        two_stage_results = two_stage_search(query, client, prefetch_limit=prefetch_limit, top_k=top_k)
        two_stage_latencies.append(time.perf_counter() - t0)
        two_stage_tokens.append(sum(count_tokens(r.text) for r in two_stage_results))

        query_vecs = next(colbert_model.query_embed(query))
        isolated_text = []
        for r in two_stage_results:
            top_sentences = isolate_sentences(r.text, query_vecs, colbert_model, top_n=1)
            isolated_text.extend(s for s, _ in top_sentences)
        isolated_tokens.append(count_tokens(" ".join(isolated_text)))

    def avg(values):
        return sum(values) / len(values)

    print(f"Queries: {len(QUERIES)} | top_k={top_k} | prefetch_limit={prefetch_limit}\n")
    print(f"{'Pipeline':<35}{'Avg tokens to LLM':<22}{'Avg latency (ms)':<18}")
    print(f"{'Standard dense RAG':<35}{avg(dense_tokens):<22.0f}{avg(dense_latencies) * 1000:<18.1f}")
    print(f"{'Two-stage (BQ + ColBERT)':<35}{avg(two_stage_tokens):<22.0f}{avg(two_stage_latencies) * 1000:<18.1f}")
    print(f"{'Two-stage + sentence isolation':<35}{avg(isolated_tokens):<22.0f}{'n/a (local step)':<18}")

    reduction = (1 - avg(isolated_tokens) / avg(dense_tokens)) * 100
    print(f"\nToken payload reduction vs. standard dense RAG: {reduction:.1f}%")


if __name__ == "__main__":
    run_benchmark()
