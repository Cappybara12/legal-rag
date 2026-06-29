"""Two things the brief's table claims but never measures:
  1. The memory footprint difference between full-precision dense vectors
     and their binary-quantized form.
  2. What the token reduction is actually worth in dollars, using a real
     published LLM input-token price.

Note on (1): this is reported as theoretical math, not measured from the
live collection. Qdrant pre-allocates vector storage in fixed-size mmap
chunks (32MB blocks in this version), so on a small test collection like
ours (24 points) on-disk file sizes are dominated by that fixed overhead
and don't reflect real per-vector compression. The ratio only becomes
visible empirically at much larger scale (thousands+ of vectors).
"""

DENSE_DIM = 384  # BAAI/bge-small-en-v1.5

# Claude Sonnet 4.6 input pricing as of 2026-06: $3 / million input tokens.
# Swap this for whatever model the production system actually calls.
PRICE_PER_MILLION_INPUT_TOKENS = 3.00

STANDARD_TOKENS_PER_QUERY = 604
OPTIMIZED_TOKENS_PER_QUERY = 199


def theoretical_vector_size() -> dict:
    full_precision_bytes = DENSE_DIM * 4  # float32
    binary_quantized_bytes = DENSE_DIM / 8  # 1 bit per dimension
    return {
        "full_precision_bytes": full_precision_bytes,
        "binary_quantized_bytes": binary_quantized_bytes,
        "compression_ratio": full_precision_bytes / binary_quantized_bytes,
    }


def cost_per_n_queries(n_queries: int = 1000) -> dict:
    standard_cost = (STANDARD_TOKENS_PER_QUERY * n_queries / 1_000_000) * PRICE_PER_MILLION_INPUT_TOKENS
    optimized_cost = (OPTIMIZED_TOKENS_PER_QUERY * n_queries / 1_000_000) * PRICE_PER_MILLION_INPUT_TOKENS
    return {
        "standard_usd": standard_cost,
        "optimized_usd": optimized_cost,
        "savings_usd": standard_cost - optimized_cost,
        "savings_pct": (1 - optimized_cost / standard_cost) * 100,
    }


if __name__ == "__main__":
    print("=== Memory footprint per dense vector (theoretical, not measured) ===")
    sizes = theoretical_vector_size()
    print(f"Full precision (384-dim float32): {sizes['full_precision_bytes']:.0f} bytes")
    print(f"Binary quantized (1 bit/dim):      {sizes['binary_quantized_bytes']:.0f} bytes")
    print(f"Compression ratio:                 {sizes['compression_ratio']:.0f}x smaller")
    print("(Real measurement needs thousands+ of vectors -- at our 24-doc test")
    print(" scale, Qdrant's fixed-size storage chunks mask the actual difference.)\n")

    print(f"=== Cost projection at ${PRICE_PER_MILLION_INPUT_TOKENS:.2f}/million input tokens ===")
    for n in (1_000, 100_000):
        c = cost_per_n_queries(n)
        print(f"\nPer {n:,} queries:")
        print(f"  Standard dense RAG:        ${c['standard_usd']:.2f}")
        print(f"  Two-stage + isolation:     ${c['optimized_usd']:.2f}")
        print(f"  Savings:                   ${c['savings_usd']:.2f} ({c['savings_pct']:.1f}%)")
