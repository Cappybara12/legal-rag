"""Two-stage retrieval: a binary-quantized dense prefetch narrows the corpus
down to a candidate pool, then a ColBERT MAX_SIM rescore re-ranks that pool
using token-level alignment. Both stages run in a single Qdrant query call.
"""

from dataclasses import dataclass

from fastembed import LateInteractionTextEmbedding, TextEmbedding
from qdrant_client import models

from collection import COLLECTION_NAME, get_client

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
COLBERT_MODEL = "colbert-ir/colbertv2.0"

_dense_model = None
_colbert_model = None


def _models():
    global _dense_model, _colbert_model
    if _dense_model is None:
        _dense_model = TextEmbedding(DENSE_MODEL)
        _colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)
    return _dense_model, _colbert_model


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    company: str
    text: str
    score: float


def two_stage_search(query: str, client=None, prefetch_limit: int = 100, top_k: int = 5) -> list[RetrievedChunk]:
    """Stage 1 prefetches `prefetch_limit` candidates using the binary-quantized
    dense vector. Stage 2 rescores only those candidates against the full
    ColBERT token matrix and returns the top_k by MAX_SIM score."""
    client = client or get_client()
    dense_model, colbert_model = _models()

    dense_query = next(dense_model.query_embed(query)).tolist()
    colbert_query = next(colbert_model.query_embed(query)).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=models.Prefetch(
            query=dense_query,
            using="dense",
            limit=prefetch_limit,
            params=models.SearchParams(
                quantization=models.QuantizationSearchParams(rescore=False),
            ),
        ),
        query=colbert_query,
        using="colbert",
        limit=top_k,
        with_payload=True,
    )

    return [
        RetrievedChunk(
            chunk_id=p.payload["chunk_id"],
            doc_id=p.payload["doc_id"],
            company=p.payload["company"],
            text=p.payload["text"],
            score=p.score,
        )
        for p in results.points
    ]


def dense_only_search(query: str, client=None, top_k: int = 5) -> list[RetrievedChunk]:
    """Standard single-stage dense retrieval, used as the cost/quality baseline."""
    client = client or get_client()
    dense_model, _ = _models()
    dense_query = next(dense_model.query_embed(query)).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_query,
        using="dense",
        limit=top_k,
        with_payload=True,
    )

    return [
        RetrievedChunk(
            chunk_id=p.payload["chunk_id"],
            doc_id=p.payload["doc_id"],
            company=p.payload["company"],
            text=p.payload["text"],
            score=p.score,
        )
        for p in results.points
    ]


if __name__ == "__main__":
    query = "What are the limits on indemnification for third-party IP infringement claims?"
    client = get_client()

    print("=== Two-stage (BQ prefetch + ColBERT rescore) ===")
    for r in two_stage_search(query, client, prefetch_limit=100, top_k=3):
        print(f"[{r.score:.3f}] {r.chunk_id} ({r.company})")
        print(r.text[:200].replace("\n", " ") + "...\n")

    print("=== Dense-only baseline ===")
    for r in dense_only_search(query, client, top_k=3):
        print(f"[{r.score:.3f}] {r.chunk_id} ({r.company})")
        print(r.text[:200].replace("\n", " ") + "...\n")
