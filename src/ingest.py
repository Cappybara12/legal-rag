"""Embeds every chunk with both the dense model and ColBERT, then upserts
into Qdrant. One model call per vector type, batched across all chunks."""

import json
from pathlib import Path

from fastembed import LateInteractionTextEmbedding, TextEmbedding
from qdrant_client import models

from collection import COLLECTION_NAME, create_collection, get_client

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
COLBERT_MODEL = "colbert-ir/colbertv2.0"


def load_chunks() -> list[dict]:
    path = Path(__file__).parent.parent / "data" / "chunks.json"
    return json.loads(path.read_text())


def main():
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")

    dense_model = TextEmbedding(DENSE_MODEL)
    colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)

    dense_vectors = list(dense_model.embed(texts))
    colbert_vectors = list(colbert_model.embed(texts))

    client = get_client()
    create_collection(client, recreate=True)

    points = [
        models.PointStruct(
            id=i,
            vector={
                "dense": dense_vectors[i].tolist(),
                "colbert": colbert_vectors[i].tolist(),
            },
            payload={
                "doc_id": chunks[i]["doc_id"],
                "company": chunks[i]["company"],
                "chunk_id": chunks[i]["chunk_id"],
                "text": chunks[i]["text"],
            },
        )
        for i in range(len(chunks))
    ]

    batch_size = 16
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i:i + batch_size], wait=True)
    print(f"Upserted {len(points)} points into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
