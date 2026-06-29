"""Creates the Qdrant collection with a dense vector (for the BQ prefetch)
and a ColBERT multivector (for the MAX_SIM rescore) living side by side."""

from qdrant_client import QdrantClient, models

COLLECTION_NAME = "legal_discovery"
DENSE_DIM = 384  # BAAI/bge-small-en-v1.5
COLBERT_DIM = 128  # colbert-ir/colbertv2.0


def get_client(url: str = "http://localhost:6333") -> QdrantClient:
    return QdrantClient(url=url, timeout=60)


def create_collection(client: QdrantClient, recreate: bool = False):
    if recreate and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True),
                ),
            ),
            "colbert": models.VectorParams(
                size=COLBERT_DIM,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                on_disk=True,
                hnsw_config=models.HnswConfigDiff(m=0),  # rescore-only, no HNSW index needed
            ),
        },
    )


if __name__ == "__main__":
    client = get_client()
    create_collection(client, recreate=True)
    print(f"Collection '{COLLECTION_NAME}' ready")
