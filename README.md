# Optimizing RAG Token Costs in Legal Discovery

This repository contains the codebase for a highly optimized, two-stage legal discovery pipeline built using **Qdrant** and **FastEmbed**.

It implements:
1. **Stage 1 (Prefetch):** Fast candidate retrieval using compressed 1-bit **Binary Quantized** dense vectors.
2. **Stage 2 (Rescore):** Precise token-level late interaction using **ColBERT MAX_SIM** comparison natively in Qdrant.
3. **Local Sentence Isolation:** Extracts and forwards only the specific sentence(s) driving the match to the LLM context, reducing token costs by **67%**.

## Directory Structure

* `src/`:
  * `collection.py`: Setup script for the Qdrant collection schemas and index configs.
  * `ingest.py`: Script to generate dual-embeddings and upsert documents.
  * `query.py`: Script to execute the unified two-stage query using Qdrant's universal Query API.
  * `isolate.py`: Logic to isolate matching sentences locally using ColBERT MAX_SIM.
  * `benchmark.py`: Comparative audit comparing token counts and latencies.
  * `sweep.py`: Prefetch limit sweep illustrating speed/accuracy recovery trade-offs.
  * `footprint_and_cost.py`: Calculates RAM savings (32x) and dollar-cost projections.
* `data/`:
  * `generate_documents.py`: Script to synthesize mock litigation files and contract clauses.

## Setup & Execution

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Qdrant locally (required before ingestion):
   ```bash
   docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
   ```
   If you prefer to run it in the background, add `-d` and a container name, for example:
   ```bash
   docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
   ```
4. Generate the synthetic legal documents and chunk data:
   ```bash
   python3 data/generate_documents.py
   ```
   This creates the chunk file used by the ingest step.
5. Ingest documents:
   ```bash
   python3 src/ingest.py
   ```
6. Run benchmarks:
   ```bash
   python3 src/benchmark.py
   ```

## Screenshots

A few relevant screenshots would make the workflow much easier to follow. I suggest adding 2–3 images in these spots:

- Qdrant running locally after Docker startup
- The generated document/chunk output after running the data generation script
- The benchmark or query output showing the two-stage retrieval flow

When you share the images, place them in the `screenshots/` folder and update the placeholders below:

- ![Qdrant running](screenshots/qdrant-running.png)
- ![Generated data](screenshots/generated-data.png)
- ![Benchmark output](screenshots/benchmark-output.png)
