"""Sentence-level context isolation.

Qdrant's MAX_SIM rescore gives you a single aggregate score per chunk, not a
breakdown of which tokens drove it. To shrink a winning chunk down to the
sentence(s) that actually matched the query, we re-run MAX_SIM locally,
sentence by sentence, against the same ColBERT query vector.
"""

import re

import numpy as np

SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())  # collapse newlines/whitespace from the chunk
    sentences = SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def max_sim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """ColBERT's late-interaction score: for each query token, take its best
    match among the document's tokens, then sum those best matches."""
    sims = query_vecs @ doc_vecs.T  # (num_query_tokens, num_doc_tokens)
    return float(sims.max(axis=1).sum())


def isolate_sentences(chunk_text: str, query_vecs: np.ndarray, colbert_model, top_n: int = 2) -> list[tuple[str, float]]:
    sentences = split_sentences(chunk_text)
    if not sentences:
        return [(chunk_text, 0.0)]

    sentence_vecs = list(colbert_model.embed(sentences))
    scored = [(sentences[i], max_sim(query_vecs, sentence_vecs[i])) for i in range(len(sentences))]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]


def build_optimized_prompt(query: str, chunk_texts: list[str], colbert_model, top_n_sentences: int = 1) -> str:
    """Takes the query and retrieved chunk texts, extracts the top_n_sentences per chunk,
    and formats them into an optimized prompt payload."""
    query_vecs = next(colbert_model.query_embed(query))
    
    context_parts = []
    for i, text in enumerate(chunk_texts):
        top_sentences = isolate_sentences(text, query_vecs, colbert_model, top_n=top_n_sentences)
        isolated_text = " ".join(s for s, _ in top_sentences)
        context_parts.append(f"[Source Chunk {i+1}]: {isolated_text}")
        
    context_str = "\n\n".join(context_parts)
    
    prompt = (
        "Answer the user's question based strictly on the provided context below. "
        "If the context does not contain the answer, state that clearly.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    return prompt


def build_standard_prompt(query: str, chunk_texts: list[str]) -> str:
    """Takes the query and retrieved chunk texts, and formats the entire chunk texts
    into a standard prompt payload."""
    context_parts = []
    for i, text in enumerate(chunk_texts):
        context_parts.append(f"[Source Chunk {i+1}]: {text}")
        
    context_str = "\n\n".join(context_parts)
    
    prompt = (
        "Answer the user's question based strictly on the provided context below. "
        "If the context does not contain the answer, state that clearly.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    return prompt


if __name__ == "__main__":
    from fastembed import LateInteractionTextEmbedding
    import tiktoken

    colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    query = "What are the limits on indemnification for third-party IP infringement claims?"
    query_vecs = next(colbert_model.query_embed(query))

    chunk = """Each party (the "Indemnifying Party") shall indemnify, defend, and hold harmless the
    other party from third-party claims arising out of the Indemnifying Party's gross negligence
    or willful misconduct in the performance of this Agreement. Notwithstanding the foregoing,
    Service Provider's indemnification obligations with respect to third-party intellectual
    property infringement claims shall be capped at an amount equal to the fees paid by Client
    under the applicable Statement of Work during the twelve (12) months preceding the claim,
    and shall exclude any claim arising from open-source components disclosed to Client prior to
    delivery."""

    print("=== Sentence-Level Scoring ===")
    for sentence, score in isolate_sentences(chunk, query_vecs, colbert_model, top_n=2):
        print(f"[{score:.2f}] {sentence}\n")

    print("=== Prompt Payload Generation ===")
    encoder = tiktoken.get_encoding("cl100k_base")
    
    std_prompt = build_standard_prompt(query, [chunk])
    opt_prompt = build_optimized_prompt(query, [chunk], colbert_model, top_n_sentences=1)
    
    std_tokens = len(encoder.encode(std_prompt))
    opt_tokens = len(encoder.encode(opt_prompt))
    
    print("--- Standard Prompt Payload ---")
    print(std_prompt)
    print(f"Token Count: {std_tokens}\n")
    
    print("--- Optimized Prompt Payload (Sentence Isolated) ---")
    print(opt_prompt)
    print(f"Token Count: {opt_tokens}\n")
    
    reduction = (1 - opt_tokens / std_tokens) * 100
    print(f"Token reduction for single chunk: {reduction:.1f}%")

