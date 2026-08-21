"""
src/answering/extractive.py
Step 6 (Option A) — Zero-model extractive answering. Returns the
top RRF-fused chunk directly, templated. No inference, no model load
— latency is bounded by retrieve() alone (~35ms warm, confirmed).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.retrieval.hybrid_retriever import retrieve

MIN_RRF_CHUNKS = 1  # if retrieve() returns nothing, refuse instead of crashing

def answer(query: str, k: int = 5, candidate_k: int = 20) -> dict:
    """
    Returns:
    {
        "answer": str,
        "cited_chunk_ids": [str],
        "confidence": float,   # placeholder proxy, see note below
        "refused": bool
    }
    """
    chunks = retrieve(query, k=k, candidate_k=candidate_k)

    if not chunks or len(chunks) < MIN_RRF_CHUNKS:
        return {
            "answer": "I don't have enough information in the retrieved context to answer that confidently.",
            "cited_chunk_ids": [],
            "confidence": 0.0,
            "refused": True,
        }

    top = chunks[0]
    context = top.get("context_text") or top["text"]

    return {
        "answer": f"Based on the passage: {context}",
        "cited_chunk_ids": [c["chunk_id"] for c in chunks],
        "confidence": 1.0,  # no real scoring model yet — RRF gives rank not calibrated confidence
        "refused": False,
    }

if __name__ == "__main__":
    import time
    q = "What did the QPR manager say about transfers?"
    t0 = time.perf_counter()
    result = answer(q)
    t1 = time.perf_counter()
    print(f"Answered in {(t1-t0)*1000:.2f}ms")
    print(f"Refused: {result['refused']}")
    print(f"Answer: {result['answer'][:200]}")
    print(f"Cited: {result['cited_chunk_ids']}")