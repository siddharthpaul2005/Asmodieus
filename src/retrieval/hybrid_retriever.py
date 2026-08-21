"""
src/retrieval/hybrid_retriever.py
Step 5b — Hybrid dense + sparse retrieval with Reciprocal Rank Fusion via Pinecone Integrated Search.

- Uses Pinecone Integrated Search (multilingual-e5-large) for dense retrieval.
- Uses bm25s in-process for sparse lexical retrieval.
- Fuses candidates using Reciprocal Rank Fusion (RRF).
- Preserves score thresholds and extractive metadata contract.
"""
import json
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.retrieval.bm25_index import load_bm25, tokenize

try:
    from src.guardrails.lang_detect import detect_language
except ImportError:
    detect_language = None

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
META_PATH = str(PROJECT_ROOT / "data" / "processed" / "chunk_meta.jsonl")
CORPUS_PATH = str(PROJECT_ROOT / "data" / "processed" / "corpus.jsonl")

# Dense score floor for fusion filtering
DENSE_FLOOR_FOR_FUSION = 0.70

_pinecone_index = None
_chunk_meta_map = None
_bm25 = None
_bm25_chunk_ids = None


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "asmodieus")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set.")
        pc = Pinecone(api_key=api_key)
        _pinecone_index = pc.Index(index_name)
    return _pinecone_index


def _load_all():
    global _chunk_meta_map, _bm25, _bm25_chunk_ids
    if _chunk_meta_map is None:
        # We no longer read corpus.jsonl here.
        # Pinecone dense_search automatically populates _chunk_meta_map with hit.fields!
        _chunk_meta_map = {}
        _bm25, _bm25_chunk_ids = load_bm25()


def dense_search(query: str, k: int = 20):
    """Searches Pinecone using integrated embedding."""
    _load_all()
    index = _get_pinecone_index()
    t0 = time.perf_counter()
    res = index.search(
        namespace=os.getenv("PINECONE_NAMESPACE", "default"),
        top_k=k,
        inputs={"text": query}
    )
    t1 = time.perf_counter()
    if os.getenv("VERBOSE", "0") == "1":
        print(f"    pinecone_search: {(t1 - t0) * 1000:.2f}ms")

    hits = getattr(res.result, "hits", []) if hasattr(res, "result") else getattr(res, "hits", [])
    ranked_ids = []
    scores = {}

    for hit in hits:
        cid = str(hit.id)
        score = float(getattr(hit, "score", 0.0))
        ranked_ids.append(cid)
        scores[cid] = score

        # If not present in local metadata map, populate from Pinecone fields
        if cid not in _chunk_meta_map:
            fields = getattr(hit, "fields", {}) or {}
            _chunk_meta_map[cid] = {
                "idx": cid,
                "chunk_id": cid,
                "source_chunk_id": cid,
                "text": fields.get("text", ""),
                "context_text": fields.get("text", ""),
                "strategy": fields.get("strategy", "passage"),
                "language": fields.get("language", "eng"),
            }

    return ranked_ids, scores


def sparse_search(query: str, k: int = 20) -> list[str]:
    _load_all()
    tokens = tokenize(query)
    results, scores = _bm25.retrieve(bm25s_tokenize(tokens), k=k) if hasattr(_bm25, "retrieve") else ([], [])
    top_indices = results[0] if len(results) > 0 else []
    return [str(_bm25_chunk_ids[i]) for i in top_indices if i < len(_bm25_chunk_ids)]


def bm25s_tokenize(tokens: list[str]):
    import bm25s
    return bm25s.tokenize([" ".join(tokens)], show_progress=False)


def rrf_fuse(ranked_lists: list[list[str]], k_const: int = 60, top_k: int = 5) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_const + rank)
    fused = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
    return fused[:top_k]


import functools

@functools.lru_cache(maxsize=100)
def retrieve(
    query: str,
    k: int = 5,
    candidate_k: int = 20,
    verbose: bool = False,
    lang: str | None = None,
) -> list[dict]:
    _load_all()

    if lang is None and detect_language is not None:
        try:
            lang = detect_language(query)
        except Exception:
            lang = None

    t0 = time.perf_counter()
    dense_ranked, dense_scores = dense_search(query, k=candidate_k)
    t1 = time.perf_counter()

    sparse_ranked = sparse_search(query, k=candidate_k)
    t2 = time.perf_counter()

    fused_idxs = rrf_fuse([dense_ranked, sparse_ranked], top_k=candidate_k)
    t3 = time.perf_counter()

    if verbose:
        print(f"  dense_search:  {(t1 - t0) * 1000:.2f}ms")
        print(f"  sparse_search: {(t2 - t1) * 1000:.2f}ms")
        print(f"  rrf_fuse:      {(t3 - t2) * 1000:.2f}ms")

    strong = [idx for idx in fused_idxs if dense_scores.get(idx, 0.0) >= DENSE_FLOOR_FOR_FUSION]
    ordered = strong if len(strong) >= k else fused_idxs

    results = []
    for idx in ordered:
        meta = dict(_chunk_meta_map.get(idx, {
            "chunk_id": idx,
            "text": "",
            "language": "eng"
        }))
        meta["_dense_score"] = dense_scores.get(idx, 0.0)
        results.append(meta)

    if lang:
        same_lang = [r for r in results if r.get("language") == lang]
        if len(same_lang) >= k:
            return same_lang[:k]
        backfill = [r for r in results if r.get("language") == "eng" and r not in same_lang]
        combined = same_lang + backfill
        if combined:
            return combined[:k]
        return results[:k]

    return results[:k]


def main():
    print("Warming up Pinecone + BM25...")
    _load_all()
    print("Warm.")

    test_query = "What did the QPR manager say about transfers?"

    print("\n--- dense only ---")
    ranked, scores = dense_search(test_query, k=5)
    for idx in ranked:
        r = _chunk_meta_map.get(idx, {})
        print(f"[{scores.get(idx, 0):.3f}] {r.get('text', '')[:120]}")

    print("\n--- fused (timed, per-stage) ---")
    t0 = time.perf_counter()
    results = retrieve(test_query, k=5, candidate_k=20, verbose=True)
    t1 = time.perf_counter()
    print(f"Total: {(t1 - t0) * 1000:.2f}ms")
    for r in results:
        print(f"[{r.get('strategy', 'passage')}] (dense={r['_dense_score']:.3f}) {r.get('text', '')[:120]}")


if __name__ == "__main__":
    main()