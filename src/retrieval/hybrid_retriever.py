"""
src/retrieval/hybrid_retriever.py
Step 5b — Hybrid dense + sparse retrieval with Reciprocal Rank Fusion.

Fixes from review:
- Removed a dead/duplicate `retrieve()` that shadowed the real one and
  referenced undefined names (_existing_fusion_logic, chunk_meta). Because
  it was defined first and Python lets later defs silently overwrite
  earlier ones, that "language-aware" version NEVER RAN — language
  filtering was doing nothing at all, for any query.
- Language filtering now actually happens in the one real `retrieve()`,
  using same-language chunks first and backfilling with English if there
  aren't enough.
- Dense cosine score is now captured alongside RRF fusion and attached to
  each returned chunk dict as `_dense_score`, so extractive.py can gate
  and report confidence on the chunk it's ACTUALLY returning, instead of
  re-fetching a possibly-different chunk's score separately.
- Fused candidates with near-zero dense support (i.e. pure BM25 lexical
  flukes with no real semantic backing) are filtered out before being
  returned, unless doing so would leave fewer than k results.
"""
import json
import os
import sys
import time

import hnswlib
import bm25s

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.embedding.embedder import embed_query
from src.retrieval.bm25_index import load_bm25, tokenize

try:
    from src.guardrails.lang_detect import detect_language
except ImportError:
    detect_language = None

INDEX_PATH = "data/processed/index.bin"
META_PATH = "data/processed/chunk_meta.jsonl"
DIM = 384

# Below this dense cosine score, a fused-in candidate is treated as a
# lexical-only fluke (BM25 loved it, dense found nothing there).
DENSE_FLOOR_FOR_FUSION = 0.75

_hnsw_index = None
_chunk_meta = None
_bm25 = None
_bm25_chunk_ids = None


def _load_all():
    global _hnsw_index, _chunk_meta, _bm25, _bm25_chunk_ids
    if _hnsw_index is None:
        _chunk_meta = []
        with open(META_PATH, "r", encoding="utf-8") as f:
            for line in f:
                _chunk_meta.append(json.loads(line))

        _hnsw_index = hnswlib.Index(space="cosine", dim=DIM)
        _hnsw_index.load_index(INDEX_PATH, max_elements=len(_chunk_meta))
        _hnsw_index.set_ef(50)

        _bm25, _bm25_chunk_ids = load_bm25()


def dense_search(query: str, k: int = 20):
    """Returns (ranked_idx_list, {idx: cosine_score}). Score dict lets
    callers judge candidate quality, not just rank position."""
    _load_all()
    te0 = time.perf_counter()
    qvec = embed_query(query).reshape(1, -1)
    te1 = time.perf_counter()
    labels, distances = _hnsw_index.knn_query(qvec, k=k)
    te2 = time.perf_counter()
    print(f"    embed_query: {(te1 - te0) * 1000:.2f}ms | hnsw_knn: {(te2 - te1) * 1000:.2f}ms")

    ranked = labels[0].tolist()
    # hnswlib cosine space returns distance = 1 - cosine_sim
    scores = {idx: 1.0 - float(dist) for idx, dist in zip(ranked, distances[0].tolist())}
    return ranked, scores


def sparse_search(query: str, k: int = 20) -> list[int]:
    _load_all()
    tokens = tokenize(query)
    results, scores = _bm25.retrieve(bm25s.tokenize([" ".join(tokens)], show_progress=False), k=k)
    top_indices = results[0]
    return [_bm25_chunk_ids[i] for i in top_indices]


def rrf_fuse(ranked_lists: list[list[int]], k_const: int = 60, top_k: int = 5) -> list[int]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_const + rank)
    fused = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
    return fused[:top_k]


def retrieve(
    query: str,
    k: int = 5,
    candidate_k: int = 20,
    verbose: bool = False,
    lang: str | None = None,
) -> list[dict]:
    """
    lang: language code of the query. Pass this in from your STT pipeline
    if it already knows the language (more reliable than re-detecting from
    text here). If given, same-language chunks are preferred, backfilled
    with English if there aren't enough same-language results.
    """
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

    # over-fetch at fusion time (candidate_k, not k) so language filtering
    # downstream has enough candidates to work with
    fused_idxs = rrf_fuse([dense_ranked, sparse_ranked], top_k=candidate_k)
    t3 = time.perf_counter()

    if verbose:
        print(f"  dense_search:  {(t1 - t0) * 1000:.2f}ms")
        print(f"  sparse_search: {(t2 - t1) * 1000:.2f}ms")
        print(f"  rrf_fuse:      {(t3 - t2) * 1000:.2f}ms")

    # drop fused candidates with near-zero dense support (pure lexical
    # flukes) — but never let this filtering leave us with fewer than k
    strong = [idx for idx in fused_idxs if dense_scores.get(idx, 0.0) >= DENSE_FLOOR_FOR_FUSION]
    ordered = strong if len(strong) >= k else fused_idxs

    results = []
    for idx in ordered:
        meta = dict(_chunk_meta[idx])
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
        # nothing in target language or English survived filtering —
        # fall back to unfiltered ranked results rather than returning nothing
        return results[:k]

    return results[:k]


def _print_result(idx: int):
    r = _chunk_meta[idx]
    print(f"[{r['strategy']}] {r['text'][:120]}")


def main():
    print("Warming up (loading model + indexes)...")
    from src.embedding.embedder import get_model
    get_model()
    _load_all()
    print("Warm.")

    test_query = "What did the QPR manager say about transfers?"

    print("\n--- dense only ---")
    ranked, _ = dense_search(test_query, k=5)
    for idx in ranked:
        _print_result(idx)

    print("\n--- sparse only ---")
    for idx in sparse_search(test_query, k=5):
        _print_result(idx)

    print("\n--- fused (timed, per-stage) ---")
    t0 = time.perf_counter()
    results = retrieve(test_query, k=5, candidate_k=20, verbose=True)
    t1 = time.perf_counter()
    print(f"Total: {(t1 - t0) * 1000:.2f}ms")
    for r in results:
        print(f"[{r['strategy']}] (dense={r['_dense_score']:.3f}) {r['text'][:120]}")


if __name__ == "__main__":
    main()