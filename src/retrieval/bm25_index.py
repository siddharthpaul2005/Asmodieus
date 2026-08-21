"""
src/retrieval/bm25_index.py
Step 5a — BM25 sparse index over chunk_meta.jsonl, via bm25s
(sparse-matrix backed, fast at 300K+ doc scale — rank_bm25 was the
bottleneck at ~436ms/query; bm25s brings this down to single-digit ms).
Stopwords stripped so question-boilerplate ("what","did","say")
doesn't dominate scoring.
"""
import json
import os
import re
import bm25s

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
META_PATH = str(PROJECT_ROOT / "data" / "processed" / "chunk_meta.jsonl")
CORPUS_PATH = str(PROJECT_ROOT / "data" / "processed" / "corpus.jsonl")
INDEX_DIR = str(PROJECT_ROOT / "data" / "processed" / "bm25s_index")

import string
def _strip_punct(text: str) -> str:
    for p in string.punctuation + "।॥":
        text = text.replace(p, " ")
    return text
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "did", "do", "does", "what", "who", "when", "where", "why", "how",
    "which", "this", "that", "these", "those", "and", "or", "but", "if",
    "of", "to", "in", "on", "for", "with", "about", "as", "by", "at",
    "it", "its", "he", "she", "they", "them", "his", "her", "their",
    "you", "your", "i", "we",
}

def tokenize(text: str) -> list[str]:
    words = _strip_punct(text.lower()).split()
    return [t for t in words if t not in _STOPWORDS]

def build_bm25():
    file_path = META_PATH if os.path.exists(META_PATH) else CORPUS_PATH
    chunk_ids = []
    corpus_tokens = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("chunk_id") or str(row.get("idx"))
            chunk_ids.append(cid)
            corpus_tokens.append(tokenize(row["text"]))


    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    os.makedirs(INDEX_DIR, exist_ok=True)
    retriever.save(INDEX_DIR)
    with open(os.path.join(INDEX_DIR, "chunk_ids.json"), "w") as f:
        json.dump(chunk_ids, f)
    print(f"bm25s index built over {len(chunk_ids)} chunks -> {INDEX_DIR}")
    return retriever, chunk_ids

def load_bm25():
    if os.path.exists(INDEX_DIR):
        retriever = bm25s.BM25.load(INDEX_DIR)
        with open(os.path.join(INDEX_DIR, "chunk_ids.json")) as f:
            chunk_ids = json.load(f)
        return retriever, chunk_ids
    return build_bm25()

if __name__ == "__main__":
    build_bm25()