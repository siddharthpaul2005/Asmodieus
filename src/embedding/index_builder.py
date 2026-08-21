"""
Step 4b — Build the vector index from corpus.jsonl.
Runs chunker.chunk_passage on every passage, embeds every resulting
chunk, stores in HNSW.

Saves:
    data/processed/index.bin
    data/processed/chunk_meta.jsonl
"""

import json
import os
import sys

import hnswlib
import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
)

from src.chunking.chunker import chunk_passage
from src.embedding.embedder import embed_texts, cosine_sim


CORPUS_PATH = "data/processed/corpus.jsonl"
INDEX_PATH = "data/processed/index.bin"
META_PATH = "data/processed/chunk_meta.jsonl"

DIM = 384


def build():
    # ---------------------------------------------------------
    # 1. Load corpus
    # ---------------------------------------------------------
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    # ---------------------------------------------------------
    # 2. Chunk every passage
    # ---------------------------------------------------------
    all_chunks = []

    for i, row in enumerate(passages):
        chunks = chunk_passage(
            row["chunk_id"],
            row["text"],
            embed_fn=embed_texts
        )

        # Thread language information into every generated chunk
        for c in chunks:
            c.extra_meta["language"] = row.get("language", "eng")

        all_chunks.extend(chunks)

        if (i + 1) % 500 == 0:
            print(
                f"Chunked {i + 1}/{len(passages)} passages "
                f"-> {len(all_chunks)} chunks so far"
            )

    print(f"Total chunks across all strategies: {len(all_chunks)}")

    # ---------------------------------------------------------
    # 3. Embed all chunks
    # ---------------------------------------------------------
    texts = [c.text for c in all_chunks]

    BATCH = 256
    vectors = []

    for i in range(0, len(texts), BATCH):
        batch_vectors = embed_texts(texts[i:i + BATCH])
        vectors.append(batch_vectors)

        print(
            f"Embedded {min(i + BATCH, len(texts))}/{len(texts)}"
        )

    vectors = np.vstack(vectors).astype(np.float32)

    # ---------------------------------------------------------
    # 4. Build HNSW vector index
    # ---------------------------------------------------------
    index = hnswlib.Index(
        space="cosine",
        dim=DIM
    )

    index.init_index(
        max_elements=len(all_chunks),
        ef_construction=200,
        M=16
    )

    index.add_items(
        vectors,
        np.arange(len(all_chunks))
    )

    index.set_ef(50)

    index.save_index(INDEX_PATH)

    # ---------------------------------------------------------
    # 5. Save chunk metadata
    # ---------------------------------------------------------
    with open(META_PATH, "w", encoding="utf-8") as f:
        for i, c in enumerate(all_chunks):
            f.write(
                json.dumps(
                    {
                        "idx": i,
                        "chunk_id": c.chunk_id,
                        "source_chunk_id": c.source_chunk_id,
                        "text": c.text,
                        "context_text": c.context_text,
                        "strategy": c.strategy,
                        "token_count": c.token_count,
                        "language": c.extra_meta.get(
                            "language",
                            "eng"
                        ),
                    },
                    ensure_ascii=False
                )
                + "\n"
            )

    # ---------------------------------------------------------
    # 6. Done
    # ---------------------------------------------------------
    print(f"Saved index -> {INDEX_PATH}")
    print(f"Saved metadata -> {META_PATH}")


if __name__ == "__main__":
    build()