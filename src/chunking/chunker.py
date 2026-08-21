"""
src/chunking/chunker.py
Step 4c — Multi-strategy chunker: fixed+overlap, semantic breakpoint,
sentence-window. All three run per passage, tagged by strategy in
metadata, fused at query time via RRF (see retrieval/).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np


@dataclass
class Chunk:
    chunk_id: str
    source_chunk_id: str
    text: str
    context_text: str
    strategy: str
    token_count: int
    extra_meta: dict = field(default_factory=dict)


_SENT_SPLIT_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z0-9"\u2018\u201c])'
)


def split_sentences(text: str) -> List[str]:
    text = text.strip()

    if not text:
        return []

    sents = _SENT_SPLIT_RE.split(text)

    return [s.strip() for s in sents if s.strip()]


def word_count(text: str) -> int:
    return len(text.split())


def chunk_fixed(
    source_id: str,
    text: str,
    window: int = 256,
    overlap_pct: float = 0.15
) -> List[Chunk]:

    words = text.split()

    if len(words) <= window:
        return [
            Chunk(
                chunk_id=f"{source_id}_fixed_0",
                source_chunk_id=source_id,
                text=text,
                context_text=text,
                strategy="fixed",
                token_count=len(words),
            )
        ]

    step = max(1, int(window * (1 - overlap_pct)))

    chunks = []
    idx = 0

    for start in range(0, len(words), step):
        piece = words[start:start + window]

        if not piece:
            break

        chunk_text = " ".join(piece)

        chunks.append(
            Chunk(
                chunk_id=f"{source_id}_fixed_{idx}",
                source_chunk_id=source_id,
                text=chunk_text,
                context_text=chunk_text,
                strategy="fixed",
                token_count=len(piece),
            )
        )

        idx += 1

        if start + window >= len(words):
            break

    return chunks


def chunk_semantic(
    source_id: str,
    text: str,
    embed_fn: Callable[[List[str]], "np.ndarray"],
) -> List[Chunk]:

    sents = split_sentences(text)

    if len(sents) <= 1:
        return [
            Chunk(
                chunk_id=f"{source_id}_semantic_0",
                source_chunk_id=source_id,
                text=text,
                context_text=text,
                strategy="semantic",
                token_count=word_count(text),
            )
        ]

    vecs = embed_fn(sents)

    sims = [
        float(np.dot(vecs[i], vecs[i + 1]))
        for i in range(len(sents) - 1)
    ]

    mean_sim = sum(sims) / len(sims)

    var = sum(
        (s - mean_sim) ** 2
        for s in sims
    ) / len(sims)

    std_sim = var ** 0.5

    threshold = max(
        0.5,
        mean_sim - 0.5 * std_sim
    )

    groups: List[List[str]] = [[sents[0]]]

    for i, sim in enumerate(sims):
        if sim < threshold:
            groups.append([sents[i + 1]])
        else:
            groups[-1].append(sents[i + 1])

    chunks = []

    for idx, g in enumerate(groups):
        chunk_text = " ".join(g)

        chunks.append(
            Chunk(
                chunk_id=f"{source_id}_semantic_{idx}",
                source_chunk_id=source_id,
                text=chunk_text,
                context_text=chunk_text,
                strategy="semantic",
                token_count=word_count(chunk_text),
            )
        )

    return chunks


def chunk_sentence_window(
    source_id: str,
    text: str,
    window: int = 2
) -> List[Chunk]:

    sents = split_sentences(text)

    if not sents:
        return []

    chunks = []

    for i, sent in enumerate(sents):
        lo = max(0, i - window)
        hi = min(len(sents), i + window + 1)

        context = " ".join(sents[lo:hi])

        chunks.append(
            Chunk(
                chunk_id=f"{source_id}_window_{i}",
                source_chunk_id=source_id,
                text=sent,
                context_text=context,
                strategy="sentence_window",
                token_count=word_count(sent),
            )
        )

    return chunks


def chunk_passage(
    chunk_id: str,
    text: str,
    embed_fn: Callable[[List[str]], "np.ndarray"],
) -> List[Chunk]:

    if not text or not text.strip():
        return []

    out: List[Chunk] = []

    out.extend(
        chunk_fixed(chunk_id, text)
    )

    out.extend(
        chunk_semantic(
            chunk_id,
            text,
            embed_fn
        )
    )

    out.extend(
        chunk_sentence_window(
            chunk_id,
            text
        )
    )

    return out