'''
EMBEDDER MODEL WRAPPER
Multilingual e5-small — 384-dim, normalized embeddings, supports
Hindi/Indic + English (MSMARCO-XI is multilingual). Requires
"query: "/"passage: " prefixes per e5's training scheme — dropping
these measurably hurts retrieval quality, not cosmetic.
'''
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "intfloat/multilingual-e5-small"
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {_MODEL_NAME} on {device}...")
        _model = SentenceTransformer(_MODEL_NAME, device=device)
    return _model

def embed_texts(texts: list[str]) -> np.ndarray:
    """For indexing passages/chunks — e5 requires 'passage: ' prefix."""
    model = get_model()
    prefixed = [f"passage: {t}" for t in texts]
    return model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=True)

def embed_query(text: str) -> np.ndarray:
    """For queries at retrieval time — e5 requires 'query: ' prefix."""
    model = get_model()
    return model.encode([f"query: {text}"], convert_to_numpy=True, normalize_embeddings=True)[0]

def cosine_sim(a: str, b: str) -> float:
    vecs = embed_texts([a, b])
    return float(np.dot(vecs[0], vecs[1]))