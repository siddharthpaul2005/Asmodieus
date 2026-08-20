# Contract — DO NOT change signatures without telling the other person

# chunking/chunker.py (done)
def chunk_passage(source_id: str, text: str) -> list[Chunk]: ...

# embedding/*.py
def embed_texts(texts: list[str]) -> list[list[float]]: ...  # returns vectors

# retrieval/*.py
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    # returns [{"chunk_id": str, "text": str, "context_text": str,
    #           "score": float, "strategy": str, "source_chunk_id": str}, ...]
    ...

# answering/*.py
def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    # returns {"answer": str | None, "cited_chunk_ids": list[str],
    #          "confidence": float, "grounded": bool}
    ...

# guardrails/*.py 
def check_off_topic(query: str) -> bool: ...
def check_input_safety(query: str) -> bool: ...
def check_grounded(answer: dict, retrieved_chunks: list[dict]) -> bool: ...