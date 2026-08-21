"""
src/answering/extractive.py
Step 6 — Zero-model extractive answering with multilingual support.

Change from previous version:
- Added a lexical-overlap gate (_term_overlap) on top of the dense-score
  gate. Dense cosine similarity alone is not a reliable relevance signal
  for short/generic web-boilerplate passages (anisotropy — generic text
  sits unusually close to many unrelated queries in embedding space, so it
  can score 0.8+ against almost anything). Requiring some real shared
  content words between query and passage catches exactly the case where
  a passage is a "confident" but content-free match.
- If retrieval returns nothing that clears BOTH gates, this now refuses
  instead of returning the top dense hit regardless. A correct refusal is
  a better production behavior than a fluent wrong answer.
"""
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.retrieval.hybrid_retriever import retrieve

DENSE_SCORE_FLOOR = 0.85
MIN_TERM_OVERLAP = 0.40  # fraction of query content-words that must appear in the passage

SUPPORTED_LANGS = {"eng", "hin", "ben", "guj"}

_NO_INFO_MSG = {
    "eng": "I don't have enough information in the retrieved context to answer that confidently.",
    "hin": "मेरे पास इस प्रश्न का विश्वसनीय उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
    "ben": "এই প্রশ্নের নির্ভরযোগ্য উত্তর দেওয়ার জন্য আমার কাছে পর্যাপ্ত তথ্য নেই।",
    "guj": "આ પ્રશ્નનો વિશ્વસનીય જવાબ આપવા માટે મારી પાસે પૂરતી માહિતી નથી.",
}

_ANSWER_PREFIX = {
    "eng": "",
    "hin": "",
    "ben": "",
    "guj": "",
}

_STOPWORDS = {
    "eng": {"what", "was", "is", "are", "the", "a", "an", "of", "did", "does",
            "how", "why", "who", "when", "where", "which", "to", "in", "on",
            "for", "and", "or", "it", "that", "this", "be", "do"},
}


def _msg(lang: str, table: dict) -> str:
    return table.get(lang, table["eng"])


def _content_terms(text: str, lang: str) -> set:
    import string
    # Replace standard punctuation and Indic full stops (danda) with space
    for p in string.punctuation + "।॥":
        text = text.replace(p, " ")
    words = text.lower().split()
    stop = _STOPWORDS.get(lang, set())
    return {w for w in words if w not in stop and len(w) > 1}


def _term_overlap(query: str, passage: str, lang: str) -> float:
    """Fraction of query content-terms that also appear in the passage."""
    q_terms = _content_terms(query, lang)
    if not q_terms:
        return 0.0
    p_terms = _content_terms(passage, lang)
    shared = q_terms & p_terms
    return len(shared) / len(q_terms)


def answer(query: str, lang: str, k: int = 5, candidate_k: int = 20) -> dict:
    if lang not in SUPPORTED_LANGS:
        lang = "eng"

    chunks = retrieve(query, k=k, candidate_k=candidate_k, lang=lang)

    if not chunks:
        return {
            "answer": _msg(lang, _NO_INFO_MSG),
            "cited_chunk_ids": [],
            "confidence": 0.0,
            "language": lang,
            "refused": True,
        }

    same_lang_chunks = [c for c in chunks if c.get("language") == lang]

    # walk same-language candidates in rank order, return the first one that
    # clears BOTH gates — dense similarity AND real term overlap
    for c in same_lang_chunks:
        text = c.get("context_text") or c["text"]
        dense_score = c.get("_dense_score", 0.0)
        overlap = _term_overlap(query, text, lang)

        if dense_score >= DENSE_SCORE_FLOOR and overlap >= MIN_TERM_OVERLAP:
            return {
                "answer": _msg(lang, _ANSWER_PREFIX) + text,
                "cited_chunk_ids": [ch["chunk_id"] for ch in chunks],
                "confidence": round(dense_score, 3),
                "term_overlap": round(overlap, 3),
                "language": lang,
                "refused": False,
            }

    # nothing cleared both gates -> refuse rather than guess
    top = same_lang_chunks[0] if same_lang_chunks else chunks[0]
    return {
        "answer": _msg(lang, _NO_INFO_MSG),
        "cited_chunk_ids": [c["chunk_id"] for c in chunks],
        "confidence": round(top.get("_dense_score", 0.0), 3),
        "language": lang,
        "refused": True,
    }


if __name__ == "__main__":
    import time
    from src.guardrails.lang_detect import detect_language

    print("Extractive answering — type a question (Ctrl+C to quit)")
    print("Language is auto-detected from script (no prefix needed)\n")
    while True:
        try:
            q = input("Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not q:
            continue

        q_lang = detect_language(q)

        t0 = time.perf_counter()
        result = answer(q, lang=q_lang)
        t1 = time.perf_counter()
        print(f"  ({(t1 - t0) * 1000:.1f}ms) lang={result['language']} refused={result['refused']} "
              f"confidence={result['confidence']} overlap={result.get('term_overlap', '-')}")
        print(f"  Answer: {result['answer'][:300]}")
        print(f"  Cited:  {result['cited_chunk_ids']}\n")