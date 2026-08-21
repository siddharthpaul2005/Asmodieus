"""
src/guardrails/checker.py

Backend guardrail enforcement — source of truth.
The client-side guardrail (frontend/src/App.jsx) mirrors these tiers for
instant UX feedback, but this module is what actually gates the RAG pipeline.

Four sequential checks (interface.py contract):
    check_off_topic(query: str) -> bool
    check_input_safety(query: str) -> bool
    check_grounded(answer: dict, retrieved_chunks: list[dict]) -> bool

run_all_guardrails() orchestrates all tiers and returns a GuardrailResult.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional


# ─── Dynamic Rules Loading ──────────────────────────────────────────────────

import json
from pathlib import Path

_TIER1_PATTERNS: list[re.Pattern] = []
_TIER2_PATTERNS: list[re.Pattern] = []
_TIER3_PATTERNS: list[re.Pattern] = []

def _load_guardrails():
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        json_path = project_root / "data" / "guardrails.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # TIER 1
        t1 = data.get('TIER_1', {})
        for pat in t1.get('regex_patterns', []):
            _TIER1_PATTERNS.append(re.compile(pat, re.IGNORECASE))
        
        exact_words = t1.get('exact_words', [])
        if exact_words:
            escaped_words = [re.escape(w) for w in exact_words]
            chunk_size = 500
            for i in range(0, len(escaped_words), chunk_size):
                chunk = escaped_words[i:i + chunk_size]
                pattern = r"(?:^|\s|[.,!?;:])(" + "|".join(chunk) + r")(?=$|\s|[.,!?;:])"
                _TIER1_PATTERNS.append(re.compile(pattern, re.IGNORECASE))

        # TIER 2
        for pat in data.get('TIER_2', {}).get('regex_patterns', []):
            _TIER2_PATTERNS.append(re.compile(pat, re.IGNORECASE))

        # TIER 3
        for pat in data.get('TIER_3', {}).get('regex_patterns', []):
            _TIER3_PATTERNS.append(re.compile(pat, re.IGNORECASE))
            
    except Exception as e:
        print(f"Failed to load guardrails.json: {e}")

_load_guardrails()

def run_all_guardrails(query: str) -> GuardrailResult:
    """
    Run all tiers sequentially, highest severity first.
    Stops at the first tier that fires and returns a GuardrailResult.
    Blocked queries should NEVER be forwarded to the RAG pipeline.

    Usage:
        result = run_all_guardrails(query)
        if result.blocked:
            return {"refusal": "Not ethically correct to answer."}
        # ... proceed with RAG
    """
    t0 = time.perf_counter()

    tier_map = [
        ("TIER_1", _TIER1_PATTERNS),
        ("TIER_2", _TIER2_PATTERNS),
        ("TIER_3", _TIER3_PATTERNS),
    ]

    for tier_key, patterns in tier_map:
        flagged: list[str] = []
        for pattern in patterns:
            matches = pattern.findall(query)
            flagged.extend(matches)

        if flagged:
            latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            meta = _TIER_META[tier_key]
            return GuardrailResult(
                blocked=True,
                tier=tier_key,
                tier_label=meta["label"],
                category=meta["category"],
                action=meta["action"],
                flagged_words=list(set(w.lower() for w in flagged)),
                latency_ms=latency_ms,
            )

    latency_ms = round((time.perf_counter() - t0) * 1000, 3)
    return GuardrailResult(
        blocked=False,
        tier=None,
        tier_label=None,
        category="Clean",
        action="PASS",
        flagged_words=[],
        latency_ms=latency_ms,
    )
