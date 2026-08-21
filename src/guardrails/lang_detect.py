"""
src/guardrails/lang_detect.py
Detects language from Unicode script ranges. No model, no API call —
this is deliberately not ML-based since STT (Sarvam) hands you raw native
script text directly (Devanagari for Hindi, Bengali script for Bengali,
Gujarati script for Gujarati) with no language tag attached, and script
range checks are ~instant, which matters under a 200ms budget.

Counts characters per script rather than checking just the first char,
since punctuation/digits/spaces at the start could otherwise misclassify.
"""

_RANGES = {
    "hin": (0x0900, 0x097F),  # Devanagari
    "ben": (0x0980, 0x09FF),  # Bengali
    "guj": (0x0A80, 0x0AFF),  # Gujarati
}


def detect_language(text: str) -> str:
    counts = {lang: 0 for lang in _RANGES}
    for ch in text:
        cp = ord(ch)
        for lang, (lo, hi) in _RANGES.items():
            if lo <= cp <= hi:
                counts[lang] += 1
                break

    best_lang = max(counts, key=counts.get)
    if counts[best_lang] > 0:
        return best_lang
    return "eng"  # no native-script characters found -> Latin/English