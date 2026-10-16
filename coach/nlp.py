"""Shared preprocessing: runs once per turn, reused by every analyzer.

spaCy gives us sentence segmentation, tokenization, lemmas and POS tags in one
pass -- the classical NLP base layer.
"""

from __future__ import annotations

from functools import lru_cache

import spacy
from spacy.tokens import Doc

MODEL = "en_core_web_sm"


@lru_cache(maxsize=1)
def get_nlp():
    try:
        return spacy.load(MODEL)
    except OSError as exc:
        raise SystemExit(
            f"spaCy model '{MODEL}' missing. Run: python -m spacy download {MODEL}"
        ) from exc


def preprocess(text: str) -> Doc:
    return get_nlp()(text)
