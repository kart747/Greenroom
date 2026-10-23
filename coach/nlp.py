"""Shared preprocessing: runs once per turn, reused by every analyzer.

spaCy gives us sentence segmentation, tokenization, lemmas and POS tags in one
pass -- the classical NLP base layer.

For spoken turns, Whisper's word timings ride along on the Doc as
doc._.timing (a spaCy custom attribute). That keeps the analyzer contract
"a Doc in, findings out" unchanged while letting the pace analyzer see time.
Typed turns have doc._.timing = None.
"""

from __future__ import annotations

from functools import lru_cache

import spacy
from spacy.tokens import Doc

MODEL = "en_core_web_sm"

if not Doc.has_extension("timing"):
    Doc.set_extension("timing", default=None)


@lru_cache(maxsize=1)
def get_nlp():
    try:
        return spacy.load(MODEL)
    except OSError as exc:
        raise SystemExit(
            f"spaCy model '{MODEL}' missing. Run: python -m spacy download {MODEL}"
        ) from exc


def preprocess(text: str, timing: list[dict] | None = None) -> Doc:
    doc = get_nlp()(text)
    doc._.timing = timing
    return doc
