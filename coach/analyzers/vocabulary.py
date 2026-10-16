"""Vocabulary analyzer.

CLASSICAL BASELINE (implemented):
  - lemma lookup in a curated lexicon of overused basic words
  - POS-aware: "light" as ADJ vs NOUN matters, so we only flag words whose POS
    fits the lexicon entry
  - intensifier + adjective pattern ("very good") flagged as one phrase
  - lexical diversity: type-token ratio over content-word lemmas

MODERN VERSION (TODO -- yours to implement, see `contextual_suggestions`):
  The baseline suggests the SAME synonyms everywhere. "bright" fits "a bright
  student" but not "a bright problem". A masked language model (BERT) can rank
  candidates by how well they fit THIS sentence. Implement it, then compare the
  two versions on eval/vocab_testset.jsonl -- that comparison is your report.
"""

from __future__ import annotations

import json
from pathlib import Path

from spacy.tokens import Doc, Token

from .base import Analyzer, AnalyzerResult, Finding

LEXICON_PATH = Path(__file__).resolve().parents[2] / "data" / "basic_words.json"
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
LEXICON_POS = {"ADJ", "NOUN", "VERB"}


class VocabularyAnalyzer(Analyzer):
    name = "vocabulary"

    def __init__(self, lexicon_path: Path = LEXICON_PATH, use_contextual: bool = False):
        data = json.loads(lexicon_path.read_text())
        self.lexicon: dict[str, list[str]] = data["words"]
        self.intensifiers: set[str] = set(data["intensifiers"])
        self.use_contextual = use_contextual

    def analyze(self, doc: Doc) -> AnalyzerResult:
        findings: list[Finding] = []
        covered: set[int] = set()  # token indices already part of a finding

        for tok in doc:
            if tok.i in covered or not self._is_basic(tok):
                continue
            start = tok.i
            # extend left over an intensifier: "very good", "really nice"
            if tok.i > 0 and doc[tok.i - 1].lower_ in self.intensifiers:
                start = tok.i - 1
            span = doc[start:tok.i + 1]
            covered.update(range(start, tok.i + 1))

            suggestions = self.lexicon[tok.lemma_.lower()]
            if self.use_contextual:
                suggestions = self.contextual_suggestions(doc, tok, suggestions)

            kind = "intensified_basic" if start < tok.i else "basic_word"
            msg = (f'"{span.text}" is very common. A more precise word makes you sound fluent.'
                   if kind == "basic_word"
                   else f'"{span.text}" leans on an intensifier. One strong word works better.')
            findings.append(Finding(self.name, kind, span.text, span.start_char,
                                    span.end_char, msg, suggestions[:3]))

        content_lemmas = [t.lemma_.lower() for t in doc if t.pos_ in CONTENT_POS and not t.is_stop]
        ttr = len(set(content_lemmas)) / len(content_lemmas) if content_lemmas else 0.0
        metrics = {
            "content_words": len(content_lemmas),
            "type_token_ratio": round(ttr, 3),
            "basic_word_count": len(findings),
        }
        return AnalyzerResult(self.name, findings, metrics)

    def _is_basic(self, tok: Token) -> bool:
        if tok.lemma_.lower() not in self.lexicon or tok.pos_ not in LEXICON_POS:
            return False
        # An adjective that is the object of a preposition is a fixed idiom
        # ("for good", "at best", "in short") -- replacing it breaks the meaning.
        if tok.pos_ == "ADJ" and tok.dep_ == "pobj":
            return False
        return True

    def contextual_suggestions(self, doc: Doc, target: Token, candidates: list[str]) -> list[str]:
        """TODO (your NLP part): rank `candidates` by fit in this exact sentence.

        Suggested approach with a masked language model (e.g. distilbert-base-uncased
        via transformers' "fill-mask" pipeline, runs offline once downloaded):
          1. Replace the target word in the sentence with the [MASK] token.
          2. Ask the model for scores of each candidate at the mask position
             (fill-mask supports a `targets=` argument).
          3. Return candidates sorted by score; drop ones scoring very low.

        Think about: multi-word spans ("very good" -> mask both?), candidates
        the tokenizer splits into several word-pieces, and latency per call.
        """
        raise NotImplementedError("Contextual ranking not implemented yet")
