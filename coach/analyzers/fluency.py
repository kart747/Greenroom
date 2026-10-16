"""Fluency analyzer (rule-based, classical NLP).

Detects filler words/phrases, immediate repetitions ("I I went") and computes
simple fluency stats. Pure pattern matching over tokens -- fast and explainable.
"""

from __future__ import annotations

from spacy.tokens import Doc

from .base import Analyzer, AnalyzerResult, Finding

FILLER_WORDS = {"um", "umm", "uh", "uhh", "er", "erm", "hmm", "ah"}
FILLER_PHRASES = [("you", "know"), ("i", "mean"), ("kind", "of"), ("sort", "of")]


class FluencyAnalyzer(Analyzer):
    name = "fluency"

    def analyze(self, doc: Doc) -> AnalyzerResult:
        findings: list[Finding] = []
        words = [t for t in doc if not t.is_punct and not t.is_space]
        lower = [t.lower_ for t in words]

        # 1. single-word fillers
        for tok in words:
            if tok.lower_ in FILLER_WORDS:
                findings.append(self._finding("filler", doc, tok.i, tok.i + 1,
                    f'"{tok.text}" is a filler. A short silent pause sounds more confident.'))

        # 2. multi-word fillers ("you know", "I mean")
        for i in range(len(words) - 1):
            pair = (lower[i], lower[i + 1])
            if pair in FILLER_PHRASES:
                # "kind of"/"sort of" before a noun is real meaning ("a kind of fruit")
                if pair[0] in {"kind", "sort"} and i > 0 and lower[i - 1] in {"a", "the", "this", "that", "what"}:
                    continue
                findings.append(self._finding("filler_phrase", doc, words[i].i, words[i + 1].i + 1,
                    f'"{words[i].text} {words[i + 1].text}" used as a filler phrase.'))

        # 3. immediate repetition ("I I", "the the")
        for i in range(len(words) - 1):
            if lower[i] == lower[i + 1] and lower[i] not in FILLER_WORDS and lower[i] not in {"very", "really", "so", "no", "bye", "ha"}:
                findings.append(self._finding("repetition", doc, words[i].i, words[i + 1].i + 1,
                    f'Repeated "{words[i].text}". Try to restart the sentence calmly instead.'))

        n_words = len(words)
        n_sents = max(1, len(list(doc.sents)))
        fillers = sum(1 for f in findings if f.kind.startswith("filler"))
        metrics = {
            "word_count": n_words,
            "avg_sentence_length": round(n_words / n_sents, 2),
            "fillers_per_100_words": round(100 * fillers / n_words, 2) if n_words else 0.0,
        }
        return AnalyzerResult(self.name, findings, metrics)

    def _finding(self, kind: str, doc: Doc, start_tok: int, end_tok: int, message: str) -> Finding:
        span = doc[start_tok:end_tok]
        return Finding(self.name, kind, span.text, span.start_char, span.end_char, message)
