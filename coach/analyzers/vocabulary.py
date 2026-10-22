"""Vocabulary analyzer: finds overused basic words and suggests stronger ones.

CLASSICAL BASELINE
  - lemma lookup in a curated lexicon of overused basic words
    (data/basic_words.json), so "got", "gets" and "getting" all match "get"
  - POS-aware: only flag a word whose part of speech fits the lexicon entry
    ("good" the adjective, not "for good"; "hard" in "work hard" is an adverb)
  - intensifier + adjective ("very good") flagged as one phrase
  - fixed idioms are skipped ("for good", "good morning", phrasal verbs like
    "get up"), even when spaCy misparses them
  - lexical diversity: type-token ratio over content-word lemmas
  - suggestions are inflected to match the learner's word with lemminflect:
    "got" -> "received", "bigger" -> "larger", "things" -> "matters"
  - it suggests the SAME first three candidates in every sentence

CONTEXTUAL VERSION (config["analyzers"]["contextual"])
  The baseline is context-blind: "bright" fits "a bright student" but not
  "a bright problem". A masked language model (DistilBERT) has learned which
  words fit which contexts, so we ask it:
    1. replace the whole flagged span with ONE [MASK]
       ("The food was very good." -> "The food was [MASK].")
       The intensifier goes too, because the stronger word replaces both.
    2. score every candidate at the mask position (fill-mask with targets=)
    3. sort by score; drop candidates below 2% of the best one (keep >= 1)
  Compare the two with: python -m eval.eval_vocab --table
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol

from spacy.tokens import Doc, Span, Token

from ..errors import Unavailable
from .base import Analyzer, AnalyzerResult, Finding

LEXICON_PATH = Path(__file__).resolve().parents[2] / "data" / "basic_words.json"
HF_DIR = Path(__file__).resolve().parents[2] / "models" / "hf"
BERT_MODEL = "distilbert/distilbert-base-uncased"
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
LEXICON_POS = {"ADJ", "NOUN", "VERB"}
MIN_RELATIVE_SCORE = 0.02     # keep candidates scoring >= 2% of the best one

# Penn tags where the candidate must change form to fit the sentence.
# Base forms (JJ, NN, VB, VBP) are left alone: inflecting "things" (a plural
# lexicon entry for "stuff") to the singular NN tag would give "thing".
INFLECT_TAGS = {"NNS", "VBD", "VBG", "VBN", "VBZ", "JJR", "JJS"}


class MaskScorer(Protocol):
    """What the contextual ranker needs from a masked language model.
    The real one wraps transformers; tests pass a fake."""
    mask_token: str

    def single_piece(self, word: str) -> bool: ...
    def score(self, text: str, targets: list[str]) -> dict[str, float]: ...


class VocabularyAnalyzer(Analyzer):
    name = "vocabulary"

    def __init__(self, lexicon_path: Path = LEXICON_PATH, use_contextual: bool = False,
                 scorer: MaskScorer | None = None):
        data = json.loads(lexicon_path.read_text())
        self.lexicon: dict[str, list[str]] = data["words"]
        self.intensifiers: set[str] = set(data["intensifiers"])
        self.idioms: set[tuple[str, str]] = {tuple(p) for p in data.get("idioms", [])}
        self.use_contextual = use_contextual
        self._scorer = scorer          # None = load DistilBERT lazily on first use

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

            suggestions = [inflect(c, tok.tag_) for c in self.lexicon[tok.lemma_.lower()]]
            if self.use_contextual:
                suggestions = self.contextual_suggestions(doc, tok, suggestions, span)

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
        # Idiom bigrams, checked on the words themselves, because spaCy sometimes
        # misparses them: in "left for good last year" it reads "good" as an
        # adjective describing "year", so the pobj rule above misses it.
        doc = tok.doc
        prev = doc[tok.i - 1].lower_ if tok.i > 0 else ""
        nxt = doc[tok.i + 1].lower_ if tok.i + 1 < len(doc) else ""
        if (prev, tok.lower_) in self.idioms or (tok.lower_, nxt) in self.idioms:
            return False
        # Phrasal verbs ("get up", "get along") mean something else entirely;
        # "receive up" is nonsense.
        if tok.pos_ == "VERB" and any(c.dep_ == "prt" for c in tok.children):
            return False
        return True

    def contextual_suggestions(self, doc: Doc, target: Token, candidates: list[str],
                               span: Span | None = None) -> list[str]:
        """Rank `candidates` (already inflected) by how well they fit THIS sentence."""
        span = span if span is not None else doc[target.i:target.i + 1]
        try:
            scorer = self._scorer or get_mask_scorer()
        except Unavailable:
            return candidates          # model missing: keep the baseline order
        # Mask the whole span inside its sentence; the sentence is all the
        # context the model needs, and shorter input is faster.
        sent = span.sent
        offset = sent.start_char
        text = (sent.text[:span.start_char - offset] + scorer.mask_token
                + sent.text[span.end_char - offset:])

        # One [MASK] = one word-piece. A candidate the tokenizer splits into
        # several pieces ("thought-provoking" -> thought, -, provoking) cannot be
        # scored at a single mask fairly: scoring only its first piece would
        # really score a different word. So those are kept, unscored, after the
        # scored ones instead of being dropped (they may still be good words).
        scorable = [c for c in candidates if scorer.single_piece(c)]
        unscored = [c for c in candidates if c not in scorable]
        if not scorable:
            return candidates
        scores = scorer.score(text, scorable)
        ranked = sorted(scorable, key=lambda c: scores.get(c.lower(), 0.0), reverse=True)
        best = scores.get(ranked[0].lower(), 0.0)
        kept = [c for c in ranked if scores.get(c.lower(), 0.0) >= MIN_RELATIVE_SCORE * best] or ranked[:1]
        return kept + unscored


def inflect(candidate: str, tag: str) -> str:
    """Put a base-form candidate into the learner's grammatical form."""
    if tag not in INFLECT_TAGS or " " in candidate or "-" in candidate:
        return candidate
    try:
        from lemminflect import getInflection
    except ImportError:
        return candidate               # optional: without it, base forms are still useful
    if tag in ("JJR", "JJS"):
        # Adjectives: only trust lemminflect's dictionary. Its rules would invent
        # "majorest" or "significanter"; long adjectives take "more"/"most".
        forms = getInflection(candidate, tag=tag, inflect_oov=False)
        if forms:
            return forms[0]
        return ("more " if tag == "JJR" else "most ") + candidate
    forms = getInflection(candidate, tag=tag)
    return forms[0] if forms else candidate


class HFMaskScorer:
    """DistilBERT fill-mask on the CPU (the GPU is busy with the LLM and Whisper).
    Loaded from models/hf/ only; setup_models.py downloads it once."""

    def __init__(self, model_name: str = BERT_MODEL, cache_dir: Path = HF_DIR):
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise Unavailable("Smart suggestions off: transformers is not installed.") from exc
        try:
            tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
            model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
        except OSError as exc:
            raise Unavailable("Smart suggestions off: the BERT model is missing. Run: python setup_models.py") from exc
        self.tokenizer = tok
        self.mask_token = tok.mask_token
        self.pipe = pipeline("fill-mask", model=model, tokenizer=tok, device=-1)

    def single_piece(self, word: str) -> bool:
        return len(self.tokenizer.tokenize(word)) == 1

    def score(self, text: str, targets: list[str]) -> dict[str, float]:
        out = self.pipe(text, targets=[t.lower() for t in targets], top_k=len(targets))
        return {o["token_str"].strip(): float(o["score"]) for o in out}


_scorer_lock = threading.Lock()
_scorer: HFMaskScorer | None = None
_scorer_error: Unavailable | None = None


def get_mask_scorer() -> HFMaskScorer:
    """Load DistilBERT once per process and share it (it's ~260 MB in memory).
    A failure is remembered too: retrying a missing model on every flagged
    word would cost ~0.1 s each time for nothing."""
    global _scorer, _scorer_error
    with _scorer_lock:
        if _scorer_error is not None:
            raise _scorer_error
        if _scorer is None:
            try:
                _scorer = HFMaskScorer()
            except Unavailable as err:
                _scorer_error = err
                raise
        return _scorer
