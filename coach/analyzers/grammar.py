"""Grammar analyzer: a small offline correction model + a diff.

How it works, per learner turn:
  1. split the turn into sentences (spaCy already did this)
  2. a seq2seq grammatical-error-correction model rewrites each sentence
     correctly (T5-base fine-tuned for GEC, ~220M parameters, CPU)
       "She don't like coffee."  ->  "She doesn't like coffee."
  3. align original and corrected WORDS with difflib.SequenceMatcher; every
     changed stretch becomes one finding, the corrected words its suggestion
       replace  "don't" -> "doesn't"
       delete   "discuss about" -> "discuss"   (finding on "about", suggestion "")
       insert   "my mother also a doctor" -> "...mother is also..."
                (an insertion has no original words, so it is attached to the
                next word: finding on "also", suggestion "is also")

What is ignored, and why:
  - case-only and punctuation-only changes: the text comes from speech
    recognition, whose capitals and commas are guesses, not the learner's
    choices. Comparing lowercase words with punctuation removed drops them.
  - contraction changes ("don't" <-> "do not"): style, not an error.
  - deleted fillers ("umm"): the fluency analyzer already reports those.

Why a model instead of rules: learner errors are endless (tense, agreement,
articles, prepositions, word order); a few hundred rules would cover a
fraction. The trade-off is that the model sometimes "corrects" fine English,
so the eval (eval/eval_grammar.py) measures its precision.
"""

from __future__ import annotations

import difflib
import re
import threading
from pathlib import Path
from typing import Protocol

from spacy.tokens import Doc, Span

from ..errors import Unavailable
from .base import Analyzer, AnalyzerResult, Finding
from .fluency import FILLER_WORDS

HF_DIR = Path(__file__).resolve().parents[2] / "models" / "hf"
GEC_MODEL = "vennify/t5-base-grammar-correction"
WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*")
CONTRACTIONS = {"n't": " not", "'re": " are", "'m": " am", "'ve": " have", "'ll": " will", "'d": " would"}


class Corrector(Protocol):
    """Anything that turns sentences into corrected sentences (tests use a fake)."""
    def correct(self, sentences: list[str]) -> list[str]: ...


class GrammarAnalyzer(Analyzer):
    name = "grammar"

    def __init__(self, corrector: Corrector | None = None):
        # Load (or reuse) the model now, so a missing model is noticed when the
        # session starts (and the analyzer is left out), not mid-session.
        self.corrector = corrector or get_corrector()

    def analyze(self, doc: Doc) -> AnalyzerResult:
        # Sentences with no letters ("...", "2023") have nothing to correct.
        sents = [s for s in doc.sents if any(t.is_alpha for t in s)]
        corrected = self.corrector.correct([s.text for s in sents]) if sents else []
        findings: list[Finding] = []
        for sent, fixed in zip(sents, corrected):
            findings.extend(self._diff(doc, sent, fixed))
        return AnalyzerResult(self.name, findings,
                              {"error_count": len(findings), "sentences": len(sents)})

    def _diff(self, doc: Doc, sent: Span, fixed: str) -> list[Finding]:
        orig = [(m.group(), sent.start_char + m.start(), sent.start_char + m.end())
                for m in WORD.finditer(sent.text)]
        new = [m.group() for m in WORD.finditer(fixed)]
        a = [norm(w) for w, _, _ in orig]
        b = [norm(w) for w in new]
        out = []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
            if op == "equal":
                continue
            if op != "insert" and same_meaning(a[i1:i2], b[j1:j2]):
                continue
            if op == "delete" and all(w in FILLER_WORDS for w in a[i1:i2]):
                continue
            fix = " ".join(new[j1:j2])
            if op == "insert":
                if not orig:
                    continue
                # Attach the missing words to a real word so there is something to highlight.
                if i1 < len(orig):
                    i2 = i1 + 1
                    fix = f"{fix} {orig[i1][0]}"
                else:
                    i1, i2 = len(orig) - 1, len(orig)
                    fix = f"{orig[i1][0]} {fix}"
            start, end = orig[i1][1], orig[i2 - 1][2]
            text = doc.text[start:end]
            msg = (f'You can leave out "{text}".' if not fix
                   else f'Try "{fix}" instead of "{text}".')
            out.append(Finding(self.name, "grammar", text, start, end, msg, [fix] if fix else []))
        return out


def norm(word: str) -> str:
    return word.lower().replace("’", "'")


def same_meaning(a: list[str], b: list[str]) -> bool:
    """True if two word lists differ only by contractions ("don't" vs "do not")."""
    def expand(words: list[str]) -> str:
        s = " ".join(words)
        for short, long in CONTRACTIONS.items():
            s = s.replace(short, long)
        return s.replace("won not", "will not").replace("ca not", "can not").replace("cannot", "can not")
    return expand(a).split() == expand(b).split()


class T5Corrector:
    """vennify/t5-base-grammar-correction on the CPU, from models/hf/ only."""

    def __init__(self, model_name: str = GEC_MODEL, cache_dir: Path = HF_DIR):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise Unavailable("Grammar feedback off: transformers/torch are not installed.") from exc
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
        except OSError as exc:
            raise Unavailable("Grammar feedback off: the grammar model is missing. Run: python setup_models.py") from exc
        self.model.eval()
        self.torch = torch

    def correct(self, sentences: list[str]) -> list[str]:
        # The model was trained with this "grammar: " task prefix.
        batch = self.tokenizer([f"grammar: {s}" for s in sentences], return_tensors="pt",
                               padding=True, truncation=True, max_length=128)
        with self.torch.no_grad():
            out = self.model.generate(**batch, max_new_tokens=int(batch["input_ids"].shape[1] * 1.5) + 8,
                                      num_beams=4)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)


_lock = threading.Lock()
_corrector: T5Corrector | None = None
_error: Unavailable | None = None


def get_corrector() -> T5Corrector:
    """Load the correction model once per process and share it. A failure is
    remembered, so a missing model isn't searched for again at every session."""
    global _corrector, _error
    with _lock:
        if _error is not None:
            raise _error
        if _corrector is None:
            try:
                _corrector = T5Corrector()
            except Unavailable as err:
                _error = err
                raise
        return _corrector
