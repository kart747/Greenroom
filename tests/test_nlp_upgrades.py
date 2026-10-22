"""Phase 4 NLP: contextual ranking, inflection, idioms, grammar diffing.
A fake masked LM and a fake corrector stand in for DistilBERT and T5."""

import pytest

from coach.analyzers import GrammarAnalyzer, VocabularyAnalyzer, default_analyzers
from coach.analyzers import grammar as grammar_mod
from coach.analyzers import vocabulary as vocab_mod
from coach.analyzers.vocabulary import inflect
from coach.errors import Unavailable
from coach.nlp import preprocess


class FakeScorer:
    """Pretends to be DistilBERT: fixed scores, remembers the masked text."""
    mask_token = "[MASK]"

    def __init__(self, scores):
        self.scores = scores
        self.texts = []

    def single_piece(self, word):
        return "-" not in word and " " not in word

    def score(self, text, targets):
        self.texts.append(text)
        return {t: self.scores.get(t, 0.0) for t in targets}


class FakeCorrector:
    def __init__(self, fixes):
        self.fixes = fixes
        self.seen = []

    def correct(self, sentences):
        self.seen.extend(sentences)
        return [self.fixes.get(s, s) for s in sentences]


def finding(result, text):
    return next(f for f in result.findings if f.text == text)


# ---- contextual ranking ------------------------------------------------
def test_contextual_masks_whole_span_and_ranks():
    scorer = FakeScorer({"excellent": 0.40, "great": 0.30, "fantastic": 0.05, "wonderful": 0.001})
    r = VocabularyAnalyzer(use_contextual=True, scorer=scorer).analyze(preprocess("The food was very good."))
    # the intensifier is masked together with the word: ONE mask for "very good"
    assert scorer.texts == ["The food was [MASK]."]
    assert finding(r, "very good").suggestions == ["excellent", "great", "fantastic"]


def test_contextual_drops_low_scores_but_keeps_one():
    scorer = FakeScorer({"great": 0.5, "excellent": 0.009})   # 0.009 < 2% of 0.5
    analyzer = VocabularyAnalyzer(use_contextual=True, scorer=scorer)
    doc = preprocess("It was good.")
    ranked = analyzer.contextual_suggestions(doc, doc[2], ["great", "excellent"])
    assert ranked == ["great"]
    # all scores 0 = the model has no opinion: keep every candidate, original order
    zero = VocabularyAnalyzer(use_contextual=True, scorer=FakeScorer({}))
    assert zero.contextual_suggestions(doc, doc[2], ["great", "excellent"]) == ["great", "excellent"]


def test_multi_piece_candidates_go_after_scored_ones():
    scorer = FakeScorer({"intriguing": 0.3, "fascinating": 0.2})
    analyzer = VocabularyAnalyzer(use_contextual=True, scorer=scorer)
    doc = preprocess("The talk was interesting.")
    ranked = analyzer.contextual_suggestions(doc, doc[3], ["fascinating", "thought-provoking", "intriguing"])
    assert ranked == ["intriguing", "fascinating", "thought-provoking"]


def test_contextual_falls_back_when_model_missing(monkeypatch):
    def missing():
        raise Unavailable("no model")
    monkeypatch.setattr(vocab_mod, "get_mask_scorer", missing)
    base = VocabularyAnalyzer().analyze(preprocess("The hotel was nice."))
    ctx = VocabularyAnalyzer(use_contextual=True).analyze(preprocess("The hotel was nice."))
    assert ctx.findings[0].suggestions == base.findings[0].suggestions


# ---- inflection + idioms -----------------------------------------------
def test_suggestions_match_the_learners_word_form():
    r = VocabularyAnalyzer().analyze(preprocess("We got many gifts and the rooms were bigger."))
    assert finding(r, "got").suggestions[0] == "received"
    assert finding(r, "bigger").suggestions[0] == "larger"
    assert inflect("things", "NN") == "things"       # base tags are never changed
    assert inflect("large", "JJS") == "largest"
    assert inflect("significant", "JJS") == "most significant"   # never "significantest"
    assert inflect("serious", "JJR") == "more serious"
    assert inflect("thought-provoking", "JJR") == "thought-provoking"


@pytest.mark.parametrize("text", [
    "She left the company for good last year.",      # the known spaCy misparse
    "Good morning, sir, thank you for the opportunity.",
    "Thank you, have a good day.",
    "I get up at six every day.",                    # phrasal verb
    "Hard work pays off.",
])
def test_idioms_are_not_flagged(text):
    assert VocabularyAnalyzer().analyze(preprocess(text)).findings == []


# ---- grammar -----------------------------------------------------------
def grammar(text, fixes):
    return GrammarAnalyzer(corrector=FakeCorrector(fixes)).analyze(preprocess(text))


def test_grammar_replacement_with_offsets():
    text = "She don't like coffee."
    r = grammar(text, {text: "She doesn't like coffee."})
    [f] = r.findings
    assert (f.text, f.suggestions, f.kind) == ("don't", ["doesn't"], "grammar")
    assert text[f.start:f.end] == f.text
    assert r.metrics == {"error_count": 1, "sentences": 1}


def test_grammar_deletion_and_insertion():
    r = grammar("I want to discuss about my project.", {"I want to discuss about my project.": "I want to discuss my project."})
    assert [(f.text, f.suggestions) for f in r.findings] == [("about", [])]
    t = "My mother also a doctor."
    r = grammar(t, {t: "My mother is also a doctor."})
    assert [(f.text, f.suggestions) for f in r.findings] == [("also", ["is also"])]
    t = "I like"
    r = grammar(t, {t: "I like it."})
    assert [(f.text, f.suggestions) for f in r.findings] == [("like", ["like it"])]


def test_grammar_ignores_case_punctuation_contractions_and_fillers():
    t = "umm i do not like it"
    r = grammar(t, {t: "I don't like it."})          # case, punctuation, contraction, filler
    assert r.findings == []


def test_grammar_works_sentence_by_sentence():
    text = "He go to work. It is fine."
    corrector = FakeCorrector({"He go to work.": "He goes to work."})
    r = GrammarAnalyzer(corrector=corrector).analyze(preprocess(text))
    assert corrector.seen == ["He go to work.", "It is fine."]
    assert [f.text for f in r.findings] == ["go"]


def test_default_analyzers_respects_flags_and_missing_models(monkeypatch):
    names = lambda cfg: [a.name for a in default_analyzers(cfg)]   # noqa: E731
    assert names(None) == ["vocabulary", "fluency"]
    monkeypatch.setattr(grammar_mod, "get_corrector", lambda: FakeCorrector({}))
    assert names({"analyzers": {"grammar": True}}) == ["vocabulary", "fluency", "grammar"]

    def missing():
        raise Unavailable("no model")
    monkeypatch.setattr(grammar_mod, "get_corrector", missing)
    assert names({"analyzers": {"grammar": True}}) == ["vocabulary", "fluency"]


def test_eval_forms_accept_inflections():
    from eval.eval_vocab import forms
    assert {"receive", "received", "receives", "receiving"} <= forms("receive")
