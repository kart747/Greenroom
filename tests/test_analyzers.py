from coach.analyzers import FluencyAnalyzer, VocabularyAnalyzer
from coach.nlp import preprocess


def texts(result, kind=None):
    return [f.text for f in result.findings if kind is None or f.kind == kind]


def test_vocab_flags_basic_and_intensified_words():
    r = VocabularyAnalyzer().analyze(preprocess("The food was very good and the hotel was nice."))
    assert "very good" in texts(r, "intensified_basic")
    assert "nice" in texts(r, "basic_word")
    assert r.findings[0].suggestions, "basic words should come with suggestions"


def test_vocab_uses_lemmas():
    r = VocabularyAnalyzer().analyze(preprocess("We got many things from the market."))
    assert {"got", "things"} <= set(texts(r))


def test_vocab_ignores_wrong_pos():
    # "good" as a noun ("for good") is not the adjective we want to replace
    r = VocabularyAnalyzer().analyze(preprocess("She left the city for good."))
    assert "good" not in texts(r)


def test_vocab_clean_sentence_has_no_findings():
    r = VocabularyAnalyzer().analyze(preprocess("The lecture was fascinating and well structured."))
    assert r.findings == []


def test_fluency_fillers_and_repetition():
    r = FluencyAnalyzer().analyze(preprocess("Umm I I think, you know, it was fine."))
    assert texts(r, "filler") == ["Umm"]
    assert texts(r, "filler_phrase") == ["you know"]
    assert texts(r, "repetition") == ["I I"]
    assert r.metrics["fillers_per_100_words"] > 0


def test_fluency_kind_of_as_real_meaning_not_flagged():
    r = FluencyAnalyzer().analyze(preprocess("It is a kind of fruit."))
    assert texts(r, "filler_phrase") == []


def test_finding_offsets_match_text():
    text = "Honestly the trip was really nice."
    r = VocabularyAnalyzer().analyze(preprocess(text))
    for f in r.findings:
        assert text[f.start:f.end] == f.text
