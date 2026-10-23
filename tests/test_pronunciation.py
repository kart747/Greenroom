"""Pronunciation scoring with a fake phone dictionary and a fake recognizer
(no espeak, no wav2vec2 model)."""

import numpy as np

from coach import pronunciation as pr

DICT = {  # what espeak would say, American / British
    "en-us": {"i": ["aɪ"], "think": ["θ", "ɪ", "ŋ", "k"], "very": ["v", "ɛ", "ɹ", "i"], "car": ["k", "ɑːɹ"],
              "people": ["p", "iː", "p", "əl"], "bird": ["b", "ɜː", "d"]},
    "en-gb": {"i": ["aɪ"], "think": ["θ", "ɪ", "ŋ", "k"], "very": ["v", "ɛ", "ɹ", "i"], "car": ["k", "ɑː"],
              "people": ["p", "iː", "p", "əl"], "bird": ["b", "ɜː", "d"]},
}


def g2p(words, lang):
    return [DICT[lang][w.lower()] for w in words]


def score(text, heard):
    return pr.score_words(text, heard.split(), g2p=g2p)


def test_canon_splits_r_coloured_vowels_and_drops_marks():
    assert pr.canon(["ɑːɹ", "tʰ", "ɐ", "ɚ", "ɜː"]) == ["ɑː", "ɹ", "t", "ə", "ə", "ɹ", "ɜ", "ɹ"]


def test_perfect_reading_scores_100():
    r = score("I think", "aɪ θ ɪ ŋ k")
    assert r["score"] == 100 and all(w["status"] == "good" for w in r["words"])


def test_th_as_t_is_caught_with_a_tip():
    r = score("I think", "aɪ t ɪ ŋ k")
    think = r["words"][1]
    assert think["status"] == "close" and think["score"] == 0.75
    assert think["phones"][0] == {"expected": "θ", "heard": "t", "score": 0.0}
    assert "tongue between your teeth" in think["tip"]


def test_v_as_w_and_missing_sounds():
    very = score("very", "w ɛ ɹ i")["words"][0]
    assert very["phones"][0]["heard"] == "w" and "lower lip" in very["tip"]
    gone = score("think", "θ ɪ ŋ")["words"][0]
    assert gone["phones"][-1] == {"expected": "k", "heard": None, "score": 0.0} and "missing" in gone["tip"]


def test_accent_features_are_not_errors():
    assert score("car", "k ɑː")["score"] == 100                 # no r after the vowel (British, Indian English)
    assert score("very", "v ɛ r i")["score"] == 100             # rolled r
    very = score("very", "ʋ ɛ ɹ i")["words"][0]            # v/w as one sound: half right, not wrong
    assert very["phones"][0]["score"] == 0.5 and very["status"] == "good" and "lower lip" in very["tip"]
    assert score("people", "p iː p l")["score"] == 100          # syllabic l, no "uh"
    assert score("bird", "b ɚ d")["score"] == 100               # r-coloured vowel = "ir" in bird


def test_assess_with_fake_recognizer():
    class Rec:
        def recognize(self, audio):
            return ["aɪ", "θ", "ɪ", "ŋ", "k"]

    class Silent:
        def recognize(self, audio):
            return []
    r = pr.assess(np.zeros(16000, dtype="float32"), "I think", recognizer=Rec(), g2p=g2p)
    assert r["score"] == 100 and r["empty"] is False
    assert pr.assess(np.zeros(16000), "I think", recognizer=Silent(), g2p=g2p)["empty"] is True
