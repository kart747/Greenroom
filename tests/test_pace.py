"""Speaking speed and pauses from word timings (no Whisper needed)."""

from coach.analyzers import PaceAnalyzer
from coach.nlp import preprocess
from coach.personas import load_personas
from coach.report import summarize
from coach.session import Session
from coach.voice import VoicePipeline
from tests.fakes import FakeLLM, FakeSTT


def timed(text, gaps):
    """Word timings: 0.4 s per word, with the given gap before each word."""
    words, t = [], 0.0
    for w, g in zip(text.split(), gaps):
        t += g
        words.append({"word": w, "start": round(t, 2), "end": round(t + 0.4, 2), "prob": 0.9})
        t += 0.4
    return words


def test_speed_pauses_and_long_pause_finding():
    text = "I worked on a big, important project."
    words = timed(text, [0, 0.05, 0.3, 0.05, 1.8, 0.05, 0.05])
    r = PaceAnalyzer().analyze(preprocess(text, words))
    span = words[-1]["end"] - words[0]["start"]            # 5.05 s for 7 words
    assert r.metrics["words_per_minute"] == round(7 / span * 60, 1)
    assert r.metrics["longest_pause_s"] == 1.8
    assert r.metrics["mean_pause_s"] == round((0.3 + 1.8) / 2, 2)
    [f] = r.findings
    assert (f.kind, f.text) == ("long_pause", "big")       # placed on the word after the silence
    assert text[f.start:f.end] == "big" and "1.8 s" in f.message


def test_typed_turns_have_no_pace():
    r = PaceAnalyzer().analyze(preprocess("I typed this."))
    assert r.metrics == {} and r.findings == []


def test_offsets_survive_punctuation_and_repeats():
    text = "Uh, so, I I think so."
    words = timed(text, [0, 0.1, 0.1, 0.1, 2.0, 0.1])
    [f] = PaceAnalyzer().analyze(preprocess(text, words)).findings
    assert text[f.start:f.end] == "think" and f.start == text.index("think")


def test_voice_turn_timings_reach_the_report_and_typed_turns_dont_dilute():
    session = Session(load_personas()["buddy"], FakeLLM())
    pipe = VoicePipeline(session, FakeSTT("Well I think the trip was lovely."), None)
    list(pipe.run_turn(audio=b"x"))
    list(pipe.run_turn(text="And I typed this one."))
    summary = summarize(session.finish())
    pace = summary["averages"]["pace"]
    # only the spoken turn has a speed: the typed turn must not halve it
    assert pace["words_per_minute"] > 60 and pace["longest_pause_s"] == 2.0
    assert summary["top_issues"]["pace"][0]["text"] == "the"
