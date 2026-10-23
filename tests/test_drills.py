"""Speaking drills: grammar cards, answer checking, and the check endpoint."""

import pytest
from fastapi.testclient import TestClient

from coach import drills
from coach.config import load_config
from coach.engines import Engines
from coach.server import _make_cards, create_app
from coach.storage import Storage
from tests.fakes import FakeLLM, FakeSTT

SENT = "I am having two years experience in Java."
FINDING = {"analyzer": "grammar", "kind": "grammar", "text": "am having", "start": 2, "end": 11,
           "message": 'Try "have" instead of "am having".', "suggestions": ["have"]}


def test_grammar_card_from_finding():
    key, data = drills.grammar_card(FINDING, SENT)
    assert key == "am having -> have"
    assert data["corrected"] == "I have two years experience in Java."
    deletion = {**FINDING, "text": "about", "start": 18, "end": 23, "suggestions": []}
    key2, data2 = drills.grammar_card(deletion, "I want to discuss about my project.")
    assert key2 == "about -> (leave out)" and data2["corrected"] == "I want to discuss my project."


@pytest.mark.parametrize("answer, ok", [
    ("I have two years experience in Java", True),
    ("i HAVE two years of experience.", True),           # case and punctuation don't matter
    ("I am having two years experience", False),         # the mistake is still there
    ("two years experience", False),                     # the fix is missing
])
def test_check_grammar(answer, ok):
    _, card = drills.grammar_card(FINDING, SENT)
    assert drills.check(card, answer)["correct"] is ok


def test_one_word_mistakes_are_checked_in_context():
    sent = "There is many students in my class."
    _, card = drills.grammar_card({**FINDING, "text": "is", "start": 6, "end": 8, "suggestions": ["are"]}, sent)
    assert card["context"] == [["there"], ["many"]]
    assert drills.check(card, "There are many students in my class, and it is fun")["correct"]   # "is" elsewhere is fine
    assert not drills.check(card, "There is many students")["correct"]


def test_check_deletion_and_vocabulary():
    _, card = drills.grammar_card({**FINDING, "text": "about", "start": 18, "end": 23, "suggestions": []},
                                  "I want to discuss about my project.")
    assert drills.check(card, "I want to discuss my project")["correct"]
    assert not drills.check(card, "I want to discuss about my project")["correct"]
    vocab = {"kind": "basic_word", "front": "got", "suggestions": ["received", "obtained"]}
    assert drills.check(vocab, "I receive a lot of gifts")["correct"]      # any form of a suggestion
    r = drills.check(vocab, "I got a lot of gifts")
    assert not r["correct"] and "received" in r["expected"]


def test_cards_from_record_and_check_endpoint(tmp_path):
    cfg = load_config(hw={"gpu": None, "vram_mb": 0, "ram_gb": 16.0})
    cfg["analyzers"] = {"contextual": False, "grammar": False}
    storage = Storage(tmp_path / "g.db", export_dir=None)
    record = {"turns": [{"text": SENT, "results": [
        {"analyzer": "grammar", "findings": [FINDING], "metrics": {}},
        {"analyzer": "vocabulary", "findings": [], "metrics": {}}]}]}
    _make_cards(storage, record)
    _make_cards(storage, record)                                      # same mistake again: one card
    [card] = storage.due_cards()
    assert card["front"] == "am having -> have" and card["kind"] == "grammar"

    stt = FakeSTT("I have two years experience in Java.")
    c = TestClient(create_app(cfg, Engines(cfg, llm=FakeLLM(), stt=stt, load=False), storage))
    by_text = c.post(f"/api/review/{card['id']}/check", data={"text": "I am having two years experience"}).json()
    assert by_text["correct"] is False and by_text["expected"] == "I have two years experience in Java."
    by_voice = c.post(f"/api/review/{card['id']}/check", files={"audio": ("a.webm", b"x", "audio/webm")}).json()
    assert by_voice == {"heard": "I have two years experience in Java.", "correct": True,
                        "expected": "I have two years experience in Java.", "why": 'Right: “have”, not “am having”.'}
    assert c.post(f"/api/review/{card['id']}/check", data={"text": " "}).json()["correct"] is False
    assert c.post("/api/review/999/check", data={"text": "x"}).status_code == 404
