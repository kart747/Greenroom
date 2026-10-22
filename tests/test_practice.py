"""Phase 5: exercises from templates, SM-2 scheduling, review cards via the API."""

import pytest

from coach import srs
from coach.exercises import build_exercises, collect_issues, make_exercise
from coach.personas import load_personas
from coach.session import Session
from tests.fakes import FakeLLM


def record_for(*turns):
    s = Session(load_personas()["buddy"], FakeLLM(), level="B1")
    for t in turns:
        list(s.user_turn(t))
    return s.finish()


# ---- SM-2 ---------------------------------------------------------------
def test_sm2_good_reviews_grow_1_6_then_times_ease():
    ease, interval, reps = 2.5, 0, 0
    days = []
    for _ in range(4):
        ease, interval, reps, due = srs.review(ease, interval, reps, 4, now=0)
        days.append(interval)
        assert due == interval * srs.DAY
    assert days[:2] == [1, 6]
    assert days[2] == round(6 * 2.5) and days[3] == round(15 * 2.5)   # "good" keeps ease at 2.5
    assert ease == 2.5


def test_sm2_again_resets_and_returns_soon():
    ease, interval, reps, due = srs.review(2.5, 15, 3, 0, now=1000)
    assert (interval, reps) == (0, 0)
    assert due == 1000 + srs.RELEARN_DELAY
    assert ease == pytest.approx(1.7)


def test_sm2_easy_raises_hard_lowers_ease_with_floor():
    assert srs.review(2.5, 0, 0, 5, now=0)[0] == 2.6
    assert srs.review(2.5, 0, 0, 3, now=0)[0] == pytest.approx(2.36)
    assert srs.review(1.3, 0, 0, 0, now=0)[0] == 1.3        # never below 1.3
    with pytest.raises(ValueError):
        srs.review(2.5, 0, 0, 2)


# ---- exercises --------------------------------------------------------------
def test_intensified_basic_template_matches_spec_example():
    ex = make_exercise({"kind": "intensified_basic", "text": "very good",
                        "suggestions": ["excellent", "great"], "sentence": "The food was very good."})
    assert ex["prompt"] == "Describe your favourite food in two sentences without using “very” or “good”."
    assert "excellent" in ex["tip"]


def test_three_exercises_prefer_different_kinds():
    rec = record_for("Umm the food was very good, very good.", "I I think the hotel was nice, you know.")
    issues = collect_issues(rec)
    assert issues[0]["text"].lower() == "very good" and issues[0]["count"] == 2
    exercises = build_exercises(rec)
    assert len(exercises) == 3
    assert len({e["kind"] for e in exercises}) == 3          # variety first
    for e in exercises:
        assert e["title"] and e["prompt"]


def test_repetition_and_grammar_templates():
    rep = make_exercise({"kind": "repetition", "text": "I I", "suggestions": [],
                         "sentence": "I I worked on a project."})
    assert "“I worked on a project.”" in rep["prompt"]
    gram = make_exercise({"kind": "grammar", "text": "don't", "suggestions": ["doesn't"],
                          "start": 4, "end": 9, "sentence": "She don't like tea."})
    assert "“She doesn't like tea.”" in gram["prompt"]
    deletion = make_exercise({"kind": "grammar", "text": "about", "suggestions": [],
                              "start": 18, "end": 23, "sentence": "I want to discuss about it."})
    assert "“I want to discuss it.”" in deletion["prompt"]


def test_no_findings_no_exercises():
    assert build_exercises(record_for("The lecture was fascinating.")) == []


# ---- API: cards are created at session end and can be graded ------------------
def test_session_end_creates_cards_and_grading_reschedules(tmp_path):
    from fastapi.testclient import TestClient

    from coach.config import load_config
    from coach.engines import Engines
    from coach.server import create_app
    from coach.storage import Storage
    cfg = load_config(hw={"gpu": None, "vram_mb": 0, "ram_gb": 16.0})
    cfg["analyzers"] = {"contextual": False, "grammar": False}
    c = TestClient(create_app(cfg, Engines(cfg, llm=FakeLLM(), load=False), Storage(tmp_path / "g.db", export_dir=None)))
    sid = c.post("/api/sessions", json={"persona_id": "buddy"}).json()["id"]
    c.post(f"/api/sessions/{sid}/turn", data={"text": "Umm the trip was very good and the hotel was nice."})
    report = c.post(f"/api/sessions/{sid}/end").json()
    assert len(report["exercises"]) == 3

    cards = c.get("/api/review/due").json()["cards"]
    assert {card["front"] for card in cards} == {"very good", "nice"}   # vocabulary only, no "umm"
    card = next(cd for cd in cards if cd["front"] == "nice")
    assert card["example"].startswith("Umm the trip") and card["suggestions"]

    r = c.post(f"/api/review/{card['id']}", json={"grade": 4}).json()
    assert r["interval"] == 1 and r["reps"] == 1
    assert {cd["front"] for cd in c.get("/api/review/due").json()["cards"]} == {"very good"}
    assert c.get("/api/home").json()["due_cards"] == 1
    assert c.post(f"/api/review/{card['id']}", json={"grade": 2}).status_code == 422
    assert c.post("/api/review/999", json={"grade": 4}).status_code == 404
