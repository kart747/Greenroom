"""Tutor mode: prompt + memory, in-chat feedback (tutor only), hint / say it
better / define, resuming chats, the sidebar list, memory management, and
saving open chats on shutdown. Fake LLM/TTS, temporary database."""

import json

import pytest
from fastapi.testclient import TestClient

from coach import memory
from coach.config import load_config
from coach.engines import Engines
from coach.personas import load_personas
from coach.server import create_app
from coach.session import Session
from coach.storage import Storage
from coach.text_stream import speakable
from tests.fakes import FakeLLM, FakeTTS

CPU = {"gpu": None, "vram_mb": 0, "ram_gb": 16.0}
PROFILE = {"name": "Asha", "goal": "placements", "level": "B1", "accent": "auto", "silence_ms": 1200}


@pytest.fixture
def env(tmp_path):
    cfg = load_config(hw=CPU)
    cfg["analyzers"] = {"contextual": False, "grammar": False}
    llm = FakeLLM("Good question! Say **I have two years of experience**. What do you do?",
                  facts=["Final-year computer science student", "Has a TCS interview next week"])
    storage = Storage(tmp_path / "g.db", export_dir=None)
    app = create_app(cfg, Engines(cfg, llm=llm, tts=FakeTTS(), load=False), storage)
    c = TestClient(app)
    c.put("/api/profile", json=PROFILE)
    return c, llm, storage, app


def events(res):
    return [json.loads(line) for line in res.text.splitlines() if line.strip()]


def wait_feedback(c, sid, n, since=0):
    """Poll like the UI does until n turns have feedback (analysis is async)."""
    import time
    for _ in range(100):
        turns = c.get(f"/api/sessions/{sid}/feedback", params={"since": since}).json()["turns"]
        if len(turns) >= n:
            return turns
        time.sleep(0.05)
    raise AssertionError("feedback never arrived")


# ---- prompts + memory ----------------------------------------------------------
def test_tutor_prompt_has_rules_memory_and_correction_style(tmp_path):
    tutor = load_personas()["tutor"]
    p = tutor.system_prompt("B1", "About the learner: likes cricket", "thorough")
    assert "at most three" in p and "likes cricket" in p and "Stay in character" not in p
    assert "unless they ask" in tutor.system_prompt("B1", corrections="off")
    # roleplay characters never get the learner's memory
    rao = load_personas()["hr_interviewer"].system_prompt("B1", "SECRET")
    assert "SECRET" not in rao and "Do NOT correct" in rao


def test_learner_context_from_memory_and_mistakes(tmp_path):
    storage = Storage(tmp_path / "g.db", export_dir=None)
    storage.add_memories(["Studies in Chennai"])
    summary = {"counts": {"vocabulary": {"very good": 3, "nice": 1}},
               "top_issues": {"grammar": [{"text": "am having", "count": 2, "suggestions": ["have"]},
                                          {"text": "was", "count": 5, "suggestions": ["were"]}]}}
    rec = {"started": 1.0, "persona": "tutor", "level": "B1", "mode": "tutor", "turns": [], "transcript": []}
    storage.save_session(rec, summary)
    ctx = memory.learner_context(PROFILE, storage)
    assert "Asha" in ctx and "placements" in ctx and "Studies in Chennai" in ctx
    assert '"very good" (3x)' in ctx and '"nice"' not in ctx          # only repeated words
    assert '"am having" -> "have"' in ctx and '"was"' not in ctx       # one-word slips are too vague


def test_remember_extracts_new_facts_only(tmp_path):
    storage = Storage(tmp_path / "g.db", export_dir=None)
    storage.add_memories(["Final-year computer science student"])
    llm = FakeLLM(facts=["Final-year computer science student", "Has a TCS interview next week"])
    rec = {"turns": [{"text": "I have an interview at TCS next week."}]}
    assert memory.remember(llm, storage, rec) == ["Has a TCS interview next week"]
    assert [m["text"] for m in storage.list_memories()] == ["Final-year computer science student",
                                                             "Has a TCS interview next week"]
    assert memory.remember(llm, storage, {"turns": []}) == []        # nothing said, no LLM call


def test_memory_keeps_newest_20(tmp_path):
    storage = Storage(tmp_path / "g.db", export_dir=None)
    storage.add_memories([f"fact {i}" for i in range(25)])
    facts = [m["text"] for m in storage.list_memories()]
    assert len(facts) == 20 and facts[0] == "fact 5" and facts[-1] == "fact 24"


def test_speakable_strips_markdown():
    assert speakable('You said "goed" -> **went**.\n- I went home.') == 'You said "goed". Better: went.\nI went home.'


# ---- the tutor chat end to end ---------------------------------------------------
def test_tutor_chat_feedback_tools_end_and_resume(env):
    c, llm, storage, _ = env
    c.put("/api/profile", json={**PROFILE, "chat_feedback": "tutor_only"})
    start = c.post("/api/sessions", json={"mode": "tutor"}).json()
    assert start["mode"] == "tutor" and start["opening"]["text"].startswith("Hi Asha!")
    sid = start["id"]
    ev = events(c.post(f"/api/sessions/{sid}/turn", data={"text": "I am having two years experience, it was very good."}))
    assert ev[-1]["type"] == "done"
    assert "About the learner" in llm.seen[0][0]["content"]        # memory reached the tutor

    turns = wait_feedback(c, sid, 1)
    assert turns[0]["index"] == 0 and "very good" in {f["text"] for f in turns[0]["findings"]}
    assert wait_feedback(c, sid, 0, since=1) == []                 # nothing newer yet

    hint = c.post(f"/api/sessions/{sid}/hint").json()["suggestions"]
    assert len(hint) == 3
    better = c.post(f"/api/sessions/{sid}/better", json={"text": "I am having two years experience"}).json()
    assert better["better"] == "I have two years of experience."
    d = c.post("/api/define", json={"word": "dreadful", "sentence": "The traffic was dreadful."}).json()
    assert d["word"] == "dreadful" and d["part_of_speech"] == "adjective"
    assert c.post("/api/speak", json={"text": "**Hello** there", "session_id": sid}).json()["wav"]

    report = c.post(f"/api/sessions/{sid}/end").json()
    db_id = report["id"]
    convs = c.get("/api/conversations").json()
    assert convs[0]["id"] == db_id and convs[0]["mode"] == "tutor"
    assert convs[0]["title"].startswith("I am having two years experience")

    # resume: history and earlier feedback come back; the chat continues in the same row
    again = c.post("/api/sessions", json={"resume_id": db_id}).json()
    assert again["opening"] is None and [m["role"] for m in again["history"]] == ["assistant", "user", "assistant"]
    assert again["feedback"][0]["index"] == 0
    sid2 = again["id"]
    events(c.post(f"/api/sessions/{sid2}/turn", data={"text": "Thank you, that helps."}))
    assert wait_feedback(c, sid2, 2)[1]["index"] == 1
    report2 = c.post(f"/api/sessions/{sid2}/end").json()
    assert report2["id"] == db_id and report2["summary"]["turns"] == 2
    assert len(c.get("/api/conversations").json()) == 1           # updated, not duplicated


def test_in_chat_feedback_follows_the_learners_choice(env):
    """The learner chooses: after the session / only with Sam / gentle / all.
    Feedback is a marker on the learner's own message; characters never say it."""
    c, *_ = env
    text = "Umm the food was very good and the hotel was nice."
    sids = {}
    for mode, body in (("tutor", {"mode": "tutor"}), ("roleplay", {"persona_id": "hr_interviewer"})):
        sids[mode] = c.post("/api/sessions", json=body).json()["id"]
        events(c.post(f"/api/sessions/{sids[mode]}/turn", data={"text": text}))
    c.put("/api/profile", json={**PROFILE, "chat_feedback": "all"})
    for sid in sids.values():
        wait_feedback(c, sid, 1)                     # analysis finished for both chats

    def markers(policy, mode):
        c.put("/api/profile", json={**PROFILE, "chat_feedback": policy})
        r = c.get(f"/api/sessions/{sids[mode]}/feedback").json()
        assert r["policy"] == policy
        return [[f["text"] for f in t["findings"]] for t in r["turns"]]

    everything = markers("all", "tutor")
    assert {"Umm", "very good", "nice"} <= set(everything[0])
    assert markers("all", "roleplay") == everything
    assert markers("after", "tutor") == [] and markers("after", "roleplay") == []
    assert markers("tutor_only", "tutor") == everything and markers("tutor_only", "roleplay") == []
    # gentle: only the most important finding; a weak phrase matters more than "Umm"
    assert markers("gentle", "tutor") == [["very good"]] == markers("gentle", "roleplay")

    # "Say it better" is a correction: in a roleplay only when feedback is on for every chat
    c.put("/api/profile", json={**PROFILE, "chat_feedback": "tutor_only"})
    assert c.post(f"/api/sessions/{sids['roleplay']}/better", json={"text": "x"}).status_code == 409
    assert c.post(f"/api/sessions/{sids['tutor']}/better", json={"text": "x"}).status_code == 200
    c.put("/api/profile", json={**PROFILE, "chat_feedback": "gentle"})
    assert c.post(f"/api/sessions/{sids['roleplay']}/better", json={"text": "x"}).status_code == 200
    assert len(c.post(f"/api/sessions/{sids['roleplay']}/hint").json()["suggestions"]) == 3   # hints always
    assert c.put("/api/profile", json={**PROFILE, "chat_feedback": "loud"}).status_code == 422

    # roleplays still end with a report and can't be resumed
    db_id = c.post(f"/api/sessions/{sids['roleplay']}/end").json()["id"]
    assert c.post("/api/sessions", json={"resume_id": db_id}).status_code == 409


def test_memory_endpoints_and_learning_after_chat(env):
    c, llm, storage, _ = env
    sid = c.post("/api/sessions", json={"mode": "tutor"}).json()["id"]
    events(c.post(f"/api/sessions/{sid}/turn", data={"text": "I have an interview at TCS next week."}))
    c.post(f"/api/sessions/{sid}/end")
    import time
    for _ in range(100):                      # remember() runs in a background thread
        facts = c.get("/api/memory").json()["facts"]
        if len(facts) == 2:
            break
        time.sleep(0.05)
    assert [f["text"] for f in facts] == ["Final-year computer science student", "Has a TCS interview next week"]
    left = c.delete(f"/api/memory/{facts[0]['id']}").json()["facts"]
    assert [f["text"] for f in left] == ["Has a TCS interview next week"]
    assert c.delete("/api/memory").json() == {"facts": []}


def test_tutor_corrections_setting_validated_and_used(env):
    c, llm, *_ = env
    assert c.put("/api/profile", json={**PROFILE, "tutor_corrections": "loud"}).status_code == 422
    c.put("/api/profile", json={**PROFILE, "tutor_corrections": "thorough"})
    sid = c.post("/api/sessions", json={"mode": "tutor"}).json()["id"]
    events(c.post(f"/api/sessions/{sid}/turn", data={"text": "Hi"}))
    assert "at most three" in llm.seen[-1][0]["content"]


def test_open_chats_are_saved_on_shutdown(env):
    c, _, storage, app = env
    with TestClient(app) as client:            # entering/leaving runs the app's lifespan
        sid = client.post("/api/sessions", json={"mode": "tutor"}).json()["id"]
        events(client.post(f"/api/sessions/{sid}/turn", data={"text": "Let's talk about cricket."}))
    convs = storage.list_conversations()
    assert len(convs) == 1 and convs[0]["mode"] == "tutor"


def test_resume_rebuilds_session_state():
    persona = load_personas()["tutor"]
    s = Session(persona, FakeLLM(), level="B1")
    list(s.user_turn("I goed to school."))
    rec = s.finish()
    r = Session.resume(rec, persona, FakeLLM(), system_prompt="NEW PROMPT")
    assert r.messages[0]["content"] == "NEW PROMPT" and r.messages[1:] == rec["transcript"]
    assert r.turns[0].text == "I goed to school." and r.ready_feedback()[0]["index"] == 0
    list(r.user_turn("Sorry, I went."))
    rec2 = r.finish()
    assert [t["text"] for t in rec2["turns"]] == ["I goed to school.", "Sorry, I went."]
    assert rec2["started"] == rec["started"] and rec2["mode"] == "tutor"
