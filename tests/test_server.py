"""The HTTP API, end to end, with fake LLM/STT/TTS and a temporary database."""

import base64
import json

import pytest
from fastapi.testclient import TestClient

from coach.config import load_config
from coach.engines import Engines
from coach.personas import load_personas, personas_for_goal, resolve_voice
from coach.server import create_app
from coach.storage import Storage
from tests.fakes import FakeLLM, FakeSTT, FakeTTS

CPU = {"gpu": None, "vram_mb": 0, "ram_gb": 16.0}
PROFILE = {"name": "Asha", "goal": "placements", "level": "B1", "accent": "auto", "silence_ms": 1200}


@pytest.fixture
def make_client(tmp_path):
    def make(stt=True, tts=True, llm=None):
        cfg = load_config(hw=CPU)
        cfg["analyzers"] = {"contextual": False, "grammar": False}   # never load real NLP models in tests
        engines = Engines(cfg, llm=llm or FakeLLM("Nice to meet you. What do you study?"),
                          stt=FakeSTT() if stt else None, tts=FakeTTS() if tts else None, load=False)
        storage = Storage(tmp_path / "g.db", export_dir=tmp_path / "export")
        return TestClient(create_app(cfg, engines, storage))
    return make


def events(res):
    return [json.loads(line) for line in res.text.splitlines() if line.strip()]


def test_profile_roundtrip_and_validation(make_client):
    c = make_client()
    assert c.get("/api/profile").json()["profile"] is None
    assert c.put("/api/profile", json=PROFILE).status_code == 200
    assert c.get("/api/profile").json()["profile"]["name"] == "Asha"
    assert c.put("/api/profile", json={**PROFILE, "goal": "astronaut"}).status_code == 422
    assert c.put("/api/profile", json={**PROFILE, "name": "  "}).status_code == 422


def test_status_reports_missing_voice(make_client):
    s = make_client(stt=False, tts=False).get("/api/status").json()
    assert s["llm"]["state"] == "ready"
    assert s["stt"]["state"] == "off" and s["tts"]["state"] == "off"


def test_personas_filtered_by_goal(make_client):
    c = make_client()
    c.put("/api/profile", json=PROFILE)
    ids = {p["id"] for p in c.get("/api/personas").json()}
    assert ids == {"hr_interviewer", "buddy"}           # placements + general
    c.put("/api/profile", json={**PROFILE, "goal": "study_abroad", "destination": "uk"})
    ps = {p["id"]: p for p in c.get("/api/personas").json()}
    assert set(ps) == {"visa_officer", "admissions_interviewer", "buddy"}
    assert ps["visa_officer"]["voice"] == "british_male"   # UK destination, gender kept


def test_resolve_voice_rules():
    p = load_personas()
    rao, maya = p["hr_interviewer"], p["buddy"]
    assert resolve_voice(rao, {"accent": "american"}) == "american_male"      # explicit wins
    assert resolve_voice(maya, {"accent": "british"}) == "british_female"
    assert resolve_voice(maya, {"accent": "auto", "goal": "study_abroad", "destination": "canada"}) == "american_female"
    assert resolve_voice(rao, {"accent": "auto", "goal": "job"}) == "british_male"  # persona default
    assert resolve_voice(rao, None) == "british_male"
    assert {q.id for q in personas_for_goal(p, None)} == set(p)


def test_full_conversation_voice_and_text(make_client):
    c = make_client()
    c.put("/api/profile", json=PROFILE)
    start = c.post("/api/sessions", json={"persona_id": "hr_interviewer"}).json()
    assert start["opening"]["text"].startswith("Good morning")
    assert base64.b64decode(start["opening"]["wav"])[:4] == b"RIFF"
    sid = start["id"]

    voice = events(c.post(f"/api/sessions/{sid}/turn", files={"audio": ("t.webm", b"fake", "audio/webm")}))
    assert voice[0] == {"type": "transcript", "text": "Umm the trip was very good."}
    audio = [e for e in voice if e["type"] == "audio"]
    assert [a["text"] for a in audio] == ["Nice to meet you.", "What do you study?"]
    assert voice[-1]["type"] == "done" and "total_s" in voice[-1]["latency"]

    typed = events(c.post(f"/api/sessions/{sid}/turn", data={"text": "I I study, you know, computers."}))
    assert typed[-1]["type"] == "done"

    report = c.post(f"/api/sessions/{sid}/end").json()
    assert report["summary"]["turns"] == 2
    assert report["persona_name"] == "Mr. Rao"
    first = report["turns"][0]
    spans = {first["text"][f["start"]:f["end"]] for f in first["findings"]}
    assert {"Umm", "very good"} <= spans
    assert "stt_s" in report["latency"]
    # the session was saved and the live one is gone
    assert c.get(f"/api/reports/{report['id']}").json()["summary"]["turns"] == 2
    assert c.post(f"/api/sessions/{sid}/turn", data={"text": "hi"}).status_code == 404
    prog = c.get("/api/progress").json()
    assert len(prog["sessions"]) == 1 and prog["most_flagged"][0]["text"] == "very good"
    home = c.get("/api/home").json()
    assert home["session_count"] == 1 and home["last_session"]["id"] == report["id"]


def test_voice_turn_without_stt_gives_error_event(make_client):
    c = make_client(stt=False)
    sid = c.post("/api/sessions", json={"persona_id": "buddy"}).json()["id"]
    ev = events(c.post(f"/api/sessions/{sid}/turn", files={"audio": ("t.webm", b"x", "audio/webm")}))
    assert ev[0]["type"] == "error" and "Type" in ev[0]["message"]


def test_text_works_without_any_voice(make_client):
    c = make_client(stt=False, tts=False)
    start = c.post("/api/sessions", json={"persona_id": "buddy"}).json()
    assert start["opening"]["wav"] is None
    ev = events(c.post(f"/api/sessions/{start['id']}/turn", data={"text": "Hello there."}))
    assert "".join(e["delta"] for e in ev if e["type"] == "text").strip()


def test_end_without_turns_saves_nothing(make_client):
    c = make_client()
    sid = c.post("/api/sessions", json={"persona_id": "buddy"}).json()["id"]
    assert c.post(f"/api/sessions/{sid}/end").json() == {"empty": True}
    assert c.get("/api/progress").json()["sessions"] == []


def test_llm_offline_blocks_start_with_message(make_client):
    from coach.errors import Unavailable

    class DownLLM(FakeLLM):
        def check(self):
            raise Unavailable("Coach offline: start Ollama.")
    r = make_client(llm=DownLLM()).post("/api/sessions", json={"persona_id": "buddy"})
    assert r.status_code == 503 and "Ollama" in r.json()["detail"]


def test_abort_stops_reply_and_next_turn_still_works(make_client):
    import threading

    class SlowLLM(FakeLLM):
        """Streams forever until the turn is aborted."""
        def __init__(self):
            super().__init__()
            self.streaming = threading.Event()

        def stream_chat(self, messages):
            self.seen.append(list(messages))
            if len(self.seen) == 1:
                for i in range(10_000):
                    self.streaming.set()
                    yield f"word{i} "
                    threading.Event().wait(0.001)
            else:
                yield "Okay. "

    llm = SlowLLM()
    c = make_client(llm=llm)
    sid = c.post("/api/sessions", json={"persona_id": "buddy"}).json()["id"]
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("ev", events(
        c.post(f"/api/sessions/{sid}/turn", data={"text": "Tell me a long story."}))))
    t.start()
    llm.streaming.wait(5)
    assert c.post(f"/api/sessions/{sid}/abort").json() == {"ok": True}
    t.join(10)
    assert result["ev"][-1]["type"] == "aborted"
    ev = events(c.post(f"/api/sessions/{sid}/turn", data={"text": "Sorry, go on."}))
    assert ev[-1]["type"] == "done"
