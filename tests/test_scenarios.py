"""Custom scenarios: LLM-designed roleplay characters (fake LLM)."""

import json

import pytest
from fastapi.testclient import TestClient

from coach import scenarios
from coach.config import load_config
from coach.engines import Engines
from coach.server import create_app
from coach.storage import Storage
from tests.fakes import FakeLLM, FakeTTS

PROFILE = {"name": "Asha", "goal": "job", "level": "B1", "accent": "auto", "silence_ms": 1200}


@pytest.fixture
def client(tmp_path):
    cfg = load_config(hw={"gpu": None, "vram_mb": 0, "ram_gb": 16.0})
    cfg["analyzers"] = {"contextual": False, "grammar": False}
    llm = FakeLLM()
    storage = Storage(tmp_path / "g.db", export_dir=None)
    c = TestClient(create_app(cfg, Engines(cfg, llm=llm, tts=FakeTTS(), load=False), storage))
    c.put("/api/profile", json=PROFILE)
    return c, llm, storage, cfg


def test_design_trims_and_checks_fields():
    class Messy(FakeLLM):
        def complete_json(self, messages, schema, temperature=0.3, max_tokens=300):
            self.prompt = messages[0]["content"]
            return {"name": "  Mr.   Verma " + "x" * 80, "role": "", "personality": "Strict.\n\nBut fair.",
                    "opening": "", "gender": "robot"}
    llm = Messy()
    d = scenarios.design(llm, "I want to return a faulty phone at a shop", "situation")
    assert d["name"].startswith("Mr. Verma") and len(d["name"]) == 40
    assert d["role"] == "a conversation partner" and d["opening"] == "Hello! Shall we begin?"
    assert d["personality"] == "Strict. But fair." and d["gender"] == "female"
    assert "return a faulty phone" in llm.prompt and "OTHER person" in llm.prompt


def test_job_description_prompt_and_persona():
    llm = FakeLLM()
    d = scenarios.design(llm, "Data analyst. Skills: SQL, Excel, Power BI. 0-2 years.", "job")
    prompt = llm.json_calls[-1][0][0]["content"]
    assert "job interview" in prompt and "Power BI" in prompt
    p = scenarios.to_persona(7, d)
    assert p.id == "custom_7" and p.mode == "roleplay" and p.voice == "american_female"
    # the generated character still gets the fixed roleplay rules
    assert "Do NOT correct" in p.system_prompt("B1") and "Stay in character" in p.system_prompt("B1")


def test_create_practise_report_and_hide(client):
    c, llm, storage, cfg = client
    assert c.post("/api/scenarios", json={"text": "short"}).status_code == 422
    assert c.post("/api/scenarios", json={"text": "I need two days of leave from my manager.", "kind": "poem"}).status_code == 422
    made = c.post("/api/scenarios", json={"text": "I need to ask my manager for two days of leave."}).json()
    assert made["id"].startswith("custom_") and made["name"] == "Ms. Iyer" and made["kind"] == "situation"
    assert [s["id"] for s in c.get("/api/scenarios").json()] == [made["id"]]
    # custom scenarios don't clutter the goal-based character list
    assert made["id"] not in {p["id"] for p in c.get("/api/personas").json()}

    start = c.post("/api/sessions", json={"persona_id": made["id"]}).json()
    assert start["opening"]["text"] == "Hi, you wanted to talk to me about something?"
    sid = start["id"]
    c.post(f"/api/sessions/{sid}/turn", data={"text": "I need two days off next week for my sister's wedding."})
    report = c.post(f"/api/sessions/{sid}/end").json()
    assert report["persona_name"] == "Ms. Iyer"

    c.delete(f"/api/scenarios/{made['id'].split('_')[1]}")
    assert c.get("/api/scenarios").json() == []
    assert c.get(f"/api/reports/{report['id']}").json()["persona_name"] == "Ms. Iyer"   # old reports keep the name

    # a restarted app loads saved scenarios again
    app2 = create_app(cfg, Engines(cfg, llm=llm, load=False), storage)
    assert TestClient(app2).get(f"/api/reports/{report['id']}").json()["persona_name"] == "Ms. Iyer"
