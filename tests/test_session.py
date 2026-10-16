from coach.personas import load_personas
from coach.report import render_text, summarize
from coach.session import Session


class FakeLLM:
    """Stands in for Ollama so tests run offline and instantly."""

    def __init__(self):
        self.seen = []

    def stream_chat(self, messages):
        self.seen.append(list(messages))
        yield "That sounds "
        yield "great! Tell me more?"


def test_personas_load_and_build_prompts():
    personas = load_personas()
    assert {"buddy", "hr_interviewer", "visa_officer"} <= set(personas)
    prompt = personas["buddy"].system_prompt("A2")
    assert "Maya" in prompt and "A2" in prompt and "Stay in character" in prompt


def test_session_streams_reply_and_analyzes_in_background():
    llm = FakeLLM()
    s = Session(load_personas()["buddy"], llm, level="B1")
    reply = "".join(s.user_turn("Umm the movie was very good, very good."))
    assert reply == "That sounds great! Tell me more?"
    # the coach sees system prompt + opening + user turn
    assert [m["role"] for m in llm.seen[0]] == ["system", "assistant", "user"]

    record = s.finish()
    summary = summarize(record)
    vocab = {i["text"]: i["count"] for i in summary["top_issues"]["vocabulary"]}
    assert vocab["very good"] == 2
    assert "SESSION FEEDBACK" in render_text(summary)
