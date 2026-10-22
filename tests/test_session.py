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


def test_analysis_waits_until_the_reply_is_done():
    """BERT/T5/Kokoro share the CPU, so analysis must not run during a reply."""
    import time

    from coach.analyzers.base import Analyzer, AnalyzerResult

    events = []

    class SlowLLM(FakeLLM):
        def stream_chat(self, messages):
            events.append("reply start")
            for word in ["Nice", " to", " meet", " you."]:
                time.sleep(0.05)
                yield word
            events.append("reply end")

    class Probe(Analyzer):
        name = "probe"

        def analyze(self, doc):
            events.append("analysis")
            return AnalyzerResult(self.name)

    s = Session(load_personas()["buddy"], SlowLLM(), analyzers=[Probe()])
    with s.replying():                      # e.g. the voice pipeline still synthesizing
        list(s.user_turn("Hello there."))
        time.sleep(0.1)
        assert "analysis" not in events     # still waiting: a reply section is open
    s.finish()
    assert events == ["reply start", "reply end", "analysis"]
