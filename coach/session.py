"""A coaching session.

Each learner turn goes two ways:
  1. to the LLM coach (in character) -> streamed reply
  2. to the analyzers, in a background thread -> findings for the end report
The analyzers are never in the reply path, so they never add latency.

They also don't COMPETE with the reply: BERT and T5 run on the CPU, and so
does Kokoro. Measured on the dev laptop, running them at the same moment
pushed the first sentence's speech synthesis from ~0.9 s to ~2.4 s. So the
analysis of a turn waits until the coach's reply has been produced, and then
runs while the coach is speaking, when the CPU has nothing else to do.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .analyzers import Analyzer, AnalyzerResult, Finding, default_analyzers
from .llm import ChatModel
from .nlp import preprocess
from .personas import Persona

# Safety net: if a reply is never finished (a stuck client), analysis starts anyway.
MAX_WAIT_FOR_REPLY_S = 30


@dataclass
class Turn:
    text: str
    results: list[AnalyzerResult] = field(default_factory=list)


class Session:
    def __init__(self, persona: Persona, llm: ChatModel, level: str = "B1",
                 analyzers: list[Analyzer] | None = None,
                 system_prompt: str | None = None, opening: str | None = None):
        """system_prompt / opening override the persona's defaults; the tutor
        passes a prompt that includes what it remembers about the learner."""
        self.persona = persona
        self.level = level
        self.llm = llm
        self.analyzers = analyzers if analyzers is not None else default_analyzers()
        self._opening = opening if opening is not None else persona.opening
        self.messages = [{"role": "system", "content": system_prompt or persona.system_prompt(level)},
                         {"role": "assistant", "content": self._opening}]
        self.turns: list[Turn] = []
        self.prior_duration_s = 0.0         # time spent in earlier visits (resumed chats)
        self.latencies: list[dict] = []
        self._pending: list[tuple[Turn, Future]] = []
        self._pool = ThreadPoolExecutor(max_workers=1)  # spaCy isn't thread-safe; 1 worker keeps order
        self._replying = 0                  # how many reply sections are open right now
        self._reply_lock = threading.Lock()
        self._quiet = threading.Event()     # set = no reply being produced: analysis may run
        self._quiet.set()
        self.started = time.time()
        self._visit_started = time.time()   # this visit; duration adds up across resumes

    @property
    def opening(self) -> str:
        return self._opening

    @property
    def mode(self) -> str:
        return self.persona.mode

    @classmethod
    def resume(cls, record: dict, persona: Persona, llm: ChatModel,
               analyzers: list[Analyzer] | None = None, system_prompt: str | None = None) -> Session:
        """Continue a saved conversation (Tutor chats, like ChatGPT's history).
        The system prompt is rebuilt fresh, so the tutor uses what it knows
        about the learner NOW; the chat history and earlier feedback come back."""
        s = cls(persona, llm, level=record["level"], analyzers=analyzers, system_prompt=system_prompt)
        s.messages = [s.messages[0], *record["transcript"]]
        s._opening = record["transcript"][0]["content"] if record["transcript"] else s._opening
        s.turns = [Turn(t["text"], [_result_from_dict(r) for r in t["results"]]) for t in record["turns"]]
        s.latencies = list(record.get("latency", {}).get("turns", []))
        s.started = record["started"]
        s.prior_duration_s = record.get("duration_s", 0.0)
        return s

    def ready_feedback(self, since: int = 0) -> list[dict]:
        """Findings for turns whose analysis has FINISHED, from turn `since` on.
        Never waits: the Tutor screen polls this to put feedback on the
        learner's messages as soon as it exists."""
        done = {id(t): f for t, f in self._pending if f.done()}
        out = []
        for i, turn in enumerate(self.turns[since:], start=since):
            if id(turn) in done:
                results = done[id(turn)].result()
            elif turn.results or not any(t is turn for t, _ in self._pending):
                results = turn.results          # restored from a saved chat
            else:
                continue                        # still being analyzed
            findings = sorted((f.to_dict() for r in results for f in r.findings),
                              key=lambda f: (f["start"], -f["end"]))
            out.append({"index": i, "text": turn.text, "findings": findings})
        return out

    @contextmanager
    def replying(self):
        """Mark "the coach is producing a reply" (LLM + speech synthesis).
        Background analysis waits while any reply section is open. Sections can
        nest: the voice pipeline opens one around STT + LLM + TTS, and
        user_turn opens its own so text-only use is covered too."""
        with self._reply_lock:
            self._replying += 1
            self._quiet.clear()
        try:
            yield
        finally:
            with self._reply_lock:
                self._replying -= 1
                if self._replying == 0:
                    self._quiet.set()

    def user_turn(self, text: str) -> Iterator[str]:
        """Record a learner turn, queue its analysis, stream the coach's reply."""
        with self.replying():   # opened BEFORE queueing, so the analysis really waits
            turn = Turn(text)
            self.turns.append(turn)
            self._pending.append((turn, self._pool.submit(self._analyze, text)))
            self.messages.append({"role": "user", "content": text})

            reply = []
            try:
                for chunk in self.llm.stream_chat(self.messages):
                    reply.append(chunk)
                    yield chunk
            finally:
                # Barge-in closes this generator early (the learner started talking).
                # Keep the part of the reply that was already produced, so the
                # coach's memory of the conversation matches what the learner heard.
                if reply:
                    self.messages.append({"role": "assistant", "content": "".join(reply)})

    def add_latency(self, timings: dict) -> None:
        """Store one turn's timings (from the voice pipeline) for the session record."""
        self.latencies.append(timings)

    def _analyze(self, text: str) -> list[AnalyzerResult]:
        self._quiet.wait(timeout=MAX_WAIT_FOR_REPLY_S)   # let the reply have the CPU first
        doc = preprocess(text)
        return [a.analyze(doc) for a in self.analyzers]

    def finish(self) -> dict:
        """Wait for any pending analysis and return the full session record."""
        for turn, fut in self._pending:
            turn.results = fut.result()
        self._pending.clear()
        self._pool.shutdown(wait=True)
        return {
            "persona": self.persona.id,
            "mode": self.mode,
            "level": self.level,
            "started": self.started,
            "duration_s": round(self.prior_duration_s + time.time() - self._visit_started, 1),
            "turns": [{"text": t.text, "results": [r.to_dict() for r in t.results]}
                      for t in self.turns],
            "transcript": self.messages[1:],
            "latency": {"turns": self.latencies, "average": average_latency(self.latencies)},
        }

    @staticmethod
    def save(record: dict, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"session_{int(record['started'])}_{record['persona']}.json"
        path.write_text(json.dumps(record, indent=2))
        return path


def _result_from_dict(d: dict) -> AnalyzerResult:
    """Inverse of AnalyzerResult.to_dict(), for resumed chats."""
    return AnalyzerResult(d["analyzer"], [Finding(**f) for f in d["findings"]], d["metrics"])


def average_latency(turns: list[dict]) -> dict[str, float]:
    """Mean of each numeric timing, over the turns that have it (a typed turn has
    no stt_s; a turn without voice output has no tts_first_audio_s)."""
    keys = {k for t in turns for k, v in t.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {k: round(sum(t[k] for t in turns if k in t) / sum(1 for t in turns if k in t), 3)
            for k in sorted(keys)}
