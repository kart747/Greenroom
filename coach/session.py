"""A coaching session.

Each learner turn goes two ways at once:
  1. to the LLM coach (in character) -> streamed reply
  2. to the analyzers, in a background thread -> findings for the end report
The analyzers are never in the reply path, so they never add latency.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .analyzers import Analyzer, AnalyzerResult, default_analyzers
from .llm import ChatModel
from .nlp import preprocess
from .personas import Persona


@dataclass
class Turn:
    text: str
    results: list[AnalyzerResult] = field(default_factory=list)


class Session:
    def __init__(self, persona: Persona, llm: ChatModel, level: str = "B1",
                 analyzers: list[Analyzer] | None = None):
        self.persona = persona
        self.level = level
        self.llm = llm
        self.analyzers = analyzers if analyzers is not None else default_analyzers()
        self.messages = [{"role": "system", "content": persona.system_prompt(level)},
                         {"role": "assistant", "content": persona.opening}]
        self.turns: list[Turn] = []
        self._pending: list[tuple[Turn, Future]] = []
        self._pool = ThreadPoolExecutor(max_workers=1)  # spaCy isn't thread-safe; 1 worker keeps order
        self.started = time.time()

    @property
    def opening(self) -> str:
        return self.persona.opening

    def user_turn(self, text: str) -> Iterator[str]:
        """Record a learner turn, start background analysis, stream the coach's reply."""
        turn = Turn(text)
        self.turns.append(turn)
        self._pending.append((turn, self._pool.submit(self._analyze, text)))
        self.messages.append({"role": "user", "content": text})

        reply = []
        for chunk in self.llm.stream_chat(self.messages):
            reply.append(chunk)
            yield chunk
        self.messages.append({"role": "assistant", "content": "".join(reply)})

    def _analyze(self, text: str) -> list[AnalyzerResult]:
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
            "level": self.level,
            "started": self.started,
            "duration_s": round(time.time() - self.started, 1),
            "turns": [{"text": t.text, "results": [r.to_dict() for r in t.results]}
                      for t in self.turns],
            "transcript": self.messages[1:],
        }

    @staticmethod
    def save(record: dict, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"session_{int(record['started'])}_{record['persona']}.json"
        path.write_text(json.dumps(record, indent=2))
        return path
