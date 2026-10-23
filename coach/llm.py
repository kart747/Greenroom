"""Local LLM client (Ollama).

Two kinds of calls:
  stream_chat()    the conversation itself; streamed token by token so the
                   voice pipeline can speak the first sentence early
  complete_json()  small helper tasks (hints, "say it better", word meanings,
                   remembering facts). Ollama's structured output forces the
                   answer into a JSON schema, so a 4B model can't ramble or
                   break the format the app parses.
"""

from __future__ import annotations

import json
from typing import Iterator, Protocol

import requests

from .errors import Unavailable


# Ollama unloads an idle model after 5 minutes, and reloading it takes
# about 15 s on this laptop. A learner reading the report for a few minutes
# should not pay that again, so keep it loaded for 30 minutes.
KEEP_ALIVE = "30m"


class ChatModel(Protocol):
    def stream_chat(self, messages: list[dict]) -> Iterator[str]: ...
    def complete_json(self, messages: list[dict], schema: dict, temperature: float = 0.3,
                      max_tokens: int = 300) -> dict: ...


class OllamaChat:
    def __init__(self, model: str = "qwen3:4b-instruct-2507-q4_K_M", host: str = "http://localhost:11434",
                 temperature: float = 0.7, num_ctx: int | None = None):
        """num_ctx: how many tokens of conversation the model keeps. Its cache
        lives in GPU memory, so on a 4 GB card it decides whether the model
        fits beside Whisper (see config.py). None = Ollama's default."""
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx

    def _options(self, **extra) -> dict:
        opts = {"temperature": self.temperature, **extra}
        if self.num_ctx:
            opts["num_ctx"] = self.num_ctx
        return opts

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        payload = {"model": self.model, "messages": messages, "stream": True,
                   "options": self._options(), "keep_alive": KEEP_ALIVE}
        try:
            with requests.post(f"{self.host}/api/chat", json=payload, stream=True, timeout=120) as r:
                if r.status_code == 404:
                    raise RuntimeError(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if text := chunk.get("message", {}).get("content"):
                        yield text
                    if chunk.get("done"):
                        break
        except requests.ConnectionError as exc:
            raise RuntimeError(f"Can't reach Ollama at {self.host}. Is `ollama serve` running?") from exc

    def complete_json(self, messages: list[dict], schema: dict, temperature: float = 0.3,
                      max_tokens: int = 300) -> dict:
        """One non-streamed answer, shaped by a JSON schema. A lower temperature
        than the chat: helper answers should be reliable, not creative."""
        payload = {"model": self.model, "messages": messages, "stream": False, "format": schema,
                   "options": self._options(temperature=temperature, num_predict=max_tokens),
                   "keep_alive": KEEP_ALIVE}
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=120)
            r.raise_for_status()
            return json.loads(r.json()["message"]["content"])
        except requests.ConnectionError as exc:
            raise RuntimeError(f"Can't reach Ollama at {self.host}. Is `ollama serve` running?") from exc
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise RuntimeError(f"The coach couldn't answer that ({exc}).") from exc

    def check(self) -> None:
        """Raise Unavailable unless Ollama is running and has this model pulled.
        Used by the status screen, so it must be quick."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=2)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise Unavailable(f"Coach offline: can't reach Ollama at {self.host}. Start it with `ollama serve`.") from exc
        names = {m["name"] for m in r.json().get("models", [])}
        # Ollama lists "qwen2.5:3b"; a bare "llama3" means "llama3:latest".
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names:
            raise Unavailable(f"Coach offline: model '{self.model}' is not downloaded. Run: ollama pull {self.model}")

    def warm(self) -> None:
        """Load the model into (GPU) memory now, so the learner's first reply
        doesn't include the ~15 s cold start. An empty prompt only loads it."""
        try:
            # Same options as real requests: a different num_ctx would make
            # Ollama reload the model at the learner's first turn.
            requests.post(f"{self.host}/api/generate",
                          json={"model": self.model, "prompt": "", "keep_alive": KEEP_ALIVE,
                                "options": self._options()}, timeout=120)
        except requests.RequestException:
            pass  # Ollama not running: status() will report it
