"""Local LLM client (Ollama). Streams tokens so the reply can start fast --
later the voice layer will feed the first finished sentence to TTS while the
rest is still being generated."""

from __future__ import annotations

import json
from typing import Iterator, Protocol

import requests

from .errors import Unavailable


# Ollama unloads an idle model after 5 minutes, and reloading qwen2.5:3b takes
# about 15 s on this laptop. A learner reading the report for a few minutes
# should not pay that again, so keep it loaded for 30 minutes.
KEEP_ALIVE = "30m"


class ChatModel(Protocol):
    def stream_chat(self, messages: list[dict]) -> Iterator[str]: ...


class OllamaChat:
    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434",
                 temperature: float = 0.7):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        payload = {"model": self.model, "messages": messages, "stream": True,
                   "options": {"temperature": self.temperature}, "keep_alive": KEEP_ALIVE}
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
            requests.post(f"{self.host}/api/generate",
                          json={"model": self.model, "prompt": "", "keep_alive": KEEP_ALIVE}, timeout=120)
        except requests.RequestException:
            pass  # Ollama not running: status() will report it
