"""Local LLM client (Ollama). Streams tokens so the reply can start fast --
later the voice layer will feed the first finished sentence to TTS while the
rest is still being generated."""

from __future__ import annotations

import json
from typing import Iterator, Protocol

import requests


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
                   "options": {"temperature": self.temperature}}
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
