"""Stand-ins for the real models, so the whole suite runs without a GPU, a mic,
Ollama or any downloads. Each fake has the same method the real engine has."""

from __future__ import annotations


class FakeLLM:
    """Replaces OllamaChat: streams a fixed reply in small pieces, like Ollama does."""

    def __init__(self, reply: str = "That sounds great! Tell me more about it?", facts=None):
        self.reply = reply
        self.seen: list[list[dict]] = []
        self.json_calls: list[tuple[list[dict], dict]] = []
        self.facts = facts or []

    def stream_chat(self, messages):
        self.seen.append(list(messages))
        for word in self.reply.split(" "):
            yield word + " "

    def complete_json(self, messages, schema, temperature=0.3, max_tokens=300):
        """Answers shaped like the schema asks, so helper code paths run for real."""
        self.json_calls.append((messages, schema))
        props = schema["properties"]
        if "suggestions" in props:
            return {"suggestions": ["I studied computer science.", "What about you?", "I live in Chennai."]}
        if "better" in props:
            return {"better": "I have two years of experience.", "why": "Use 'have' for experience, not 'am having'."}
        if "meaning" in props:
            return {"meaning": "very bad or worrying", "example": "The traffic was dreadful.", "part_of_speech": "adjective"}
        if "facts" in props:
            return {"facts": list(self.facts)}
        raise AssertionError(f"unexpected schema {schema}")

    def check(self):
        pass


class FakeSTT:
    """Replaces WhisperSTT: 'hears' a fixed sentence, with word timings like
    Whisper's: 0.3 s per word, 0.1 s gaps, and one 2-second pause before the
    fourth word (so the pace analyzer has something to find)."""

    def __init__(self, text: str = "Umm the trip was very good."):
        self.text = text
        self.calls = 0

    def transcribe_bytes(self, data: bytes) -> str:
        return self.transcribe_bytes_timed(data).text

    def transcribe_bytes_timed(self, data: bytes):
        from coach.stt import Transcript
        self.calls += 1
        words, t = [], 0.0
        for i, w in enumerate(self.text.split()):
            t += 2.0 if i == 3 else (0.1 if i else 0.0)
            words.append({"word": w, "start": round(t, 2), "end": round(t + 0.3, 2), "prob": 0.9})
            t += 0.3
        return Transcript(self.text, words)


class FakeTTS:
    """Replaces KokoroTTS: returns a tiny valid WAV and remembers what it spoke."""

    def __init__(self):
        self.spoken: list[tuple[str, str]] = []

    def synthesize(self, text: str, voice: str) -> bytes:
        import numpy as np

        from coach.tts import to_wav
        self.spoken.append((text, voice))
        return to_wav(np.zeros(240, dtype="float32"), 24000)
