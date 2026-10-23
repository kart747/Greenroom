"""One spoken turn: learner audio in, coach audio out, as fast as possible.

    recording ─> STT ─> session.user_turn ─> LLM tokens ─> SentenceChunker
                         │ (analyzers start in        └─> TTS per sentence ─> audio event
                         │  the background here)

The key trick is in the last step: as soon as the LLM finishes its FIRST
sentence, that sentence is synthesized and sent, while the LLM keeps writing the
rest. So the learner hears the coach after roughly
    STT time + time to the first full sentence + TTS of one sentence
instead of waiting for the whole reply.

Why sentences are synthesized one after another (not in a separate thread):
Ollama keeps generating on its own process while we run TTS; its tokens just
wait in the HTTP stream until we read them. So doing TTS inline costs no LLM
time, and the code stays a simple loop that is easy to follow.

run_turn() yields plain dict events, which the web server sends to the browser:
    {"type": "transcript", "text": ...}        what Whisper heard (voice turns)
    {"type": "text", "delta": ...}             LLM tokens, for the live bubble
    {"type": "audio", "index": i, "text": sentence, "wav": bytes}
    {"type": "done", "latency": {...}}

Latency per turn (seconds), stored in the session record:
    stt_s               transcription time
    llm_first_token_s   from sending the prompt to the first token
    tts_first_audio_s   synthesis time of the first sentence
    total_s             end of learner speech (the recording reaching the server)
                        to the first coach audio being ready
"""

from __future__ import annotations

import time
from typing import Callable, Iterator, Protocol

from .errors import Unavailable
from .session import Session
from .text_stream import SentenceChunker, speakable


class SpeechToText(Protocol):
    def transcribe_bytes(self, data: bytes) -> str: ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str, voice: str) -> bytes: ...


class VoicePipeline:
    def __init__(self, session: Session, stt: SpeechToText | None, tts: TextToSpeech | None,
                 voice: str | None = None, clock: Callable[[], float] = time.perf_counter):
        """stt/tts may be None when their models are missing: typed turns and
        text replies still work, there is just no voice in or out."""
        self.session = session
        self.stt = stt
        self.tts = tts
        self.voice = voice or session.persona.voice
        self.clock = clock

    def opening(self) -> dict:
        """The persona's first line, spoken (wav is None without TTS)."""
        text = self.session.opening
        return {"type": "audio", "index": 0, "text": text, "wav": self._synth(text)}

    def run_turn(self, audio: bytes | None = None, text: str | None = None) -> Iterator[dict]:
        """Run one learner turn from a recording OR typed text; yield events.
        The whole turn counts as "replying", so background analysis of earlier
        turns doesn't take CPU from transcription or speech synthesis."""
        with self.session.replying():
            yield from self._run_turn(audio, text)

    def _run_turn(self, audio: bytes | None, text: str | None) -> Iterator[dict]:
        start = self.clock()
        timings: dict = {"mode": "voice" if audio is not None else "text"}

        word_times = None      # Whisper's per-word timings, for the pace analyzer
        if audio is not None:
            if self.stt is None:
                raise Unavailable("Voice input is off. Type your answer instead.")
            if hasattr(self.stt, "transcribe_bytes_timed"):
                heard = self.stt.transcribe_bytes_timed(audio)
                text, word_times = heard.text, heard.words
            else:
                text = self.stt.transcribe_bytes(audio)
            timings["stt_s"] = round(self.clock() - start, 3)
            yield {"type": "transcript", "text": text}
        if not text or not text.strip():
            # Silence or noise only: don't send an empty turn to the coach.
            yield {"type": "done", "empty": True, "latency": timings}
            return

        chunker = SentenceChunker()
        index = 0
        llm_start = self.clock()
        try:
            for delta in self.session.user_turn(text.strip(), timing=word_times):
                if "llm_first_token_s" not in timings:
                    timings["llm_first_token_s"] = round(self.clock() - llm_start, 3)
                yield {"type": "text", "delta": delta}
                for sentence in chunker.feed(delta):
                    yield self._audio_event(sentence, index, timings, start)
                    index += 1
            if rest := chunker.flush():
                yield self._audio_event(rest, index, timings, start)
            yield {"type": "done", "latency": timings}
        finally:
            # Also runs on barge-in (generator closed early), so interrupted
            # turns still count in the latency numbers.
            self.session.add_latency(timings)

    def _audio_event(self, sentence: str, index: int, timings: dict, start: float) -> dict:
        synth_start = self.clock()
        wav = self._synth(sentence)
        if index == 0 and wav is not None:
            now = self.clock()
            timings["tts_first_audio_s"] = round(now - synth_start, 3)
            timings["total_s"] = round(now - start, 3)
        return {"type": "audio", "index": index, "text": sentence, "wav": wav}

    def _synth(self, text: str) -> bytes | None:
        # The tutor may format with **bold** or "- " lists; speak only the words.
        return self.tts.synthesize(speakable(text), self.voice) if self.tts else None
