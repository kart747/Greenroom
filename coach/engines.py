"""Load the models once and report which features work.

Loading Whisper and Kokoro takes a few seconds, so the server starts them in a
background thread: the window opens immediately and the status pills show
"loading..." until each engine is ready. Every engine is optional. If one
fails, its status carries the learner-facing message from Unavailable
and the rest of the app keeps working (typed chat needs only the LLM).

Tests pass fakes in and call nothing here that touches real models.
"""

from __future__ import annotations

import threading

from .errors import Unavailable
from .llm import OllamaChat

LOADING, READY, OFF = "loading", "ready", "off"


class Engines:
    def __init__(self, cfg: dict, llm=None, stt=None, tts=None, load: bool = True):
        """With load=False the given engines are used as-is (tests); a missing
        one is simply reported as off."""
        self.cfg = cfg
        self.llm = llm or OllamaChat(**{k: cfg["llm"][k] for k in ("model", "host", "temperature")})
        self.stt, self.tts = stt, tts
        self._status = {
            "stt": {"state": READY if stt else OFF, "message": "" if stt else "Voice input is off."},
            "tts": {"state": READY if tts else OFF, "message": "" if tts else "Coach voice is off."},
        }
        self._lock = threading.Lock()
        if load:
            for name in ("stt", "tts"):
                self._status[name] = {"state": LOADING, "message": "Loading..."}
            threading.Thread(target=self._load_voice, daemon=True).start()

    def _load_voice(self) -> None:
        if hasattr(self.llm, "warm"):
            threading.Thread(target=self.llm.warm, daemon=True).start()
        from .stt import load_stt
        from .tts import load_tts
        for name, loader in (("tts", load_tts), ("stt", load_stt)):
            try:
                engine = loader(self.cfg)
                _warm_up(name, engine)
                with self._lock:
                    setattr(self, name, engine)
                    self._status[name] = {"state": READY, "message": ""}
            except Unavailable as err:
                with self._lock:
                    self._status[name] = {"state": OFF, "message": str(err)}

    def status(self) -> dict:
        """Current state of every engine, checked live for the LLM (Ollama can
        be started or stopped while the app is open)."""
        try:
            self.llm.check()
            llm = {"state": READY, "message": "", "model": getattr(self.llm, "model", "")}
        except Unavailable as err:
            llm = {"state": OFF, "message": str(err), "model": getattr(self.llm, "model", "")}
        with self._lock:
            return {"llm": llm, **{k: dict(v) for k, v in self._status.items()}}


def _warm_up(name: str, engine) -> None:
    """Run each voice engine once on dummy input. The first call is slow (CUDA
    kernels compile, onnxruntime plans its graph): measured 0.78 s vs 0.29 s for
    Whisper. Better to pay that at start-up than on the learner's first turn."""
    if name == "tts":
        engine.synthesize("Hello there.", next(iter(engine.voices)))
    else:
        import numpy as np
        engine.transcribe_array(np.zeros(16_000, dtype="float32"))
