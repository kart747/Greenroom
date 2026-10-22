"""Speech-to-text: faster-whisper, fully offline.

faster-whisper is Whisper re-implemented on CTranslate2: same accuracy, about
4x faster, and it supports int8/float16 so the "small" model fits on a 4 GB GPU
next to the LLM. Model size, device and compute type come from config["stt"]
(chosen from the hardware in coach/config.py).

The browser records with MediaRecorder, which produces webm/opus. Whisper needs
16 kHz mono float samples, so transcribe_bytes() decodes with
faster_whisper.decode_audio (PyAV under the hood, no ffmpeg binary needed).

The model is downloaded once into models/whisper/ (by setup_models.py) and
loaded from there afterwards.
"""

from __future__ import annotations

import io
import site
from pathlib import Path

from .errors import Unavailable

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
WHISPER_DIR = MODELS_DIR / "whisper"
SAMPLE_RATE = 16_000

# Whisper was trained mostly on cleaned-up transcripts, so by default it quietly
# drops "um", "uh" and repeated words, exactly what the fluency analyzer
# needs to see. Starting it with a prompt written in that disfluent style makes
# it keep them (tested: "Uh, so basically, I I worked..." instead of
# "Us, so basically. I worked..."), at no extra cost.
FILLER_PROMPT = "Umm, let me think, like, hmm... Okay, uh, so, you know, I I mean."


class WhisperSTT:
    def __init__(self, whisper_size: str = "base", device: str = "cpu",
                 compute_type: str = "int8", beam_size: int = 1,
                 download_root: Path = WHISPER_DIR, local_only: bool = True):
        try:
            if device == "cuda":
                _preload_cuda_libs()
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise Unavailable("Voice off: faster-whisper is not installed. Run: python setup_models.py") from exc
        try:
            # local_files_only=True at runtime: the app must never reach the internet.
            self.model = WhisperModel(whisper_size, device=device, compute_type=compute_type,
                                      download_root=str(download_root), local_files_only=local_only)
        except Exception as exc:  # missing files, CUDA errors, ... all mean "no voice input"
            raise Unavailable(f"Voice off: Whisper '{whisper_size}' could not load ({exc}). "
                              "Run: python setup_models.py") from exc
        # Greedy decoding (beam 1) instead of the default beam search of 5: about
        # 2x faster on short conversational turns, and the accuracy difference on
        # single sentences is small. Latency matters more in a live conversation.
        self.beam_size = beam_size

    def transcribe_bytes(self, data: bytes) -> str:
        """Transcribe an uploaded recording (any container PyAV reads: webm, ogg, wav)."""
        from faster_whisper import decode_audio
        try:
            audio = decode_audio(io.BytesIO(data), sampling_rate=SAMPLE_RATE)
        except Exception as exc:  # truncated upload, unknown format
            raise ValueError(f"Could not decode the recording: {exc}") from exc
        return self.transcribe_array(audio)

    def transcribe_array(self, audio) -> str:
        # language="en": skips language detection (saves time, and a learner's
        # accent can fool the detector into picking another language).
        # vad_filter=True: Silero VAD drops silent stretches before decoding, which
        # stops Whisper from "hallucinating" text like "Thank you." on silence.
        segments, _ = self.model.transcribe(audio, language="en", vad_filter=True,
                                            beam_size=self.beam_size,
                                            condition_on_previous_text=False,
                                            initial_prompt=FILLER_PROMPT)
        return " ".join(s.text.strip() for s in segments).strip()


def load_stt(cfg: dict) -> WhisperSTT:
    """Build the STT engine from config["stt"]; raises Unavailable if it can't run."""
    stt = cfg["stt"]
    return WhisperSTT(stt["whisper_size"], stt["device"], stt["compute_type"],
                      beam_size=stt.get("beam_size", 1))


def download(whisper_size: str) -> None:
    """One-time download of a Whisper model into models/whisper/ (the only
    time STT touches the network)."""
    from faster_whisper.utils import download_model
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    download_model(whisper_size, cache_dir=str(WHISPER_DIR))


def main() -> None:
    import argparse
    import sys
    import time

    from .config import load_config
    ap = argparse.ArgumentParser(description="Whisper speech-to-text")
    ap.add_argument("audio", nargs="?", help="audio file to transcribe")
    ap.add_argument("--download", action="store_true", help="download the configured model into models/")
    args = ap.parse_args()
    cfg = load_config()
    if args.download:
        download(cfg["stt"]["whisper_size"])
        print(f"Whisper '{cfg['stt']['whisper_size']}' is in {WHISPER_DIR}")
        return
    if not args.audio:
        ap.error("give an audio file or --download")
    try:
        stt = load_stt(cfg)
    except Unavailable as err:
        sys.exit(str(err))
    data = Path(args.audio).read_bytes()
    t = time.perf_counter()
    text = stt.transcribe_bytes(data)
    print(f"{text!r}  ({time.perf_counter() - t:.2f}s)")


def _preload_cuda_libs() -> None:
    """CTranslate2 needs cuBLAS + cuDNN. pip ships them inside the
    nvidia-cublas-cu12 / nvidia-cudnn-cu12 packages, but not on the library
    search path, so load them by full path before CTranslate2 looks for them."""
    import ctypes
    for base in site.getsitepackages():
        for sub in ("nvidia/cublas/lib", "nvidia/cudnn/lib"):
            libdir = Path(base) / sub
            if not libdir.is_dir():
                continue
            for lib in sorted(libdir.glob("lib*.so*")):
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass  # some sub-libraries depend on others; the ones CT2 needs load fine


if __name__ == "__main__":
    main()
