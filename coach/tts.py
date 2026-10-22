"""Text-to-speech: Kokoro (82M parameters) through kokoro-onnx, fully offline.

Why Kokoro: it sounds far more natural than older offline engines (espeak,
Piper), it is small enough to run on the CPU faster than real time, and it has
both American and British voices, which the accent setting needs.
It runs on onnxruntime CPU, so the GPU stays free for the LLM and Whisper.

    python -m coach.tts --download      # once: fetch the model into models/
    python -m coach.tts "Hello there."   # speak a test sentence to out.wav

config["tts"] = {"speed": 1.0, "voices": {"american_female": "af_heart", ...}}
A persona's "voice" field ("british_male") picks a Kokoro voice from that map.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np

from .errors import Unavailable

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
KOKORO_DIR = MODELS_DIR / "kokoro"
FILES = ["kokoro-v1.0.onnx", "voices-v1.0.bin"]
# The official release is on GitHub; the same files are mirrored on Hugging Face,
# which is much faster from some networks. Try the mirror first, then GitHub,
# and accept a file only if its SHA-256 matches (so a mirror can't swap it).
SOURCES = [
    "https://huggingface.co/fastrtc/kokoro-onnx/resolve/main/",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/",
]
SHA256 = {
    "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}


class KokoroTTS:
    def __init__(self, voices: dict[str, str], speed: float = 1.0, model_dir: Path = KOKORO_DIR):
        model, voice_file = (model_dir / f for f in FILES)
        if not (model.exists() and voice_file.exists()):
            raise Unavailable("Voice off: the Kokoro voice model is missing. Run: python setup_models.py")
        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise Unavailable("Voice off: kokoro-onnx is not installed. Run: pip install -r requirements.txt") from exc
        self.kokoro = Kokoro(str(model), str(voice_file))
        self.voices = voices
        self.speed = speed

    def synthesize(self, text: str, voice: str) -> bytes:
        """Speak `text` in a persona voice ("british_male") -> WAV file bytes."""
        kokoro_voice = self.voices.get(voice, next(iter(self.voices.values())))
        # Kokoro voice ids start with the accent: "a" = American, "b" = British.
        # The phonemizer must match, or British voices get American vowels.
        lang = "en-gb" if kokoro_voice.startswith("b") else "en-us"
        samples, rate = self.kokoro.create(text, voice=kokoro_voice, speed=self.speed, lang=lang)
        return to_wav(samples, rate)


def to_wav(samples: np.ndarray, rate: int) -> bytes:
    """Float samples in [-1, 1] -> 16-bit PCM WAV, which every browser can play."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def load_tts(cfg: dict) -> KokoroTTS:
    """Build the TTS engine from config["tts"]; raises Unavailable if it can't run."""
    return KokoroTTS(cfg["tts"]["voices"], cfg["tts"].get("speed", 1.0))


def download(model_dir: Path = KOKORO_DIR) -> None:
    """One-time download (~340 MB). The only time Kokoro touches the network."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = model_dir / name
        if target.exists():
            print(f"  {name}: already downloaded")
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        for base in SOURCES:
            print(f"  {name}: downloading from {base.split('/')[2]}...", flush=True)
            try:
                urllib.request.urlretrieve(base + name, tmp)
            except OSError as exc:
                print(f"    failed ({exc}), trying the next source")
                continue
            if _sha256(tmp) == SHA256[name]:
                tmp.rename(target)  # rename at the end so a cut download is never mistaken for a full one
                break
            print("    checksum mismatch, trying the next source")
        else:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"Could not download {name}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Kokoro text-to-speech")
    ap.add_argument("text", nargs="?", default="Hello! Let's practise speaking English together.")
    ap.add_argument("--download", action="store_true", help="download the model files into models/")
    ap.add_argument("--voice", default="american_female")
    args = ap.parse_args()
    if args.download:
        download()
        return
    from .config import load_config
    try:
        tts = load_tts(load_config())
    except Unavailable as err:
        sys.exit(str(err))
    Path("out.wav").write_bytes(tts.synthesize(args.text, args.voice))
    print("Wrote out.wav")


if __name__ == "__main__":
    main()
