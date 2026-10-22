"""One-time setup: download every model Greenroom uses, then report what's ready.

    python setup_models.py            # everything
    python setup_models.py --no-nlp   # skip the optional BERT + grammar models (~1.1 GB)

This is the ONLY step that uses the internet. Afterwards the app runs fully
offline: every model is loaded from models/ (or Ollama's own store) with
"local files only" switched on.

What it fetches (sizes approximate):
  spaCy en_core_web_sm                      12 MB   sentence split, POS tags, lemmas
  Kokoro v1.0 (onnx + voices)              340 MB   coach voice
  Whisper (size from config.json)       75-480 MB   speech recognition
  DistilBERT base uncased (optional)       260 MB   context-aware word suggestions
  T5-base grammar correction (optional)    890 MB   grammar feedback
  Ollama model (config.json, qwen2.5:3b)   1.9 GB   the coach itself (via `ollama pull`)
Anything already downloaded is skipped, so it is safe to run again.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from coach.config import load_config

ROOT = Path(__file__).resolve().parent
HF_DIR = ROOT / "models" / "hf"
RESULTS: list[tuple[str, bool, str]] = []


def prefer_ipv4() -> None:
    """Try IPv4 addresses before IPv6 ones.
    On networks with broken IPv6 (measured on the dev laptop: IPv4 reaches
    huggingface.co in 0.5 s, IPv6 hangs until a 12 s timeout), every new
    connection otherwise stalls before falling back. IPv6 is still used when a
    host has no IPv4 address."""
    original = socket.getaddrinfo

    def ipv4_first(*args, **kwargs):
        return sorted(original(*args, **kwargs), key=lambda info: info[0] != socket.AF_INET)
    socket.getaddrinfo = ipv4_first


def step(name: str):
    """Run one setup step, record ok/failed, and keep going on errors:
    one failed download should not stop the others."""
    def wrap(fn):
        def run(*args, **kwargs):
            print(f"\n== {name}", flush=True)
            t = time.time()
            try:
                note = fn(*args, **kwargs) or "ready"
                RESULTS.append((name, True, f"{note} ({time.time() - t:.0f}s)"))
            except Exception as exc:  # report and continue
                RESULTS.append((name, False, str(exc).splitlines()[0][:160]))
        return run
    return wrap


@step("spaCy English model")
def spacy_model() -> str:
    import spacy
    try:
        spacy.load("en_core_web_sm")
        return "already installed"
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        return "downloaded"


@step("Kokoro voice (TTS)")
def kokoro() -> None:
    from coach.tts import download
    download()


@step("Whisper (speech recognition)")
def whisper(size: str) -> str:
    from coach.stt import download
    download(size)
    return f"'{size}'"


def hf_model(repo: str, patterns: list[str]) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(repo, cache_dir=str(HF_DIR), allow_patterns=patterns)


@step("DistilBERT (smart suggestions, optional)")
def bert() -> None:
    from coach.analyzers.vocabulary import BERT_MODEL
    hf_model(BERT_MODEL, ["*.json", "*.txt", "*.safetensors"])


@step("T5 grammar model (optional)")
def grammar() -> None:
    from coach.analyzers.grammar import GEC_MODEL
    # This repo publishes PyTorch .bin weights only (no safetensors).
    hf_model(GEC_MODEL, ["*.json", "*.txt", "*.model", "pytorch_model.bin"])


@step("Ollama + coach LLM")
def ollama(model: str, host: str) -> str:
    import requests
    if not shutil.which("ollama"):
        raise RuntimeError("Ollama is not installed. Get it from https://ollama.com, then run this again.")
    try:
        tags = requests.get(f"{host}/api/tags", timeout=3).json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama is not running at {host}. Start it with `ollama serve`.") from exc
    names = {m["name"] for m in tags.get("models", [])}
    if (model if ":" in model else f"{model}:latest") in names:
        return f"'{model}' already pulled"
    subprocess.run(["ollama", "pull", model], check=True)
    return f"'{model}' pulled"


def check_runtime(cfg: dict) -> None:
    """Load each engine the way the app does (offline), so "ready" means ready."""
    print("\n== Checking that everything loads offline", flush=True)
    from coach.engines import Engines
    engines = Engines(cfg, load=False)
    for name, loader in _loaders():
        try:
            loader(cfg)
            RESULTS.append((f"loads: {name}", True, "ok"))
        except Exception as exc:
            RESULTS.append((f"loads: {name}", False, str(exc).splitlines()[0][:160]))
    llm = engines.status()["llm"]
    RESULTS.append(("loads: LLM", llm["state"] == "ready", llm["message"] or "ok"))


def _loaders():
    from coach.analyzers.grammar import T5Corrector
    from coach.analyzers.vocabulary import HFMaskScorer
    from coach.stt import load_stt
    from coach.tts import load_tts
    return [("speech recognition", load_stt), ("coach voice", load_tts),
            ("smart suggestions", lambda cfg: HFMaskScorer()), ("grammar", lambda cfg: T5Corrector())]


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Greenroom's models (one time)")
    ap.add_argument("--no-nlp", action="store_true", help="skip the optional BERT and grammar models")
    args = ap.parse_args()
    prefer_ipv4()
    cfg = load_config()

    spacy_model()
    kokoro()
    whisper(cfg["stt"]["whisper_size"])
    if not args.no_nlp:
        bert()
        grammar()
    ollama(cfg["llm"]["model"], cfg["llm"]["host"])
    check_runtime(cfg)

    print("\n" + "=" * 60 + "\nSetup summary\n" + "=" * 60)
    for name, ok, note in RESULTS:
        print(f"  [{'ok' if ok else '!!'}] {name}: {note}")
    if all(ok for _, ok, _ in RESULTS):
        print("\nAll set. Start the app with: python greenroom.py")
    else:
        print("\nSome parts are missing (see above). The app still starts; those features show"
              " a message and switch themselves off. Typed chat needs only the LLM.")


if __name__ == "__main__":
    main()
