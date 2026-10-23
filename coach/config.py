"""Model choices, read from config.json in ONE place.

Priority (lowest -> highest):
  1. defaults picked from the hardware (detect_hardware + pick_defaults)
  2. values in config.json
  3. command-line flags (passed to load_config as `overrides`)

So a missing config.json still works (e.g. a classmate's laptop with no GPU),
and config.json only needs the keys you want to change.

    python -m coach.config            # show detected hardware + effective config
    python -m coach.config --write    # (re)write config.json from the hardware
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"

# Safe on any machine: CPU only, small models.
BASE_DEFAULTS: dict = {
    # The coach. Qwen3 4B Instruct (2507, non-"thinking": it answers at once)
    # replaced qwen2.5:3b after a side-by-side test: correct grammar
    # explanations and corrections where the 3B model confused "since"/"for"
    # and invented corrections. num_ctx 3072 keeps its context cache small
    # enough to share a 4 GB GPU with Whisper (Ollama: 3.4 GB vs 3.6 GB at 4096).
    "llm": {"model": "qwen3:4b-instruct-2507-q4_K_M", "host": "http://localhost:11434",
            "temperature": 0.7, "num_ctx": 3072},
    "stt": {"whisper_size": "base", "device": "cpu", "compute_type": "int8"},
    # Kokoro runs on the CPU (onnxruntime), so it never competes with the LLM for VRAM.
    # A persona's "voice" field (e.g. "british_male") is looked up in this map.
    "tts": {
        "speed": 1.0,
        "voices": {
            "american_female": "af_heart",
            "american_male": "am_michael",
            "british_female": "bf_emma",
            "british_male": "bm_george",
        },
    },
    # Model-based NLP upgrades (CPU). Each turns itself off if its model is missing.
    "analyzers": {"contextual": True, "grammar": True},
}


def detect_hardware() -> dict:
    """Return OS, CPU, RAM and NVIDIA GPU info. Uses only stdlib + nvidia-smi,
    so it works before torch/faster-whisper are installed."""
    hw = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": _cpu_name(),
        "cpu_threads": os.cpu_count() or 1,
        "ram_gb": _ram_gb(),
        "gpu": None,       # e.g. "NVIDIA GeForce RTX 3050 Laptop GPU"
        "vram_mb": 0,
    }
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip().splitlines()
            if out:
                # With several GPUs, take the first one (that's what CUDA uses by default).
                name, vram = (s.strip() for s in out[0].split(","))
                hw["gpu"], hw["vram_mb"] = name, int(vram)
        except (subprocess.SubprocessError, ValueError, OSError):
            pass  # driver present but broken -> treat as no GPU
    return hw


def _cpu_name() -> str:
    # platform.processor() is just "x86_64" on Linux; /proc/cpuinfo has the real model name.
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ram_gb() -> float:
    try:  # Linux / macOS
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return 0.0


def pick_defaults(hw: dict) -> dict:
    """Choose whisper size + compute type for this machine.

    The GPU is SHARED with the LLM, so whisper only gets what is left.
    Measured on an RTX 3050 (4 GB), six learner recordings:
      - whisper small, float16:       728 MiB, 0.20 s per turn
      - whisper small, int8_float16:  440 MiB, 0.19 s, word-for-word the same transcripts
      - whisper base,  float16:       312 MiB, 0.12 s, slightly less faithful to fillers
    int8_float16 stores the weights in 8 bits and computes in 16, so "small"
    keeps its accuracy while leaving ~0.3 GB more for the LLM.
    On CPU, int8 is about 2-4x faster than float32 with almost no accuracy loss,
    and "base" keeps transcription of a short turn well under a second on a modern CPU.
    """
    cfg = copy.deepcopy(BASE_DEFAULTS)
    stt = cfg["stt"]
    if hw.get("gpu") and hw.get("vram_mb", 0) >= 3900:  # "4 GB" cards report ~4096 MiB
        stt.update(whisper_size="small", device="cuda", compute_type="int8_float16")
    elif hw.get("gpu") and hw.get("vram_mb", 0) >= 2000:
        stt.update(whisper_size="base", device="cuda", compute_type="int8_float16")
    elif hw.get("ram_gb", 0) and hw["ram_gb"] < 6:
        stt.update(whisper_size="tiny")  # low-RAM laptop: smallest model
    return cfg


def deep_merge(base: dict, extra: dict) -> dict:
    """Return base updated with extra, merging nested dicts instead of replacing them,
    so config.json can set just {"stt": {"whisper_size": "tiny"}}."""
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path = CONFIG_PATH, overrides: dict | None = None,
                hw: dict | None = None) -> dict:
    """Effective config = hardware defaults <- config.json <- CLI overrides.

    `overrides` uses the same nested shape, e.g. {"llm": {"model": "llama3.2:3b"}}.
    Values that are None are ignored, so unset argparse flags don't erase config.
    """
    cfg = pick_defaults(hw if hw is not None else detect_hardware())
    if path.exists():
        cfg = deep_merge(cfg, json.loads(path.read_text()))
    if overrides:
        cfg = deep_merge(cfg, _drop_none(overrides))
    return cfg


def _drop_none(d: dict) -> dict:
    return {k: _drop_none(v) if isinstance(v, dict) else v
            for k, v in d.items() if v is not None}


def main() -> None:
    ap = argparse.ArgumentParser(description="Show or write the model config")
    ap.add_argument("--write", action="store_true", help="write config.json from detected hardware")
    args = ap.parse_args()
    hw = detect_hardware()
    print("Detected hardware:")
    for k, v in hw.items():
        print(f"  {k}: {v}")
    if args.write:
        CONFIG_PATH.write_text(json.dumps(pick_defaults(hw), indent=2) + "\n")
        print(f"\nWrote {CONFIG_PATH}")
    print("\nEffective config:")
    print(json.dumps(load_config(hw=hw), indent=2))


if __name__ == "__main__":
    main()
