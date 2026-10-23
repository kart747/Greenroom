"""Config loading: hardware-based defaults, config.json, CLI overrides.
Hardware is passed in as a dict, so no GPU or nvidia-smi is needed."""

import json
import subprocess

from coach import config
from coach.config import deep_merge, load_config, pick_defaults

GPU_4GB = {"gpu": "NVIDIA GeForce RTX 3050 Laptop GPU", "vram_mb": 4096, "ram_gb": 19.0}
GPU_2GB = {"gpu": "NVIDIA GeForce MX450", "vram_mb": 2048, "ram_gb": 8.0}
CPU_ONLY = {"gpu": None, "vram_mb": 0, "ram_gb": 16.0}
CPU_LOW_RAM = {"gpu": None, "vram_mb": 0, "ram_gb": 4.0}


def test_defaults_gpu_4gb_uses_small_int8_float16():
    stt = pick_defaults(GPU_4GB)["stt"]
    assert stt == {"whisper_size": "small", "device": "cuda", "compute_type": "int8_float16"}


def test_defaults_small_gpu_uses_base():
    assert pick_defaults(GPU_2GB)["stt"]["whisper_size"] == "base"


def test_defaults_cpu_uses_int8():
    stt = pick_defaults(CPU_ONLY)["stt"]
    assert stt["device"] == "cpu" and stt["compute_type"] == "int8"
    assert pick_defaults(CPU_LOW_RAM)["stt"]["whisper_size"] == "tiny"


def test_default_llm_is_small_local_model():
    llm = pick_defaults(CPU_ONLY)["llm"]
    assert llm["model"] == "qwen3:4b-instruct-2507-q4_K_M" and llm["num_ctx"] == 3072   # 4B: within the 5B limit


def test_pick_defaults_does_not_mutate_base():
    pick_defaults(GPU_4GB)
    assert config.BASE_DEFAULTS["stt"]["device"] == "cpu"


def test_deep_merge_keeps_sibling_keys():
    merged = deep_merge({"stt": {"whisper_size": "base", "device": "cpu"}},
                       {"stt": {"whisper_size": "tiny"}})
    assert merged == {"stt": {"whisper_size": "tiny", "device": "cpu"}}


def test_missing_file_gives_hardware_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.json", hw=GPU_4GB)
    assert cfg == pick_defaults(GPU_4GB)


def test_file_overrides_defaults_and_cli_overrides_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"llm": {"model": "llama3.2:3b"}, "stt": {"whisper_size": "tiny"}}))
    cfg = load_config(path, hw=GPU_4GB)
    assert cfg["llm"]["model"] == "llama3.2:3b"
    assert cfg["stt"]["whisper_size"] == "tiny"
    assert cfg["stt"]["device"] == "cuda"          # untouched keys keep the hardware default

    cfg = load_config(path, overrides={"llm": {"model": "phi3:mini"}}, hw=GPU_4GB)
    assert cfg["llm"]["model"] == "phi3:mini"


def test_none_override_does_not_erase_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"llm": {"model": "llama3.2:3b"}}))
    cfg = load_config(path, overrides={"llm": {"model": None}}, hw=CPU_ONLY)
    assert cfg["llm"]["model"] == "llama3.2:3b"


def test_detect_hardware_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(config.shutil, "which", lambda _: None)
    hw = config.detect_hardware()
    assert hw["gpu"] is None and hw["vram_mb"] == 0
    assert hw["cpu_threads"] >= 1


def test_detect_hardware_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(config.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    fake = subprocess.CompletedProcess([], 0, stdout="NVIDIA GeForce RTX 3050 Laptop GPU, 4096\n")
    monkeypatch.setattr(config.subprocess, "run", lambda *a, **k: fake)
    hw = config.detect_hardware()
    assert hw["gpu"] == "NVIDIA GeForce RTX 3050 Laptop GPU" and hw["vram_mb"] == 4096


def test_repo_config_json_is_valid():
    cfg = load_config(hw=CPU_ONLY)  # reads the real config.json
    assert set(cfg) >= {"llm", "stt"}
    assert cfg["stt"]["compute_type"] in {"int8", "float16", "int8_float16", "float32"}
