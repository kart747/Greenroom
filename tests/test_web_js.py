"""Runs the browser-side unit tests (tests/js/*.mjs) with Node, if it is installed.
The hands-free voice detector lives in the browser, so it is tested in JavaScript."""

import shutil
import subprocess
from pathlib import Path

import pytest

JS_TESTS = sorted((Path(__file__).parent / "js").glob("test_*.mjs"))


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
@pytest.mark.parametrize("script", JS_TESTS, ids=lambda p: p.name)
def test_js(script):
    r = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
