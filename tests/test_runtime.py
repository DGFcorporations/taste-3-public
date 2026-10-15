"""Real Granian private Unix socket startup/request/shutdown, synthetic DB."""

import importlib
import os
import stat
import subprocess
import sys
import time

import httpx
import pytest

from taste.config import ROOT


def test_real_private_uds_runtime_and_shutdown(tmp_path):
    try:
        importlib.import_module("taste.cli")
    except ModuleNotFoundError:
        pytest.fail("Missing UDS runtime implementation")
    socket = ROOT / "_agent-scratch" / f"uds-{os.getpid()}.sock"
    env = dict(os.environ)
    env["TASTE3_DB"] = str(tmp_path / "runtime.db")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.Popen(
        [sys.executable, "-m", "taste.cli", "serve", "--socket", str(socket)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 15
        response = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail("Private UDS server exited before accepting a request")
            try:
                with httpx.Client(
                    transport=httpx.HTTPTransport(uds=str(socket)),
                    base_url="http://private",
                    timeout=0.5,
                ) as client:
                    response = client.get("/meta")
                    break
            except httpx.TransportError:
                time.sleep(0.05)
        assert response is not None
        assert response.json()["contract"] == "v1"
        assert stat.S_IMODE(socket.stat().st_mode) == 0o600
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
    assert process.returncode == 0
    assert not socket.exists()
