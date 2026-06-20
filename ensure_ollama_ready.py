#!/usr/bin/env python3
"""Ensure Ollama is running and warm the configured Answer-with-LLM model."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CORPUS_SETTINGS_PATH = PROJECT_DIR / "llmware_data" / "corpus_settings.json"
OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
STARTUP_TIMEOUT_SECONDS = 60


def configured_ollama_model() -> str:
    if not CORPUS_SETTINGS_PATH.exists():
        return DEFAULT_OLLAMA_MODEL
    try:
        all_settings = json.loads(CORPUS_SETTINGS_PATH.read_text())
    except Exception:
        return DEFAULT_OLLAMA_MODEL

    models = [
        str(cfg["ollama_model"])
        for cfg in all_settings.values()
        if isinstance(cfg, dict) and cfg.get("ollama_model")
    ]
    if not models:
        return DEFAULT_OLLAMA_MODEL
    return Counter(models).most_common(1)[0][0]


def ollama_api_url(path: str) -> str:
    return f"http://{OLLAMA_HOST}:{OLLAMA_PORT}{path}"


def is_ollama_reachable() -> bool:
    try:
        import requests

        response = requests.get(ollama_api_url("/api/tags"), timeout=3)
        return response.ok
    except Exception:
        return False


def start_ollama_server() -> bool:
    if shutil.which("ollama") is None:
        print("Ollama is not installed or not on PATH.", flush=True)
        return False

    print("Starting Ollama…", flush=True)
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline:
        if is_ollama_reachable():
            print("Ollama is running.", flush=True)
            return True
        time.sleep(1)

    print("Timed out waiting for Ollama to start.", flush=True)
    return False


def installed_ollama_models() -> list[str]:
    try:
        import requests

        response = requests.get(ollama_api_url("/api/tags"), timeout=5)
        response.raise_for_status()
        return [entry["name"] for entry in response.json().get("models", [])]
    except Exception:
        return []


def model_is_installed(model_name: str, installed: list[str]) -> bool:
    if model_name in installed:
        return True
    base = model_name.split(":")[0]
    return any(name == model_name or name.split(":")[0] == base for name in installed)


def pull_ollama_model(model_name: str) -> bool:
    print(f"Pulling Ollama model: {model_name}", flush=True)
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Failed to pull {model_name}: {exc}", flush=True)
        return False


def warm_ollama_model(model_name: str) -> bool:
    try:
        import requests

        print(f"Warming Ollama model: {model_name}", flush=True)
        response = requests.post(
            ollama_api_url("/api/chat"),
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": " "}],
                "stream": False,
                "keep_alive": "-1",
            },
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            print(f"Failed to warm {model_name}: {data['error']}", flush=True)
            return False
        print(f"Ready: {model_name}", flush=True)
        return True
    except Exception as exc:
        print(f"Failed to warm {model_name}: {exc}", flush=True)
        return False


def main() -> int:
    model_name = configured_ollama_model()
    print(f"Configured Answer-with-LLM model: {model_name}", flush=True)

    if not is_ollama_reachable() and not start_ollama_server():
        print("Answer with LLM will be unavailable until Ollama is running.", flush=True)
        return 0

    installed = installed_ollama_models()
    if not model_is_installed(model_name, installed):
        if not pull_ollama_model(model_name):
            print("Continuing without warming the LLM.", flush=True)
            return 0

    if not warm_ollama_model(model_name):
        print("Continuing without warming the LLM.", flush=True)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
