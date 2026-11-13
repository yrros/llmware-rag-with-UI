"""Cloud LLM providers for LLM Notes, beside the local Ollama models.

Two kinds:
- Claude, through Anthropic's official Python SDK (`anthropic`).
- Any OpenAI-compatible endpoint (LM Studio, llama.cpp server, vLLM, OpenRouter, Groq, a company gateway, ...),
  through plain HTTP `POST {base_url}/chat/completions`.

A cloud model's name starts with CLOUD_MARK, so it can sit in the same model lists as Ollama's models:
"☁ claude-opus-5-5" for Claude, "☁ <endpoint name>/<model id>" for an OpenAI-compatible endpoint. Anything else
is an Ollama model.

CloudChatStream streams a reply as the same JSON lines Ollama's /api/chat sends, so the app reads every model
the same way. API keys live in the macOS Keychain (via `keyring`), never in the app's settings files.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

CLOUD_MARK = "☁ "
CLAUDE_MODELS = {  # the two Claude models offered: in-depth work, and quick jobs
    "claude-opus-5-5": "Claude Opus 5.5 · in depth",
    "claude-haiku-5-5": "Claude Haiku 5.5 · quick",
}
CLAUDE_CONTEXT = 1_000_000
CLAUDE_HOST = "api.anthropic.com"
CLAUDE_EFFORTS = ("low", "medium", "high")
# Models whose refusals the API re-runs on another Claude model (server-side fallback, "default" routing).
CLAUDE_REFUSAL_FALLBACK = {"claude-opus-5-5"}
OPENAI_DEFAULT_CONTEXT = 32768
KEYCHAIN_SERVICE = "LLM-Notes"
ONLINE_TTL = 30  # seconds a reachability check is trusted


class CloudError(RuntimeError):
    """A cloud model couldn't answer (offline, bad key, rate limit, refusal, ...): the app may use the local
    fallback instead."""


# --- Names --------------------------------------------------------------------------------------------------


def claude_ref(model: str) -> str:
    return CLOUD_MARK + model


def openai_ref(endpoint: str, model: str) -> str:
    return f"{CLOUD_MARK}{endpoint}/{model}"


def parse(name: str) -> tuple[str, str, str]:
    """(provider, endpoint name, model id): provider is "claude", "openai" or "ollama"."""
    name = str(name or "")
    if not name.startswith(CLOUD_MARK):
        return "ollama", "", name
    rest = name[len(CLOUD_MARK):]
    if rest in CLAUDE_MODELS or rest.startswith("claude-"):
        return "claude", "", rest
    endpoint, _, model = rest.partition("/")
    return "openai", endpoint, model


def is_cloud(name: str) -> bool:
    return parse(name)[0] != "ollama"


def describe(name: str) -> str:
    """A readable name: "Claude Opus 5.5 · in depth", "OpenRouter: gpt-x", or the Ollama name."""
    provider, endpoint, model = parse(name)
    if provider == "claude":
        return CLAUDE_MODELS.get(model, model)
    if provider == "openai":
        return f"{endpoint}: {model}"
    return name


# --- Keys (macOS Keychain) ------------------------------------------------------------------------------------


def key_account(provider: str, endpoint: str = "") -> str:
    return "anthropic" if provider == "claude" else f"openai:{endpoint}"


def get_key(account: str) -> str:
    try:
        import keyring

        return keyring.get_password(KEYCHAIN_SERVICE, account) or ""
    except Exception:
        return ""


def set_key(account: str, value: str) -> None:
    """Save the key in the Keychain; an empty value removes it."""
    import keyring

    if value:
        keyring.set_password(KEYCHAIN_SERVICE, account, value)
    else:
        try:
            keyring.delete_password(KEYCHAIN_SERVICE, account)
        except Exception:
            pass


def claude_key_source() -> str:
    """Where a Claude key comes from: "keychain", "environment", "profile" (`ant auth login`), or ""."""
    if get_key(key_account("claude")):
        return "keychain"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "environment"
    if (Path.home() / ".config" / "anthropic").is_dir():
        return "profile"
    return ""


# --- Reachability ----------------------------------------------------------------------------------------------

_reachable: dict[tuple[str, int], tuple[float, bool]] = {}


def reachable(host: str, port: int, ttl: float = ONLINE_TTL) -> bool:
    """Whether a TCP connection to host:port opens (cached for `ttl` seconds), so an offline Mac goes straight
    to the local model instead of waiting for a timeout."""
    now = time.monotonic()
    hit = _reachable.get((host, port))
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        socket.create_connection((host, port), timeout=2).close()
        ok = True
    except OSError:
        ok = False
    _reachable[(host, port)] = (now, ok)
    return ok


def mark_unreachable(host: str, port: int) -> None:
    _reachable[(host, port)] = (time.monotonic(), False)


def host_port(base_url: str) -> tuple[str, int]:
    url = urlparse(base_url)
    return url.hostname or "", url.port or (443 if url.scheme == "https" else 80)


def endpoint_config(settings: dict, name: str) -> dict | None:
    return next((e for e in settings.get("endpoints") or [] if e.get("name") == name), None)


def model_host(name: str, settings: dict) -> tuple[str, int] | None:
    provider, endpoint, _model = parse(name)
    if provider == "claude":
        return CLAUDE_HOST, 443
    if provider == "openai":
        config = endpoint_config(settings, endpoint)
        return host_port(config["base_url"]) if config and config.get("base_url") else None
    return None


def is_reachable(name: str, settings: dict) -> bool:
    where = model_host(name, settings)
    return bool(where and where[0]) and reachable(*where)


# --- Model lists and limits -------------------------------------------------------------------------------------


def openai_models(base_url: str, key: str, timeout: float = 5) -> list[dict]:
    """[{"id", "context_length"}] from GET {base_url}/models (context_length only when the endpoint says)."""
    import requests

    headers = {"Authorization": f"Bearer {key}"} if key else {}
    response = requests.get(base_url.rstrip("/") + "/models", headers=headers, timeout=timeout)
    response.raise_for_status()
    models = []
    for entry in response.json().get("data") or []:
        if entry.get("id"):
            length = entry.get("context_length") or entry.get("max_context_length") or entry.get("context_window")
            models.append({"id": str(entry["id"]), "context_length": int(length) if length else None})
    return sorted(models, key=lambda m: m["id"])


def context_length(name: str, settings: dict) -> int | None:
    provider, endpoint, _model = parse(name)
    if provider == "claude":
        return CLAUDE_CONTEXT
    if provider == "openai":
        config = endpoint_config(settings, endpoint) or {}
        return int(config.get("context") or OPENAI_DEFAULT_CONTEXT)
    return None


# --- Requests -------------------------------------------------------------------------------------------------


def claude_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """(system prompt, messages) for the Messages API: system messages joined into the system prompt, empty
    messages dropped, neighbours with the same role merged, and a user message first and last (no prefill)."""
    system, out = [], []
    for message in messages:
        role, content = message.get("role"), str(message.get("content") or "")
        if role == "system":
            if content.strip():
                system.append(content)
            continue
        if not content.strip():
            continue
        role = "assistant" if role == "assistant" else "user"
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + content
        else:
            out.append({"role": role, "content": content})
    if not out or out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "(The conversation so far follows.)"})
    if out[-1]["role"] != "user":
        out.append({"role": "user", "content": "Please continue."})
    return "\n\n".join(system), out


_UNSUPPORTED_SCHEMA_KEYS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength",
                            "maxLength", "minItems", "maxItems", "uniqueItems"}


def strict_schema(schema):
    """A JSON schema structured outputs accept: no numeric, length or array-size limits, and
    additionalProperties false on every object."""
    if isinstance(schema, list):
        return [strict_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out = {key: strict_schema(value) for key, value in schema.items() if key not in _UNSUPPORTED_SCHEMA_KEYS}
    if out.get("type") == "object" or "properties" in out:
        out["additionalProperties"] = False
    return out


def claude_max_tokens(num_predict) -> int:
    """Room for Claude's thinking as well as the answer: the app's Max tokens counts both, and Claude's thinking
    can't be turned off, so at least 8,000 (at most 64,000)."""
    return min(64000, max(int(num_predict or 0) * 2, 8000))


class CloudChatStream:
    """A cloud model's reply streamed on a worker thread as Ollama /api/chat lines on `lines`:
    ("line", bytes) ..., then ("end", None); or ("error", CloudError). `close()` stops it."""

    def __init__(self, name: str, payload: dict, settings: dict):
        self.lines: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._closer = None
        threading.Thread(target=self._run, args=(name, payload, settings), daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        try:
            self._closer and self._closer()
        except Exception:
            pass

    def _emit(self, **chunk) -> None:
        self.lines.put(("line", json.dumps(chunk).encode()))

    def _run(self, name: str, payload: dict, settings: dict) -> None:
        provider, endpoint, model = parse(name)
        try:
            if provider == "claude":
                self._claude(model, payload, settings)
            else:
                self._openai(endpoint, model, payload, settings)
            self.lines.put(("end", None))
        except CloudError as exc:
            if not self._stop.is_set():
                self.lines.put(("error", exc))
        except Exception as exc:  # network, SDK, bad JSON: all mean the cloud couldn't answer
            if not self._stop.is_set():
                where = model_host(name, settings)
                if where and (isinstance(exc, OSError) or "Connection" in type(exc).__name__):
                    mark_unreachable(*where)
                self.lines.put(("error", CloudError(f"{describe(name)}: {type(exc).__name__}: {exc}")))

    def _claude(self, model: str, payload: dict, settings: dict) -> None:
        import anthropic

        key = get_key(key_account("claude"))
        client = anthropic.Anthropic(api_key=key or None, max_retries=1,
                                     timeout=anthropic.Timeout(900.0, connect=5.0))
        system, messages = claude_messages(payload.get("messages") or [])
        options = payload.get("options") or {}
        effort = "low" if payload.get("think") is False else (settings.get("effort") or "medium")
        output_config: dict = {"effort": effort}
        if payload.get("format"):
            output_config["format"] = {"type": "json_schema", "schema": strict_schema(payload["format"])}
        request = {"model": model, "max_tokens": claude_max_tokens(options.get("num_predict")),
                   "messages": messages, "thinking": {"type": "adaptive", "display": "summarized"},
                   "output_config": output_config}
        if system:
            request["system"] = system
        if model in CLAUDE_REFUSAL_FALLBACK:  # a declined request is re-run on another Claude model, server-side
            request.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        started, first = time.monotonic(), None
        try:
            with client.beta.messages.stream(**request) as stream:
                self._closer = getattr(stream, "close", None)
                for event in stream:
                    if self._stop.is_set():
                        return
                    if event.type != "content_block_delta":
                        continue
                    delta = event.delta
                    if delta.type == "text_delta" and delta.text:
                        first = first or time.monotonic()
                        self._emit(message={"content": delta.text})
                    elif delta.type == "thinking_delta" and delta.thinking:
                        self._emit(message={"thinking": delta.thinking})
                final = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise CloudError("Claude rejected the API key: check it under Configure > Models and providers.") from exc
        except anthropic.RateLimitError as exc:
            raise CloudError("Claude is rate-limiting this key just now.") from exc
        except anthropic.APIConnectionError as exc:  # includes timeouts
            mark_unreachable(CLAUDE_HOST, 443)
            raise CloudError(f"Couldn't reach Claude: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise CloudError(f"Claude returned HTTP {exc.status_code}: {exc.message}") from exc
        if final.stop_reason == "refusal":
            raise CloudError("Claude declined to answer this request.")
        usage = final.usage
        prompt = int(usage.input_tokens or 0) + int(getattr(usage, "cache_read_input_tokens", 0) or 0) \
            + int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        now = time.monotonic()
        self._emit(done=True, done_reason="length" if final.stop_reason == "max_tokens" else "stop",
                   prompt_eval_count=prompt, eval_count=int(usage.output_tokens or 0),
                   total_duration=int((now - started) * 1e9),
                   eval_duration=int((now - first) * 1e9) if first else None,
                   served_by=claude_ref(getattr(final, "model", model) or model))

    def _openai(self, endpoint: str, model: str, payload: dict, settings: dict) -> None:
        import requests

        config = endpoint_config(settings, endpoint)
        if not config or not config.get("base_url"):
            raise CloudError(f"No endpoint named “{endpoint}” is set up (Configure > Models and providers).")
        headers = {"Content-Type": "application/json"}
        key = get_key(key_account("openai", endpoint))
        if key:
            headers["Authorization"] = f"Bearer {key}"
        options = payload.get("options") or {}
        body = {"model": model, "stream": True, "stream_options": {"include_usage": True},
                "messages": [{"role": m.get("role") or "user", "content": str(m.get("content") or "")}
                             for m in payload.get("messages") or []]}
        if options.get("num_predict"):
            body["max_tokens"] = int(options["num_predict"])
        if options.get("temperature") is not None:
            body["temperature"] = float(options["temperature"])
        if payload.get("format"):
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "reply", "schema": payload["format"]}}
        url = config["base_url"].rstrip("/") + "/chat/completions"
        started, first = time.monotonic(), None
        try:
            response = requests.post(url, json=body, headers=headers, stream=True, timeout=(5, 900))
            if response.status_code == 400 and "response_format" in body:  # the endpoint can't constrain JSON
                response.close()
                body.pop("response_format")
                response = requests.post(url, json=body, headers=headers, stream=True, timeout=(5, 900))
        except requests.ConnectionError as exc:
            mark_unreachable(*host_port(config["base_url"]))
            raise CloudError(f"Couldn't reach {endpoint}: {exc}") from exc
        if response.status_code >= 400:
            text = response.text[:300]
            response.close()
            raise CloudError(f"{endpoint} returned HTTP {response.status_code}: {text}")
        self._closer = response.close
        finish, usage = None, {}
        for raw in response.iter_lines():
            if self._stop.is_set():
                return
            line = raw.decode(errors="replace").strip() if raw else ""
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise CloudError(f"{endpoint}: {chunk['error']}")
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                thinking = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if text:
                    first = first or time.monotonic()
                if text or thinking:
                    self._emit(message={"content": text, "thinking": thinking})
                finish = choice.get("finish_reason") or finish
            usage = chunk.get("usage") or usage
        now = time.monotonic()
        self._emit(done=True, done_reason="length" if finish == "length" else "stop",
                   prompt_eval_count=int(usage.get("prompt_tokens") or 0),
                   eval_count=int(usage.get("completion_tokens") or 0),
                   total_duration=int((now - started) * 1e9),
                   eval_duration=int((now - first) * 1e9) if first else None,
                   served_by=openai_ref(endpoint, model))


def complete(name: str, payload: dict, settings: dict, timeout: float = 120) -> str:
    """A whole cloud reply's text (for short, non-streamed jobs); raises CloudError."""
    stream = CloudChatStream(name, payload, settings)
    parts, deadline = [], time.monotonic() + timeout
    try:
        while True:
            kind, value = stream.lines.get(timeout=max(deadline - time.monotonic(), 0.1))
            if kind == "error":
                raise value
            if kind == "end":
                return "".join(parts)
            chunk = json.loads(value)
            parts.append((chunk.get("message") or {}).get("content") or "")
            if chunk.get("done"):
                return "".join(parts)
    except queue.Empty as exc:
        raise CloudError(f"{describe(name)} took longer than {timeout:.0f} s.") from exc
    finally:
        stream.close()
