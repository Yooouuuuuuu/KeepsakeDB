"""Minimal chat-completion clients for L2. Anything with ``complete(system, user) -> str``
works; the pipeline parses JSON out of the reply itself, so JSON mode is optional.
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, List, Optional, Protocol


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class OpenAIChat:
    """Any OpenAI-compatible ``/v1/chat/completions`` (proxies, llama-server, Ollama, OpenAI)."""

    def __init__(self, base_url: str, model: str, api_key: str = "", temperature: float = 0.0,
                 timeout: float = 120.0, json_mode: bool = False, extra: Optional[Dict[str, Any]] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.json_mode = json_mode
        self.extra = extra or {}
        self.name = f"openai:{model}"

    def complete(self, system: str, user: str) -> str:
        body: Dict[str, Any] = {"model": self.model, "temperature": self.temperature,
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                **self.extra}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key or 'none'}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        return data["choices"][0]["message"]["content"] or ""


class ScriptedLLM:
    """For tests: returns canned replies in order (or the last one forever)."""

    name = "scripted"

    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.calls: List[tuple] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Dict[str, Any]:
    """Tolerant JSON extraction: fenced block, or the outermost {...}."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in LLM reply")
    return json.loads(text[start:end + 1])
