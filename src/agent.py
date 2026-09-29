"""
agent.py — Builds the system prompt and calls the LLM.
Uses OpenRouter (primary) via requests, falls back to Groq SDK.
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
PROMPTS_DIR = ROOT / "prompts"
CONFIG_PATH = ROOT / "config.json"

PromptVersion = Literal["v1", "v2", "v3"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class AgentConfig:
    def __init__(self, data: dict) -> None:
        self.agent_name: str = data["agent_name"]
        self.company: str = data["company"]
        self.product: str = data["product"]
        self.goal: str = data["goal"]
        self.model: str = data["model"]
        self.max_tokens: int = data["max_tokens_agent"]
        self.max_tokens_customer: int = data.get("max_tokens_customer", 150)
        self.max_turns: int = data["max_turns"]
        self.qualification_fields: list[str] = data["qualification_fields"]

    @classmethod
    def load(cls) -> "AgentConfig":
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return cls(json.load(f))


_config: AgentConfig | None = None


def get_config() -> AgentConfig:
    global _config
    if _config is None:
        _config = AgentConfig.load()
    return _config


def reset_config() -> None:
    global _config
    _config = None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def _load_prompt_template(version: PromptVersion) -> str:
    path = PROMPTS_DIR / f"agent_{version}.txt"
    return path.read_text(encoding="utf-8")


def build_system_prompt(version: PromptVersion) -> str:
    cfg = get_config()
    template = _load_prompt_template(version)
    return template.format(
        agent_name=cfg.agent_name,
        company=cfg.company,
        product=cfg.product,
        goal=cfg.goal,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_reasoning(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _parse_retry_after(error_msg: str) -> float | None:
    m = re.search(r"try again in\s+(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s", error_msg)
    if m:
        minutes = int(m.group(1)) if m.group(1) else 0
        return minutes * 60 + float(m.group(2)) + 2
    return None


# ---------------------------------------------------------------------------
# OpenRouter call (via urllib — no SDK needed)
# ---------------------------------------------------------------------------
def _call_openrouter(system: str, messages: list[dict], model: str, max_tokens: int) -> str:
    key = os.environ["OPENROUTER_API_KEY"]
    full_messages = [{"role": "system", "content": system}] + messages
    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": full_messages,
    }).encode()

    for attempt in range(5):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/voice-agent-eval",
                },
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
            raw = resp["choices"][0]["message"]["content"] or ""
            return _strip_reasoning(raw.strip())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()
            is_rate = exc.code == 429
            if is_rate and attempt < 4:
                wait = _parse_retry_after(body) or (15 * (attempt + 1))
                print(f"  [rate-limit] waiting {wait:.0f}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body[:200]}") from exc


# ---------------------------------------------------------------------------
# Groq fallback
# ---------------------------------------------------------------------------
def _call_groq(system: str, messages: list[dict], model: str, max_tokens: int) -> str:
    from groq import Groq
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise EnvironmentError("GROQ_API_KEY not set")
    client = Groq(api_key=key)
    full_messages = [{"role": "system", "content": system}] + messages

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=max_tokens, messages=full_messages
            )
            return _strip_reasoning(resp.choices[0].message.content.strip())
        except Exception as exc:
            msg = str(exc)
            if ("429" in msg or "rate_limit" in msg.lower()) and attempt < 4:
                wait = _parse_retry_after(msg) or (15 * (attempt + 1))
                print(f"  [rate-limit] waiting {wait:.0f}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
                continue
            raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_agent_reply(version: PromptVersion, history: list[dict], extra_system: str = "") -> str:
    cfg = get_config()
    system = build_system_prompt(version)
    if extra_system:
        system = f"{system}\n\n{extra_system}"
    messages = history if history else [{"role": "user", "content": "[call connected]"}]

    if os.environ.get("OPENROUTER_API_KEY"):
        return _call_openrouter(system, messages, cfg.model, cfg.max_tokens)
    return _call_groq(system, messages, cfg.model, cfg.max_tokens)
