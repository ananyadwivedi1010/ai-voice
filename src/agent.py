"""
agent.py — Builds the system prompt for a given version and gets a reply from Groq.

Usage:
    from src.agent import AgentConfig, get_agent_reply

    cfg = AgentConfig.load()
    history = []
    reply = get_agent_reply(version="v3", history=history)
"""

import json
import os
import time
from pathlib import Path
from typing import Literal

from groq import Groq
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
PROMPTS_DIR = ROOT / "prompts"
CONFIG_PATH = ROOT / "config.json"

PromptVersion = Literal["v1", "v2", "v3"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class AgentConfig:
    """Holds scenario-level config loaded from config.json."""

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


# Module-level singleton so we load config once per process
_config: AgentConfig | None = None


def get_config() -> AgentConfig:
    global _config
    if _config is None:
        _config = AgentConfig.load()
    return _config


def reset_config() -> None:
    """Force config to be reloaded on next get_config() call (used in tests)."""
    global _config
    _config = None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def _load_prompt_template(version: PromptVersion) -> str:
    """Read the versioned prompt file from prompts/."""
    path = PROMPTS_DIR / f"agent_{version}.txt"
    return path.read_text(encoding="utf-8")


def build_system_prompt(version: PromptVersion) -> str:
    """
    Fill placeholders in the versioned prompt with values from config.json.
    Placeholders: {agent_name}, {company}, {product}, {goal}
    """
    cfg = get_config()
    template = _load_prompt_template(version)
    return template.format(
        agent_name=cfg.agent_name,
        company=cfg.company,
        product=cfg.product,
        goal=cfg.goal,
    )


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
def _get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Retry wrapper — handles transient API errors
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type((Exception,)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _strip_reasoning(text: str) -> str:
    """
    Remove <think>...</think> reasoning blocks that qwen/qwen3.8-27b
    sometimes prepends to its output before the actual reply.
    """
    import re
    # Strip any <think>...</think> block (greedy=False to handle multiple)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _call_groq(
    client: Groq,
    system: str,
    messages: list[dict],
    max_tokens: int,
    model: str,
) -> str:
    """Make the actual API call to Groq and return the text response."""
    full_messages = [{"role": "system", "content": system}] + messages

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=full_messages,
    )
    raw = response.choices[0].message.content.strip()
    return _strip_reasoning(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_agent_reply(
    version: PromptVersion,
    history: list[dict],
    extra_system: str = "",
) -> str:
    """
    Get the next agent reply given a conversation history.

    Args:
        version:      Prompt version — 'v1', 'v2', or 'v3'.
        history:      List of {"role": "user"|"assistant", "content": "..."} dicts
                      in OpenAI format. Can be empty for the opening turn.
        extra_system: Optional extra instructions appended to the system prompt
                      (used by voice_demo for live mode context).

    Returns:
        The agent's next reply as a plain string.
    """
    cfg = get_config()
    system = build_system_prompt(version)
    if extra_system:
        system = f"{system}\n\n{extra_system}"

    client = _get_client()

    # On the very first turn history is empty — send a minimal user seed
    # so the agent can produce its opening greeting.
    messages = history if history else [{"role": "user", "content": "[call connected]"}]

    return _call_groq(
        client=client,
        system=system,
        messages=messages,
        max_tokens=cfg.max_tokens,
        model=cfg.model,
    )
