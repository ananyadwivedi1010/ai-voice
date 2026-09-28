"""
customer_sim.py — Simulates a customer using Groq playing a persona.

The customer simulator receives the full conversation history and responds
as the persona character would — in 1-2 short spoken-style sentences,
staying firmly in character.
"""

import json
import os
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

ROOT = Path(__file__).parent.parent
PERSONAS_PATH = ROOT / "personas.json"
CONFIG_PATH = ROOT / "config.json"

# ---------------------------------------------------------------------------
# Persona loader
# ---------------------------------------------------------------------------
def load_personas() -> dict[str, dict]:
    """Return a mapping of persona_id -> persona dict."""
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        personas = json.load(f)
    return {p["id"]: p for p in personas}


def get_persona(persona_id: str) -> dict:
    """Fetch a single persona by id. Raises KeyError if not found."""
    personas = load_personas()
    if persona_id not in personas:
        raise KeyError(
            f"Persona '{persona_id}' not found. "
            f"Available: {list(personas.keys())}"
        )
    return personas[persona_id]


# ---------------------------------------------------------------------------
# System prompt for the customer simulator
# ---------------------------------------------------------------------------
def _build_customer_system(persona: dict) -> str:
    return f"""You are roleplaying as a customer on an outbound phone call from a loan company.

CHARACTER:
  Name: {persona["name"]}
  Personality: {persona["description"]}
  Language style: {persona["language_style"]}

STRICT RULES:
- Stay completely in character. Do not break the fourth wall.
- Reply in 1-2 short sentences only — the length of a real spoken phone response.
- Use the natural language style described above (Hindi / English / Hinglish as fits the character).
- Do NOT use bullet points, lists, or markdown. This is a phone call.
- React realistically: you are a real person receiving a cold call, not a helpful assistant.
- If you decide to end the call, say something natural like "theek hai, rakhta hoon" or "I'm hanging up".
- NEVER signal that you are an AI or that this is a simulation.
"""


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------
import re as _re


def _strip_reasoning(text: str) -> str:
    """Strip <think>...</think> blocks from qwen model output."""
    cleaned = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL)
    return cleaned.strip()


@retry(
    retry=retry_if_exception_type((Exception,)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_groq(
    client: Groq,
    system: str,
    messages: list[dict],
    max_tokens: int,
    model: str,
) -> str:
    """Call Groq API."""
    full_messages = [{"role": "system", "content": system}] + messages
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=full_messages,
    )
    text = _strip_reasoning(response.choices[0].message.content.strip())
    if not text or len(text) < 2:
        raise ValueError("Empty response from API")
    return text


# ---------------------------------------------------------------------------
# Fallback responses (used if LLM fails)
# ---------------------------------------------------------------------------
_FALLBACK_RESPONSES = {
    "skeptical": [
        "Nahi, mujhe aapke baare mein nahi pata. Ye fraud toh nahi hai?",
        "Kaun ho aap? Mujhe verified ID dikhao pehle.",
        "Theek hai, par pehle aap apne aap ko prove karo.",
    ],
    "busy": [
        "Abhi bahut busy hoon, baad mein baat karte hain.",
        "Yaar, mera meeting chal raha hai. Later call karna.",
        "Time nahi hai abhi. Kya zaroor hai?",
    ],
    "confused": [
        "Matlab kya ho raha hai? Samajh nahi aaya.",
        "Ye loan kya hota hai? Explain karo.",
        "Aap kya bol rahe ho? Phir se kaho.",
    ],
    "price_sensitive": [
        "Pehle interest rate batao, kitna percentage hai?",
        "Doosri company mein kam mil gaya. Tumhara kya offer hai?",
        "Kitna monthly payment hoga? Exact figure batao.",
    ],
    "code_switcher": [
        "Yaar, details de de. Kya offer hai?",
        "Okay okay, basically bolo na—kya benefits milenge?",
        "Chill, bata na bhai—ye kaise kaam karega?",
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_customer_reply(
    persona_id: str,
    history: list[dict],
    model: str,
    max_tokens: int = 100,
) -> str:
    """
    Simulate the customer's next reply for a given persona.

    Args:
        persona_id:  One of the ids defined in personas.json.
        history:     Conversation so far as messages.
        model:       Groq model name from config.
        max_tokens:  Max tokens for the customer reply.

    Returns:
        The customer's next spoken reply as a plain string.
    """
    persona = get_persona(persona_id)
    system = _build_customer_system(persona)

    # On the very first customer turn, use opening line
    if not history:
        return persona.get("opening_line", "Haan, kaun bol raha hai?")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")
    client = Groq(api_key=api_key)

    # Try LLM call first
    try:
        return _call_groq(
            client=client,
            system=system,
            messages=history,
            max_tokens=max_tokens,
            model=model,
        )
    except Exception as e:
        # Fallback to realistic responses if LLM fails
        import random
        fallback_list = _FALLBACK_RESPONSES.get(persona_id, ["Haan, samajh gaya."])
        return random.choice(fallback_list)
