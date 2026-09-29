"""
customer_sim.py — Simulates a customer persona via OpenRouter/Groq.
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
PERSONAS_PATH = ROOT / "personas.json"


def load_personas() -> dict[str, dict]:
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        return {p["id"]: p for p in json.load(f)}


def get_persona(persona_id: str) -> dict:
    personas = load_personas()
    if persona_id not in personas:
        raise KeyError(f"Persona '{persona_id}' not found. Available: {list(personas.keys())}")
    return personas[persona_id]


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
- React realistically: you are a real person receiving a cold call.
- If you decide to end the call, say something like "theek hai, rakhta hoon" or "I'm hanging up".
- NEVER reveal you are an AI.
"""


def _strip_reasoning(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_retry_after(msg: str) -> float | None:
    m = re.search(r"try again in\s+(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s", msg)
    if m:
        return (int(m.group(1)) if m.group(1) else 0) * 60 + float(m.group(2)) + 2
    return None


_FALLBACK = {
    "skeptical": ["Nahi, fraud lagta hai.", "Kaun ho aap?"],
    "busy":  ["Abhi busy hoon.", "Meeting mein hoon, baad mein."],
    "confused": ["Samajh nahi aaya.", "Matlab kya hai?"],
    "price_sensitive": ["Interest rate kya hai?", "Kitna lagega?"],
    "code_switcher": ["Yaar bolo kya deal hai.", "Basically kya offer hai?"],
}


def get_customer_reply(persona_id: str, history: list[dict], model: str, max_tokens: int = 100) -> str:
    persona = get_persona(persona_id)

    if not history:
        return persona.get("opening_line", "Haan, kaun bol raha hai?")

    system = _build_customer_system(persona)
    full_messages = [{"role": "system", "content": system}] + history

    # --- OpenRouter ---
    or_key = os.environ.get("OPENROUTER_API_KEY")
    if or_key:
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
                        "Authorization": f"Bearer {or_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/voice-agent-eval",
                    },
                )
                resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
                text = _strip_reasoning((resp["choices"][0]["message"]["content"] or "").strip())
                if text and len(text) >= 2:
                    return text
            except urllib.error.HTTPError as exc:
                body = exc.read().decode()
                if exc.code == 429 and attempt < 4:
                    wait = _parse_retry_after(body) or (15 * (attempt + 1))
                    print(f"  [rate-limit] waiting {wait:.0f}s (attempt {attempt+1}/5)...")
                    time.sleep(wait)
                    continue
                break
            except Exception:
                break

    # --- Groq fallback ---
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
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
                    break
        except Exception:
            pass

    import random
    return random.choice(_FALLBACK.get(persona_id, ["Theek hai."]))
