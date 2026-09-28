"""
simulate.py — Runs N conversations per persona and logs everything to SQLite.

Usage:
    python -m src.simulate --version v3 --n 5
    python -m src.simulate --version v1 --n 2 --personas skeptical busy
"""

import argparse
import sys
import time
from pathlib import Path

from src.agent import get_agent_reply, get_config
from src.customer_sim import get_customer_reply, load_personas
from src import db

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Outcome detection
# ---------------------------------------------------------------------------

# Phrases that indicate a callback time was set or confirmed
_CALLBACK_TIME_PATTERNS = [
    "kal subah", "kal shaam", "kal dopahar", "parso", "tomorrow",
    "aaj shaam", "3 baje", "4 baje", "5 baje", "6 baje", "7 baje",
    "8 baje", "9 baje", "10 baje", "11 baje", "12 baje",
    "baje call", "baje pe", "call kar dena", "call karna",
    "call back", "callback", "hamare specialist", "hamare team",
    "number note", "note kar", "schedule", "book kar",
]

# Agent phrases that confirm a callback is booked
_AGENT_CONFIRMS_CALLBACK = [
    "call karenge", "call karungi", "call karunga",
    "call kar lenge", "number note kar liya", "note kar liya",
    "specialist call karenge", "team call karenge",
    "kal subah", "kal shaam", "tomorrow", "perfect",
    "bilkul theek hai", "zaroor", "noted",
]

# Customer phrases that clearly refuse further contact
_HARD_REFUSAL = [
    "dobara mat call", "dobara call mat", "mat karna call",
    "band karo", "remove my number", "don't call again",
    "not interested", "nahi chahiye", "mujhe nahi chahiye",
    "please remove", "block kar do", "disturb mat",
]

# Customer/agent phrases that signal natural end without booking
_POLITE_END = [
    "have a good day", "take care", "apna khayal rakhiye",
    "shukriya aapka", "dhanyavaad", "thank you for your time",
    "koi baat nahi", "samajh aata hai",
]


def _detect_outcome(agent_turns: list[str], customer_turns: list[str]) -> str:
    """
    Infer the conversation outcome from the full transcript.

    Returns one of:
      'callback_booked' — a callback time was agreed or the agent confirmed one
      'declined'        — the customer clearly refused further contact
      'dropped'         — hit max turns or ended without a clear close
    """
    agent_text = " ".join(agent_turns).lower()
    customer_text = " ".join(customer_turns).lower()

    # --- callback_booked ---
    # Either:
    #   (a) customer mentions a callback time AND agent acknowledges it, OR
    #   (b) agent books a time AND customer gives any positive signal
    customer_mentions_time = any(p in customer_text for p in _CALLBACK_TIME_PATTERNS)
    agent_confirms = any(p in agent_text for p in _AGENT_CONFIRMS_CALLBACK)

    # Broad customer agreement signals (much wider than before)
    _CUSTOMER_AGREES = [
        "theek hai", "thik hai", "theek", "bilkul", "haan", "ha",
        "okay", "ok", "sure", "achha", "accha", "acha", "sahi",
        "zaroor", "thanks", "thank you", "shukriya", "great",
    ]
    customer_agrees = any(p in customer_text for p in _CUSTOMER_AGREES)

    if customer_mentions_time and agent_confirms:
        return "callback_booked"
    if agent_confirms and customer_agrees:
        return "callback_booked"

    # --- declined ---
    # Customer explicitly refused or asked not to be called again
    if any(p in customer_text for p in _HARD_REFUSAL):
        return "declined"

    # Agent ended politely without any callback indication
    agent_ended_politely = any(p in agent_text for p in _POLITE_END)
    agent_has_callback = any(p in agent_text for p in _AGENT_CONFIRMS_CALLBACK)
    if agent_ended_politely and not agent_has_callback:
        return "declined"

    # --- dropped ---
    return "dropped"


def _is_callback_confirmed(agent_reply: str, customer_reply: str) -> bool:
    """Return True if this exchange looks like a callback was just confirmed."""
    agent_lower = agent_reply.lower()
    customer_lower = customer_reply.lower()

    agent_books = any(p in agent_lower for p in _AGENT_CONFIRMS_CALLBACK)
    time_mentioned = any(p in agent_lower or p in customer_lower
                         for p in ["kal subah", "kal shaam", "tomorrow",
                                   "baje", "schedule", "note kar"])
    customer_ok = any(p in customer_lower for p in [
        "theek hai", "thik hai", "bilkul", "haan", "okay", "ok",
        "sure", "achha", "zaroor", "thanks", "shukriya",
    ])

    return agent_books and (time_mentioned or customer_ok)


# ---------------------------------------------------------------------------
# Single conversation runner
# ---------------------------------------------------------------------------
def run_conversation(
    version: str,
    persona_id: str,
    max_turns: int,
    model: str,
    max_tokens_agent: int,
    max_tokens_customer: int,
    verbose: bool = False,
) -> int:
    """
    Run one agent↔customer conversation and persist it to SQLite.

    max_turns counts agent turns (each agent turn + customer reply = 1 turn).
    Returns the conversation_id.
    """
    db.init_db()

    agent_history: list[dict] = []
    customer_history: list[dict] = []

    turns_log: list[dict] = []
    agent_texts: list[str] = []
    customer_texts: list[str] = []

    turn_idx = 0          # global index across both speakers
    agent_turn_count = 0  # counts only agent turns (= num_turns stored in DB)

    for _ in range(max_turns):
        # ---- Agent turn ----
        agent_reply = get_agent_reply(version=version, history=agent_history)
        agent_texts.append(agent_reply)
        turns_log.append({"turn_idx": turn_idx, "speaker": "agent", "text": agent_reply})
        turn_idx += 1
        agent_turn_count += 1

        if verbose:
            print(f"[Agent]    {agent_reply}")

        agent_history.append({"role": "assistant", "content": agent_reply})
        customer_history.append({"role": "user", "content": agent_reply})

        # Stop if agent said goodbye
        agent_lower = agent_reply.lower()
        if any(s in agent_lower for s in ["bye", "goodbye", "alvida",
                                           "take care", "have a good day",
                                           "apna khayal rakhiye"]):
            if verbose:
                print("[Sim] Agent ended the call.\n")
            break

        # ---- Customer turn ----
        customer_reply = get_customer_reply(
            persona_id=persona_id,
            history=customer_history,
            model=model,
            max_tokens=max_tokens_customer,
        )
        customer_texts.append(customer_reply)
        turns_log.append({"turn_idx": turn_idx, "speaker": "customer", "text": customer_reply})
        turn_idx += 1

        if verbose:
            print(f"[Customer] {customer_reply}\n")

        customer_history.append({"role": "assistant", "content": customer_reply})
        agent_history.append({"role": "user", "content": customer_reply})

        # Stop if customer said goodbye / hung up
        cust_lower = customer_reply.lower()
        if any(s in cust_lower for s in [
            "bye", "goodbye", "alvida", "rakhta hoon", "rakhti hoon",
            "hanging up", "band karo", "phone rakh",
        ]):
            if verbose:
                print("[Sim] Customer ended the call.\n")
            break

        # Stop as soon as a callback is confirmed — no repeated goodbyes
        if _is_callback_confirmed(agent_reply, customer_reply):
            if verbose:
                print("[Sim] Callback confirmed — ending loop.\n")
            break

        time.sleep(0.3)

    outcome = _detect_outcome(agent_texts, customer_texts)

    conversation_id = db.insert_conversation(
        prompt_version=version,
        persona=persona_id,
        outcome=outcome,
        num_turns=agent_turn_count,
    )
    db.insert_turns_bulk(conversation_id=conversation_id, turns=turns_log)

    if verbose:
        print(f"[Sim] Saved conversation {conversation_id} | outcome={outcome} "
              f"| agent_turns={agent_turn_count}\n")

    return conversation_id


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
def run_batch(
    version: str,
    n: int,
    persona_ids: list[str] | None = None,
    verbose: bool = False,
    sleep_between: float = 1.5,
) -> list[int]:
    """
    Run N conversations for each persona and return all conversation ids.

    Args:
        version:       Prompt version ('v1', 'v2', 'v3').
        n:             Number of conversations per persona.
        persona_ids:   Subset of persona ids to use. None = all personas.
        verbose:       Print each turn to stdout as the simulation runs.
        sleep_between: Seconds to wait between conversations (rate-limit buffer).
    """
    cfg = get_config()
    all_personas = load_personas()

    if persona_ids:
        invalid = [p for p in persona_ids if p not in all_personas]
        if invalid:
            raise ValueError(
                f"Unknown persona(s): {invalid}. Available: {list(all_personas.keys())}"
            )
        selected = {p: all_personas[p] for p in persona_ids}
    else:
        selected = all_personas

    conversation_ids: list[int] = []
    total = len(selected) * n

    print(f"Running {total} conversations "
          f"({len(selected)} personas × {n} each) with prompt {version} "
          f"[model={cfg.model}] ...")
    print("-" * 60)

    idx = 1
    for persona_id in selected:
        for run in range(1, n + 1):
            print(f"[{idx}/{total}] persona={persona_id}  run={run}/{n}")
            try:
                conv_id = run_conversation(
                    version=version,
                    persona_id=persona_id,
                    max_turns=cfg.max_turns,
                    model=cfg.model,
                    max_tokens_agent=cfg.max_tokens,
                    max_tokens_customer=cfg.max_tokens_customer,
                    verbose=verbose,
                )
                conversation_ids.append(conv_id)
                time.sleep(sleep_between)
            except Exception as exc:
                print(f"  ERROR on run {run} for persona {persona_id}: {exc}",
                      file=sys.stderr)
            idx += 1

    print("-" * 60)
    print(f"Done. {len(conversation_ids)}/{total} conversations saved to DB.")
    return conversation_ids


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simulated conversations between the sales agent and customer personas."
    )
    parser.add_argument(
        "--version", choices=["v1", "v2", "v3"], required=True,
        help="Prompt version to use."
    )
    parser.add_argument(
        "--n", type=int, default=5,
        help="Number of conversations per persona (default: 5)."
    )
    parser.add_argument(
        "--personas", nargs="+", default=None,
        metavar="PERSONA_ID",
        help="Subset of persona ids to simulate. Defaults to all 5 personas."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each turn to stdout as the simulation runs."
    )
    parser.add_argument(
        "--sleep", type=float, default=1.5,
        help="Sleep duration (seconds) between conversations (default: 1.5)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_batch(
        version=args.version,
        n=args.n,
        persona_ids=args.personas,
        verbose=args.verbose,
        sleep_between=args.sleep,
    )
