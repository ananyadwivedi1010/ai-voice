"""
evaluate.py — Rule-based checks + LLM judge for scored conversations.

Usage:
    python -m src.evaluate                   # score all unscored conversations
    python -m src.evaluate --version v3      # score only v3 conversations
    python -m src.evaluate --sample          # score + print 10 random conversations
    python -m src.evaluate --conversation-id 42  # score a single conversation
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src import db
from src.agent import get_config

load_dotenv()

ROOT = Path(__file__).parent.parent
PROMPTS_DIR = ROOT / "prompts"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule-based checks
# ---------------------------------------------------------------------------

_MARKDOWN_PATTERNS = re.compile(
    r"(\*{1,3}[^*]+\*{1,3}"
    r"|\#{1,6}\s"
    r"|^\s*[-•]\s"
    r"|^\s*\d+\.\s"
    r"|:{2,}"
    r"|\bemoji\b"
    r"|[\U0001F300-\U0001FFFF]"
    r")",
    re.MULTILINE,
)

_COMPLIANCE_PATTERNS = re.compile(
    r"(guaranteed|guarantee|approved|approval|100%|assured|confirm"
    r"|\d+\.?\d*\s*%"
    r"|interest\s*rate\s*(is|=|:)"
    r"|rate\s*(is|=|:)\s*\d)",
    re.IGNORECASE,
)

_QUALIFICATION_KEYWORDS = {
    "employment_type": re.compile(
        r"\b(job|salaried|salary|employed|employment|business|self.?employed"
        r"|naukri|kaam|profession|kya karte hain|kya karte ho)\b",
        re.IGNORECASE,
    ),
    "monthly_income": re.compile(
        r"\b(income|salary|monthly|mahina|kitna kamate|earning|takka|lakh|hazaar"
        r"|kitni aay|income kitni)\b",
        re.IGNORECASE,
    ),
    "loan_amount": re.compile(
        r"\b(loan amount|kitna loan|kitne ka loan|how much loan|amount|rashi"
        r"|kitne chahiye|kitni zaroorat)\b",
        re.IGNORECASE,
    ),
}

# v2 compliance: flag if any reply exceeds 35 words OR contains brackets / "e.g."
_V2_BRACKET_PATTERN = re.compile(r"[()]|e\.g\.", re.IGNORECASE)


def check_qualification_complete(agent_turns: list[str]) -> bool:
    """Return True if all 3 qualification fields were covered in agent turns."""
    combined = " ".join(agent_turns)
    return all(pat.search(combined) for pat in _QUALIFICATION_KEYWORDS.values())


def check_avg_reply_words(agent_turns: list[str]) -> float:
    """
    Average word count across AGENT turns only.
    Pass only the agent turn texts — do NOT include customer turns.
    """
    if not agent_turns:
        return 0.0
    word_counts = [len(t.split()) for t in agent_turns]
    return sum(word_counts) / len(agent_turns)


def check_max_reply_words(agent_turns: list[str]) -> int:
    """Maximum word count in any single agent turn."""
    if not agent_turns:
        return 0
    return max(len(t.split()) for t in agent_turns)


def check_v2_word_limit(agent_turns: list[str]) -> bool:
    """
    Return True (= compliance FAILURE) if any agent turn:
      - exceeds 35 words, OR
      - contains parentheses '(' / ')' or the string 'e.g.'
    Used to flag v2 prompt rule violations.
    """
    for turn in agent_turns:
        if len(turn.split()) > 35:
            return True
        if _V2_BRACKET_PATTERN.search(turn):
            return True
    return False


def check_has_markdown_or_list(agent_turns: list[str]) -> bool:
    """Return True if any agent turn contains markdown / list formatting."""
    return any(_MARKDOWN_PATTERNS.search(t) for t in agent_turns)


def check_compliance_fail(agent_turns: list[str]) -> bool:
    """Return True if any agent turn contains a forbidden phrase."""
    return any(_COMPLIANCE_PATTERNS.search(t) for t in agent_turns)


def run_rule_checks(agent_turns: list[str], version: str = "") -> dict:
    """
    Run all rule-based checks on agent turns.
    Returns a partial score dict (without LLM judge fields).
    `version` is used to decide whether to run the v2 word-limit check.
    """
    result = {
        "qualification_complete": int(check_qualification_complete(agent_turns)),
        "avg_reply_words": check_avg_reply_words(agent_turns),
        "max_reply_words": check_max_reply_words(agent_turns),
        "has_markdown_or_list": int(check_has_markdown_or_list(agent_turns)),
        "compliance_fail": int(check_compliance_fail(agent_turns)),
    }
    # v2-specific: flag word-limit / bracket violations as compliance_fail
    if version == "v2" and check_v2_word_limit(agent_turns):
        result["compliance_fail"] = 1
    return result


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

def _build_transcript(turns: list) -> str:
    lines = []
    for t in turns:
        speaker = "Agent (Annie)" if t["speaker"] == "agent" else "Customer"
        lines.append(f"{speaker}: {t['text']}")
    return "\n".join(lines)


def _load_judge_prompt() -> str:
    path = PROMPTS_DIR / "judge.txt"
    return path.read_text(encoding="utf-8")


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_judge(client: Groq, model: str, prompt: str) -> str:
    """Call Groq and return raw text."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def _parse_judge_response(raw: str) -> dict | None:
    """
    Attempt to extract a JSON object from the judge's raw reply.
    Strips code fences, retries once if the first parse fails.
    Returns None if parsing fails after both attempts.
    """
    # Strip triple-backtick code fences (```json ... ``` or ``` ... ```)
    cleaned = raw
    for fence_variant in ("```json", "```"):
        if fence_variant in cleaned:
            parts = cleaned.split(fence_variant)
            # Take the content between the first pair of fences
            if len(parts) >= 3:
                cleaned = parts[1]
            elif len(parts) == 2:
                cleaned = parts[1].split("```")[0]
            break

    cleaned = cleaned.strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract the first {...} block and try again
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as exc:
            logger.warning("LLM judge JSON parse error after extraction: %s | raw=%r", exc, raw[:200])

    return None


def run_llm_judge(turns: list, cfg) -> dict:
    """
    Call the LLM judge for one conversation.
    Returns dict: language_match, objection_handled, judge_notes.
    Stores NULL (None) — not 0 — when the judge fails.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    client = Groq(api_key=api_key)
    judge_template = _load_judge_prompt()
    transcript = _build_transcript(turns)

    prompt = judge_template.format(
        transcript=transcript,
        agent_name=cfg.agent_name,
        company=cfg.company,
    )

    # --- API call ---
    try:
        raw = _call_judge(client=client, model=cfg.model, prompt=prompt)
    except Exception as exc:
        logger.error("Judge API call failed for conversation: %s", exc)
        return {
            "language_match": None,
            "objection_handled": None,
            "judge_notes": f"api_error: {str(exc)[:80]}",
        }

    logger.debug("Judge raw response: %r", raw[:300])

    # --- JSON parse ---
    result = _parse_judge_response(raw)
    if result is None:
        logger.warning("Judge returned unparseable response: %r", raw[:200])
        return {
            "language_match": None,
            "objection_handled": None,
            "judge_notes": f"parse_error: {raw[:80]}",
        }

    # Normalise values
    lang_match = result.get("language_match")
    if lang_match is not None:
        try:
            lang_match = int(lang_match)
        except (ValueError, TypeError):
            lang_match = None

    obj_handled = result.get("objection_handled")
    if obj_handled is not None:
        obj_handled = str(obj_handled)

    return {
        "language_match": lang_match,
        "objection_handled": obj_handled,
        "judge_notes": result.get("notes", ""),
    }


# ---------------------------------------------------------------------------
# Score a single conversation
# ---------------------------------------------------------------------------

def score_conversation(conv_row, verbose: bool = False) -> dict:
    """Compute and persist the full score for one conversation row."""
    cfg = get_config()
    conv_id = conv_row["id"]
    version = conv_row["prompt_version"]
    turns = db.get_turns(conv_id)

    agent_texts = [dict(t)["text"] for t in turns if t["speaker"] == "agent"]
    all_turns_dicts = [dict(t) for t in turns]

    rule_scores = run_rule_checks(agent_texts, version=version)
    judge_scores = run_llm_judge(all_turns_dicts, cfg)

    score = {**rule_scores, **judge_scores}
    db.insert_score(conv_id, score)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Conversation {conv_id} | version={version} | persona={conv_row['persona']}")
        print(f"Outcome: {conv_row['outcome']} | Turns: {conv_row['num_turns']}")
        print("-" * 60)
        for t in all_turns_dicts:
            speaker = "Annie  " if t["speaker"] == "agent" else "Customer"
            print(f"  [{t['turn_idx']}] {speaker}: {t['text']}")
        print("-" * 60)
        print(f"  qual_complete:      {score['qualification_complete']}")
        print(f"  avg_reply_words:    {score['avg_reply_words']:.1f}")
        print(f"  max_reply_words:    {score['max_reply_words']}")
        print(f"  has_markdown:       {bool(score['has_markdown_or_list'])}")
        print(f"  compliance_fail:    {bool(score['compliance_fail'])}")
        print(f"  language_match:     {score['language_match']}")
        print(f"  objection_handled:  {score['objection_handled']}")
        print(f"  judge_notes:        {score['judge_notes']}")

    return score


# ---------------------------------------------------------------------------
# Batch scorer
# ---------------------------------------------------------------------------

def score_batch(
    version: str | None = None,
    conversation_ids: list[int] | None = None,
    sample: bool = False,
    verbose: bool = False,
) -> None:
    """
    Score every conversation that does not yet have a scores row.
    Errors are logged but do not abort the batch.
    """
    db.init_db()

    if conversation_ids:
        rows = [db.get_conversation(cid) for cid in conversation_ids]
        rows = [r for r in rows if r is not None]
    else:
        rows = db.get_unscored_conversations()
        if version:
            rows = [r for r in rows if r["prompt_version"] == version]

    if not rows:
        print("No unscored conversations found.")
        return

    print(f"Scoring {len(rows)} conversation(s)...")
    failed = 0
    for i, row in enumerate(rows, 1):
        print(f"  [{i}/{len(rows)}] conv_id={row['id']} "
              f"persona={row['persona']} version={row['prompt_version']}")
        try:
            score_conversation(row, verbose=verbose)
        except Exception as exc:
            failed += 1
            logger.error("Failed to score conv %s: %s", row['id'], exc)

    print(f"Scoring complete. {len(rows) - failed} succeeded, {failed} failed.")

    if sample:
        _print_sample(n=10)


def _print_sample(n: int = 10) -> None:
    all_rows = db.get_conversations_with_scores()
    sample = random.sample(list(all_rows), min(n, len(all_rows)))

    print(f"\n{'='*60}")
    print(f"SAMPLE: {len(sample)} random conversations with scores")
    print(f"{'='*60}")

    for row in sample:
        row = dict(row)
        turns = db.get_turns(row["id"])
        print(f"\nConversation {row['id']} | {row['prompt_version']} | "
              f"{row['persona']} | {row['outcome']}")
        for t in turns:
            speaker = "Annie  " if t["speaker"] == "agent" else "Customer"
            print(f"  [{t['turn_idx']}] {speaker}: {t['text']}")
        avg = row.get("avg_reply_words") or 0.0
        print(f"  Scores → qual={row['qualification_complete']} | "
              f"words={avg:.1f} | "
              f"md={row['has_markdown_or_list']} | "
              f"compliance_fail={row['compliance_fail']} | "
              f"lang_match={row['language_match']} | "
              f"obj_handled={row['objection_handled']}")
        print(f"  Judge: {row['judge_notes']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate conversations with rule checks and LLM judge."
    )
    parser.add_argument(
        "--version", choices=["v1", "v2", "v3"], default=None,
        help="Only evaluate conversations from this prompt version."
    )
    parser.add_argument(
        "--conversation-id", type=int, default=None,
        help="Score a single conversation by id."
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="After scoring, print 10 random conversations with scores."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full transcript and scores for each evaluated conversation."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    conv_ids = [args.conversation_id] if args.conversation_id else None
    score_batch(
        version=args.version,
        conversation_ids=conv_ids,
        sample=args.sample,
        verbose=args.verbose,
    )
