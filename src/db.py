"""
db.py — SQLite schema creation and helper functions.

Tables:
  conversations  — one row per simulated conversation
  turns          — one row per speaker turn within a conversation
  scores         — one row of evaluation metrics per conversation
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

# ---------------------------------------------------------------------------
# Database path — override via DB_PATH env var
# ---------------------------------------------------------------------------
_DEFAULT_DB = Path(__file__).parent.parent / "voice_agent.db"
DB_PATH = Path(os.environ.get("DB_PATH", str(_DEFAULT_DB)))


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------
@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection with row_factory set to Row."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")  # safe for concurrent reads
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_version  TEXT    NOT NULL,           -- 'v1', 'v2', 'v3'
    persona         TEXT    NOT NULL,           -- persona id from personas.json
    outcome         TEXT    NOT NULL            -- 'callback_booked' | 'declined' | 'dropped'
                    CHECK(outcome IN ('callback_booked', 'declined', 'dropped')),
    num_turns       INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL            -- ISO-8601 UTC
);

CREATE TABLE IF NOT EXISTS turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_idx        INTEGER NOT NULL,           -- 0-based index within the conversation
    speaker         TEXT    NOT NULL            -- 'agent' | 'customer'
                    CHECK(speaker IN ('agent', 'customer')),
    text            TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(conversation_id);

CREATE TABLE IF NOT EXISTS scores (
    conversation_id         INTEGER PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    qualification_complete  INTEGER NOT NULL DEFAULT 0,  -- 0 | 1
    avg_reply_words         REAL,                        -- avg words per agent turn
    max_reply_words         REAL,                        -- max words in any agent turn
    has_markdown_or_list    INTEGER NOT NULL DEFAULT 0,  -- 0 | 1
    compliance_fail         INTEGER NOT NULL DEFAULT 0,  -- 0 | 1 (forbidden phrases)
    language_match          INTEGER,                     -- 0 | 1 (LLM judge)
    objection_handled       TEXT,                        -- '0' | '1' | 'NA' (LLM judge)
    judge_notes             TEXT                         -- one sentence from judge
);
"""


def init_db() -> None:
    """Create all tables if they do not exist. Safe to call multiple times."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------
def insert_conversation(
    prompt_version: str,
    persona: str,
    outcome: str,
    num_turns: int,
) -> int:
    """Insert a conversation row and return the new id."""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO conversations (prompt_version, persona, outcome, num_turns, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (prompt_version, persona, outcome, num_turns, now),
        )
        return cur.lastrowid


def insert_turn(
    conversation_id: int,
    turn_idx: int,
    speaker: str,
    text: str,
) -> None:
    """Append a single turn to the turns table."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO turns (conversation_id, turn_idx, speaker, text)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, turn_idx, speaker, text),
        )


def insert_turns_bulk(conversation_id: int, turns: list[dict]) -> None:
    """
    Insert multiple turns at once.
    Each dict must have keys: turn_idx, speaker, text.
    """
    rows = [(conversation_id, t["turn_idx"], t["speaker"], t["text"]) for t in turns]
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO turns (conversation_id, turn_idx, speaker, text)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------
def insert_score(conversation_id: int, score: dict) -> None:
    """
    Upsert a scores row.
    Expected score keys (all optional except conversation_id):
      qualification_complete, avg_reply_words, max_reply_words, has_markdown_or_list,
      compliance_fail, language_match, objection_handled, judge_notes
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO scores
                (conversation_id, qualification_complete, avg_reply_words, max_reply_words,
                 has_markdown_or_list, compliance_fail,
                 language_match, objection_handled, judge_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                score.get("qualification_complete", 0),
                score.get("avg_reply_words"),
                score.get("max_reply_words"),
                score.get("has_markdown_or_list", 0),
                score.get("compliance_fail", 0),
                score.get("language_match"),
                score.get("objection_handled"),
                score.get("judge_notes"),
            ),
        )


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------
def get_conversation(conversation_id: int) -> sqlite3.Row | None:
    """Fetch a single conversation row by id."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()


def get_turns(conversation_id: int) -> list[sqlite3.Row]:
    """Fetch all turns for a conversation, ordered by turn_idx."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM turns WHERE conversation_id = ? ORDER BY turn_idx",
            (conversation_id,),
        ).fetchall()


def get_all_conversations() -> list[sqlite3.Row]:
    """Fetch all conversation rows ordered by creation time."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM conversations ORDER BY created_at DESC"
        ).fetchall()


def get_unscored_conversations() -> list[sqlite3.Row]:
    """Return conversations that do not yet have a scores row."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT c.* FROM conversations c
            LEFT JOIN scores s ON c.id = s.conversation_id
            WHERE s.conversation_id IS NULL
            ORDER BY c.created_at
            """
        ).fetchall()


def get_conversations_with_scores(version: str | None = None) -> list[sqlite3.Row]:
    """
    Return conversations joined with scores.
    Optionally filter by prompt_version.
    """
    query = """
        SELECT c.*, s.*
        FROM conversations c
        JOIN scores s ON c.id = s.conversation_id
    """
    params: list = []
    if version:
        query += " WHERE c.prompt_version = ?"
        params.append(version)
    query += " ORDER BY c.created_at"
    with get_connection() as conn:
        return conn.execute(query, params).fetchall()


def fetch_raw(sql: str, params: list | None = None) -> list[sqlite3.Row]:
    """Execute an arbitrary read-only SQL statement and return all rows."""
    with get_connection() as conn:
        return conn.execute(sql, params or []).fetchall()
