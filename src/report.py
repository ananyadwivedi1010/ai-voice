"""
report.py — Generate comparison tables for v1 vs v2 vs v3.

Runs key SQL queries and prints a formatted table comparing prompt versions.

Usage:
    python -m src.report
"""

import sys
from pathlib import Path

import pandas as pd

from src import db

ROOT = Path(__file__).parent.parent
SQL_PATH = ROOT / "sql" / "analysis.sql"


def load_sql_queries() -> dict[str, str]:
    """
    Parse sql/analysis.sql and return a dict of {query_name: sql_text}.
    Expected format: comments starting with "-- QUERY:" mark each query.
    """
    if not SQL_PATH.exists():
        return {}

    text = SQL_PATH.read_text(encoding="utf-8")
    queries = {}
    current_name = None
    current_sql = []

    for line in text.split("\n"):
        if line.strip().startswith("-- QUERY:"):
            # Save previous query
            if current_name and current_sql:
                queries[current_name] = "\n".join(current_sql).strip()
            # Start new query
            current_name = line.split("-- QUERY:")[1].strip()
            current_sql = []
        elif current_name is not None:
            # Skip comment lines but keep SQL
            if not line.strip().startswith("--"):
                current_sql.append(line)

    # Save last query
    if current_name and current_sql:
        queries[current_name] = "\n".join(current_sql).strip()

    return queries


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def run_report() -> None:
    """Generate and print the full comparison report."""
    db.init_db()

    print_section("VOICE AGENT EVALUATION REPORT — v1 vs v2 vs v3")

    queries = load_sql_queries()

    # -----------------------------------------------------------------------
    # 1. Outcome distribution by version
    # -----------------------------------------------------------------------
    print_section("1. OUTCOME DISTRIBUTION")
    sql = queries.get("outcome_by_version", """
        SELECT
            prompt_version,
            outcome,
            COUNT(*) as count
        FROM conversations
        GROUP BY prompt_version, outcome
        ORDER BY prompt_version, outcome
    """)
    rows = db.fetch_raw(sql)
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        pivot = df.pivot_table(
            index="outcome", columns="prompt_version", values="count", fill_value=0, aggfunc="sum"
        )
        print(pivot.to_string())
    else:
        print("No data yet.")

    # -----------------------------------------------------------------------
    # 2. Compliance failure rate by version
    # -----------------------------------------------------------------------
    print_section("2. COMPLIANCE FAILURE RATE")
    sql = """
        SELECT
            c.prompt_version,
            ROUND(AVG(s.compliance_fail) * 100, 1) as compliance_fail_pct
        FROM conversations c
        JOIN scores s ON c.id = s.conversation_id
        GROUP BY c.prompt_version
        ORDER BY c.prompt_version
    """
    rows = db.fetch_raw(sql)
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        print(df.to_string(index=False))
    else:
        print("No data yet.")

    # -----------------------------------------------------------------------
    # 3. Average reply words and turns by version
    # -----------------------------------------------------------------------
    print_section("3. AVERAGE REPLY WORDS & TURNS")
    sql = queries.get("words_and_turns", """
        SELECT
            c.prompt_version,
            ROUND(AVG(s.avg_reply_words), 1) as avg_words_per_reply,
            ROUND(AVG(c.num_turns), 1) as avg_turns
        FROM conversations c
        JOIN scores s ON c.id = s.conversation_id
        GROUP BY c.prompt_version
        ORDER BY c.prompt_version
    """)
    rows = db.fetch_raw(sql)
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        print(df.to_string(index=False))
    else:
        print("No data yet.")

    # -----------------------------------------------------------------------
    # 4. Qualification completion rate
    # -----------------------------------------------------------------------
    print_section("4. QUALIFICATION COMPLETION RATE")
    sql = """
        SELECT
            c.prompt_version,
            ROUND(AVG(s.qualification_complete) * 100, 1) as qualification_complete_pct
        FROM conversations c
        JOIN scores s ON c.id = s.conversation_id
        GROUP BY c.prompt_version
        ORDER BY c.prompt_version
    """
    rows = db.fetch_raw(sql)
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        print(df.to_string(index=False))
    else:
        print("No data yet.")

    # -----------------------------------------------------------------------
    # 5. Language match rate (LLM judge)
    # -----------------------------------------------------------------------
    print_section("5. LANGUAGE MATCH RATE (LLM Judge)")
    sql = queries.get("language_match", """
        SELECT
            c.prompt_version,
            ROUND(AVG(s.language_match) * 100, 1) as language_match_pct
        FROM conversations c
        JOIN scores s ON c.id = s.conversation_id
        WHERE s.language_match IS NOT NULL
        GROUP BY c.prompt_version
        ORDER BY c.prompt_version
    """)
    rows = db.fetch_raw(sql)
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        print(df.to_string(index=False))
    else:
        print("No data yet.")

    # -----------------------------------------------------------------------
    # 6. Markdown/list formatting failure rate
    # -----------------------------------------------------------------------
    print_section("6. MARKDOWN/LIST FORMATTING FAILURE RATE")
    sql = """
        SELECT
            c.prompt_version,
            ROUND(AVG(s.has_markdown_or_list) * 100, 1) as markdown_fail_pct
        FROM conversations c
        JOIN scores s ON c.id = s.conversation_id
        GROUP BY c.prompt_version
        ORDER BY c.prompt_version
    """
    rows = db.fetch_raw(sql)
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        print(df.to_string(index=False))
    else:
        print("No data yet.")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print_section("SUMMARY TABLE — Key Metrics by Version")
    summary_sql = """
        SELECT
            c.prompt_version as version,
            COUNT(*) as total_conversations,
            ROUND(AVG(s.qualification_complete) * 100, 1) as qual_complete_pct,
            ROUND(AVG(s.avg_reply_words), 1) as avg_words,
            ROUND(AVG(s.compliance_fail) * 100, 1) as compliance_fail_pct,
            ROUND(AVG(s.has_markdown_or_list) * 100, 1) as markdown_fail_pct,
            ROUND(AVG(s.language_match) * 100, 1) as lang_match_pct,
            ROUND(AVG(c.num_turns), 1) as avg_turns
        FROM conversations c
        JOIN scores s ON c.id = s.conversation_id
        GROUP BY c.prompt_version
        ORDER BY c.prompt_version
    """
    rows = db.fetch_raw(summary_sql)
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        print(df.to_string(index=False))
        print("\nLegend:")
        print("  qual_complete_pct   → % of conversations where all 3 fields were asked")
        print("  avg_words           → average words per agent reply (target: <35)")
        print("  compliance_fail_pct → % using forbidden phrases (guaranteed, rates, etc.)")
        print("  markdown_fail_pct   → % using lists/markdown/emojis (bad for voice)")
        print("  lang_match_pct      → % matching customer's language (LLM judge)")
        print("  avg_turns           → average agent turns per conversation")
    else:
        print("No scored conversations found. Run simulate.py and evaluate.py first.")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        run_report()
    except Exception as exc:
        print(f"ERROR generating report: {exc}", file=sys.stderr)
        sys.exit(1)
