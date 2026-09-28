#!/usr/bin/env python3
"""
Print a single conversation in full detail.

Usage:
    python check_conv.py <conversation_id>
"""

import sys
from src import db


def print_conversation(conv_id: int) -> None:
    db.init_db()

    conv = db.get_conversation(conv_id)
    if not conv:
        print(f"Conversation {conv_id} not found.")
        sys.exit(1)

    conv = dict(conv)
    print(f"\n{'='*80}")
    print(f"Conversation #{conv['id']}  |  version={conv['prompt_version']}  "
          f"|  persona={conv['persona']}  |  outcome={conv['outcome']}  "
          f"|  agent_turns={conv['num_turns']}")
    print(f"{'='*80}\n")

    turns = db.get_turns(conv_id)
    for t in turns:
        t = dict(t)
        label = "AGENT   " if t["speaker"] == "agent" else "CUSTOMER"
        char_count = len(t["text"])
        word_count = len(t["text"].split())
        print(f"[{t['turn_idx']:2d}] {label}  ({word_count} words, {char_count} chars)")
        print(f"      {t['text']}\n")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_conv.py <conversation_id>")
        sys.exit(1)
    print_conversation(int(sys.argv[1]))
