"""
voice_demo.py — Voice layer demo using gTTS (text-to-speech) and SpeechRecognition (STT).

Two modes:
  1. --tts --conversation-id N    → replay a saved conversation with TTS
  2. --live --version v3          → live mic loop with the agent

Usage:
    python -m src.voice_demo --tts --conversation-id 12
    python -m src.voice_demo --live --version v3

NOTE: This uses free demo-grade TTS/STT. Production voice needs low-latency streaming.
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from gtts import gTTS
    import speech_recognition as sr
except ImportError:
    print("ERROR: Voice dependencies not installed.", file=sys.stderr)
    print("Install with: pip install gTTS SpeechRecognition PyAudio", file=sys.stderr)
    sys.exit(1)

from src import db
from src.agent import get_agent_reply, get_config

ROOT = Path(__file__).parent.parent
AUDIO_DIR = ROOT / "outputs" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# TTS playback mode
# ---------------------------------------------------------------------------
def replay_conversation_tts(conversation_id: int) -> None:
    """
    Load a saved conversation from SQLite and convert each turn to an MP3 with gTTS.
    Saves files to outputs/audio/.
    """
    conv = db.get_conversation(conversation_id)
    if not conv:
        print(f"Conversation {conversation_id} not found.", file=sys.stderr)
        sys.exit(1)

    turns = db.get_turns(conversation_id)
    if not turns:
        print(f"No turns found for conversation {conversation_id}.", file=sys.stderr)
        sys.exit(1)

    print(f"Replaying conversation {conversation_id} ({len(turns)} turns) with TTS...")
    print(f"Audio files will be saved to: {AUDIO_DIR}\n")

    for t in turns:
        idx, speaker, text = t["turn_idx"], t["speaker"], t["text"]
        print(f"[{idx}] {speaker.upper()}: {text}")

        # Choose language: detect if mostly Hindi/Hinglish vs English
        # Simple heuristic: if Devanagari or common Hindi words, use 'hi', else 'en'
        lang = "hi" if any(c in text for c in ["है", "हूँ", "का", "aap", "kya", "haan"]) else "en"

        # Use different TLD for agent vs customer (slight accent difference)
        tld = "co.in" if speaker == "agent" else "com"

        try:
            tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
            filename = AUDIO_DIR / f"conv{conversation_id}_turn{idx:02d}_{speaker}.mp3"
            tts.save(str(filename))
            print(f"  → Saved: {filename.name}")
        except Exception as exc:
            print(f"  ⚠ TTS error: {exc}", file=sys.stderr)

        time.sleep(0.5)

    print(f"\nDone. {len(turns)} audio files saved to {AUDIO_DIR}")


# ---------------------------------------------------------------------------
# Live mic mode
# ---------------------------------------------------------------------------
def live_voice_loop(version: str) -> None:
    """
    Live mic loop: listen to user, send to agent, speak reply, repeat.
    Stops after 8 turns or when user says "bye".
    Logs conversation to SQLite with persona='human_live'.
    """
    cfg = get_config()
    recognizer = sr.Recognizer()

    print(f"Starting live voice demo with prompt {version}...")
    print("Speak clearly into your microphone. Say 'bye' or 'goodbye' to end.\n")

    # Test mic availability
    try:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise... (stay quiet for 2 seconds)")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Ready! Listening...\n")
    except OSError as exc:
        print(f"ERROR: Microphone not available: {exc}", file=sys.stderr)
        print("Make sure PyAudio is installed and a microphone is connected.", file=sys.stderr)
        sys.exit(1)

    history = []
    turns_log = []
    turn_idx = 0

    # Agent opens the call
    agent_reply = get_agent_reply(version=version, history=history)
    history.append({"role": "assistant", "content": agent_reply})
    turns_log.append({"turn_idx": turn_idx, "speaker": "agent", "text": agent_reply})
    turn_idx += 1

    print(f"[Agent] {agent_reply}")
    _speak(agent_reply)

    # Loop: listen → respond
    for _ in range(cfg.max_turns - 1):
        user_text = _listen(recognizer)
        if not user_text:
            print("  (no speech detected, try again)")
            continue

        print(f"[You]   {user_text}")
        history.append({"role": "user", "content": user_text})
        turns_log.append({"turn_idx": turn_idx, "speaker": "customer", "text": user_text})
        turn_idx += 1

        # Check for exit
        if any(word in user_text.lower() for word in ["bye", "goodbye", "end call", "hang up"]):
            print("\nCall ended by user.")
            break

        # Agent responds
        agent_reply = get_agent_reply(version=version, history=history)
        history.append({"role": "assistant", "content": agent_reply})
        turns_log.append({"turn_idx": turn_idx, "speaker": "agent", "text": agent_reply})
        turn_idx += 1

        print(f"[Agent] {agent_reply}")
        _speak(agent_reply)

    # Save to DB
    num_agent_turns = sum(1 for t in turns_log if t["speaker"] == "agent")
    conv_id = db.insert_conversation(
        prompt_version=version,
        persona="human_live",
        outcome="dropped",  # default for live calls
        num_turns=num_agent_turns,
    )
    db.insert_turns_bulk(conv_id, turns_log)
    print(f"\nConversation saved to database with ID {conv_id}.")


def _listen(recognizer: sr.Recognizer) -> str:
    """
    Listen to the mic and return transcribed text.
    Returns empty string on error.
    """
    try:
        with sr.Microphone() as source:
            print("\n[Listening...]")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)

        # Try Hindi first, fall back to English
        try:
            text = recognizer.recognize_google(audio, language="hi-IN")
            return text
        except sr.UnknownValueError:
            # Retry with English
            text = recognizer.recognize_google(audio, language="en-IN")
            return text

    except sr.WaitTimeoutError:
        print("  (timeout — no speech detected)")
        return ""
    except sr.UnknownValueError:
        print("  (could not understand audio)")
        return ""
    except sr.RequestError as exc:
        print(f"  (Google STT error: {exc})", file=sys.stderr)
        return ""
    except Exception as exc:
        print(f"  (unexpected error: {exc})", file=sys.stderr)
        return ""


def _speak(text: str) -> None:
    """
    Speak the text using gTTS.
    Detects language heuristically and plays the audio.
    """
    lang = "hi" if any(c in text for c in ["है", "हूँ", "का", "aap", "kya", "haan"]) else "en"

    try:
        tts = gTTS(text=text, lang=lang, tld="co.in", slow=False)
        tmp_file = AUDIO_DIR / "temp_reply.mp3"
        tts.save(str(tmp_file))

        # Play the file (platform-specific)
        # On Windows, use start; on Mac, use afplay; on Linux, use mpg123 or ffplay
        if sys.platform == "win32":
            os.system(f'start /min "" "{tmp_file}"')
        elif sys.platform == "darwin":
            os.system(f'afplay "{tmp_file}"')
        else:
            os.system(f'mpg123 -q "{tmp_file}" 2>/dev/null || ffplay -nodisp -autoexit -loglevel quiet "{tmp_file}"')

        # Give time for playback (rough estimate)
        time.sleep(len(text.split()) * 0.4)

    except Exception as exc:
        print(f"  ⚠ TTS error: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Voice demo: replay saved conversations or run live mic loop."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tts", action="store_true",
        help="TTS mode: replay a saved conversation with text-to-speech."
    )
    group.add_argument(
        "--live", action="store_true",
        help="Live mode: mic loop with the agent."
    )

    parser.add_argument(
        "--conversation-id", type=int, default=None,
        help="(TTS mode) Conversation ID to replay."
    )
    parser.add_argument(
        "--version", choices=["v1", "v2", "v3"], default="v3",
        help="(Live mode) Prompt version to use."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.tts:
        if not args.conversation_id:
            print("ERROR: --conversation-id is required for --tts mode.", file=sys.stderr)
            sys.exit(1)
        replay_conversation_tts(args.conversation_id)
    elif args.live:
        db.init_db()
        live_voice_loop(args.version)
