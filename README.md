# Voice Agent Evaluation — Hinglish Sales Agent for QuickCash

**Author:** Ananya Dwivedi  
**Portfolio Project for:** Voice AI Prompt Engineer role at SquadStack.ai

---

## Overview

This project builds and evaluates **Riya**, a Hinglish voice-first sales agent that makes outbound calls for a fictional personal loan company, QuickCash. Riya qualifies leads on 3 criteria (employment type, monthly income, loan amount needed), then books a callback or politely ends the call.

The project includes:
- **3 prompt versions** (v1, v2, v3) with progressively sophisticated techniques
- **5 customer personas** simulated via LLM (skeptical, busy, confused, price-sensitive, code-switcher)
- **Rule-based + LLM evaluation** with strict compliance checks
- **SQLite database** for conversation logs and scores
- **Comparison report** showing v1 vs v2 vs v3 performance
- **FastAPI server** for production-like integration
- **Voice demo** with text-to-speech (gTTS) and speech-to-text (SpeechRecognition)

---

## Project Structure

```
voice-agent-eval/
├── prompts/
│   ├── base_template.txt       # Shared base prompt (placeholders)
│   ├── agent_v1.txt            # v1: Zero-shot + role prompting
│   ├── agent_v2.txt            # v2: Voice-first constraints
│   ├── agent_v3.txt            # v3: Few-shot + staged flow + guardrails
│   └── judge.txt               # LLM judge for language_match and objection_handled
├── src/
│   ├── agent.py                # Prompt builder and Claude API caller
│   ├── customer_sim.py         # LLM-based customer persona simulator
│   ├── simulate.py             # Run N conversations per persona
│   ├── evaluate.py             # Rule checks + LLM judge
│   ├── report.py               # v1 vs v2 vs v3 comparison table
│   ├── api.py                  # FastAPI server (POST /chat, GET /health)
│   ├── voice_demo.py           # TTS playback + live mic loop
│   └── db.py                   # SQLite schema and helpers
├── tests/
│   └── test_rules.py           # Pytest cases for rule-based checks
├── sql/
│   └── analysis.sql            # Analytical queries for report
├── outputs/
│   └── audio/                  # MP3 files from voice demo
├── config.json                 # Scenario config (agent name, model, etc.)
├── personas.json               # 5 customer personas
├── requirements.txt            # Python dependencies
├── Dockerfile                  # API container
├── .env.example                # Template for secrets
└── README.md                   # This file
```

---

## Setup

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt
```

**Note:** On Windows, PyAudio may require a prebuilt wheel. Download from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) if pip fails.

### 2. Configure Environment

```bash
# Copy the example and fill in your API key
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# Edit .env and add your Anthropic API key:
# ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Initialize Database

The database is auto-initialized on first use. To manually create tables:

```bash
python -c "from src.db import init_db; init_db()"
```

---

## How to Run

### Simulation

Run N conversations per persona for a given prompt version:

```bash
# Run 5 conversations per persona (25 total) with v3
python -m src.simulate --version v3 --n 5

# Run only skeptical and busy personas
python -m src.simulate --version v1 --n 3 --personas skeptical busy

# Verbose mode (print each turn)
python -m src.simulate --version v2 --n 2 --verbose
```

Conversations are saved to `voice_agent.db` (SQLite).

### Evaluation

Score conversations with rule-based checks + LLM judge:

```bash
# Score all unscored conversations
python -m src.evaluate

# Score only v3 conversations
python -m src.evaluate --version v3

# Score + print 10 random samples
python -m src.evaluate --sample

# Verbose mode (print full transcripts with scores)
python -m src.evaluate --verbose
```

### Report

Generate a v1 vs v2 vs v3 comparison table:

```bash
python -m src.report
```

Outputs:
- Outcome distribution
- Compliance failure rate
- Avg reply words & turns
- Qualification completion rate
- Language match rate (LLM judge)
- Markdown/list formatting failures

### API Server

Run the FastAPI server:

```bash
uvicorn src.api:app --reload --port 8000
```

Endpoints:
- `POST /chat` — Send version + history, get agent reply
- `GET /health` — Health check

Example request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-secret-key-here" \
  -d '{
    "version": "v3",
    "history": [
      {"role": "user", "content": "Haan, kaun hai?"}
    ]
  }'
```

### Docker

Build and run the API in a container:

```bash
docker build -t voice-agent-api .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-your-key-here voice-agent-api
```

### Voice Demo

#### Mode 1: Replay a saved conversation with TTS

```bash
# Convert conversation 12 to audio files
python -m src.voice_demo --tts --conversation-id 12
```

Audio files saved to `outputs/audio/`.

#### Mode 2: Live mic loop

```bash
# Live conversation with v3
python -m src.voice_demo --live --version v3
```

Speak into your microphone. Say "bye" to end. Conversation is saved to the database.

**Note:** This uses free demo-grade TTS/STT (gTTS + Google Speech). Production voice systems need low-latency streaming (e.g., Deepgram, AssemblyAI, ElevenLabs).

### Tests

Run pytest:

```bash
pytest tests/test_rules.py -v
```

---

## Prompt Versions

| Version | Techniques | Key Features |
|---------|-----------|-------------|
| **v1** | Zero-shot + Role Prompting | Basic persona definition, goal statement, and hard rules. No explicit voice constraints. |
| **v2** | Voice-First Constraints | 1-2 sentence replies, one question at a time, no markdown/lists, numbers spoken naturally ("paanch lakh"), mirror customer's language. |
| **v3** | Few-Shot + Staged Flow + Guardrails | 4-stage flow (opening → qualification → objection handling → close), 4 few-shot examples for common objections, explicit compliance reminders. |

### Metrics (to be filled after running experiments)

| Metric | v1 | v2 | v3 |
|--------|----|----|-------|
| Qualification Complete % | — | — | — |
| Avg Words per Reply | — | — | — |
| Compliance Fail % | — | — | — |
| Language Match % | — | — | — |
| Callback Booked % | — | — | — |

---

## Requirements → Design Mapping

How business requirements are enforced in prompts and evaluation:

1. **"Never promise approval"**  
   → Prompt rule: "Never promise loan approval or guaranteed eligibility."  
   → Eval check: `compliance_fail` regex matches "guaranteed", "approved", "100%".

2. **"Never quote interest rates"**  
   → Prompt rule: "Never quote, invent, or imply any specific interest rate."  
   → Eval check: `compliance_fail` regex matches "X% interest", "rate is Y".

3. **"Keep replies short for voice"**  
   → Prompt constraint (v2+): "Maximum 1-2 sentences per turn."  
   → Eval check: `avg_reply_words` should be < 35.

4. **"Mirror the customer's language"**  
   → Prompt constraint (v2+): "Match the language mix of whoever you are speaking to."  
   → Eval check: LLM judge scores `language_match` (0 or 1).

5. **"End gracefully if customer refuses"**  
   → Prompt rule: "If the customer says 'don't call again', thank them and end."  
   → Eval check: Manual review of `outcome='dropped'` conversations in sample.

---

## Failure Analysis

### Template 1: [Title]

**What failed:**  
_[Describe the issue — e.g., agent quoted a 9.5% rate, agent used bullet points, agent ignored customer's Hindi]_

**Evidence from logs:**  
_[Quote the offending turn(s) from a conversation, cite conv_id]_

**What I changed:**  
_[Describe the prompt edit or rule addition — e.g., added explicit "never use lists" rule, added few-shot example for rate question]_

---

### Template 2: [Title]

**What failed:**

**Evidence from logs:**

**What I changed:**

---

### Template 3: [Title]

**What failed:**

**Evidence from logs:**

**What I changed:**

---

## Limitations

1. **Simulated customers:**  
   All customer responses are generated by Claude playing a persona. Real callers have messier speech patterns, background noise, interruptions, and emotional nuances that an LLM simulation cannot fully capture.

2. **Demo-grade voice:**  
   The voice layer uses gTTS (offline text-to-speech) and Google's free speech-to-text API. These have high latency (2-5 seconds per turn), no streaming, and limited Hinglish accuracy. Production systems need low-latency streaming STT/TTS (e.g., Deepgram, Sarvam.ai, ElevenLabs).

3. **No real telephony integration:**  
   This project simulates phone calls in text and audio files. A production agent would integrate with a telephony platform (Twilio, Exotel, Plivo) for actual voice calls.

4. **Single-turn context:**  
   The agent does not track qualification state across turns in a structured way (e.g., a state machine). A production system might use a conversation state tracker or function calling to maintain field-level progress.

5. **No A/B testing:**  
   The comparison is deterministic — each version is tested offline. Real deployment would use A/B or multi-armed bandit testing with live traffic.

---

## Tech Stack

- **Python 3.11+**
- **LLM:** Claude (Anthropic API) — `claude-3-5-haiku-20241022`
- **Database:** SQLite3
- **API:** FastAPI + Uvicorn
- **Voice:** gTTS (TTS), SpeechRecognition + PyAudio (STT)
- **Data:** pandas
- **Testing:** pytest
- **Containers:** Docker

---

## License

This is a portfolio project by Ananya Dwivedi. All rights reserved.

---

## Contact

**Ananya Dwivedi**  
Email: _[your-email]_  
LinkedIn: _[your-linkedin]_  
GitHub: _[your-github]_

---

## Acknowledgments

Built for the Voice AI Prompt Engineer role at **SquadStack.ai** — India's leading voice AI platform for consumer sales with code-mixed Indic language support.
