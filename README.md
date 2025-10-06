# Neura Dynamics – Agentic Weather + PDF RAG

## Overview
Neura Dynamics is a Streamlit-based agent that routes questions to either:
- Weather: current conditions via OpenWeatherMap
- PDF RAG: grounded answers from your PDFs with sources

Key capabilities:
- Maintains the last 2 turns of chat history to resolve follow-ups (e.g., “there/here”).
- Weather follow-ups: extracts city from user or assistant’s last message; falls back to memory or asks a concise clarification.
- RAG follow-ups: rewrites the question using recent history for better retrieval and citations.

## Architecture
- Orchestration: `langgraph` state machine with nodes `router` → `weather` or `rag`.
- LLM: Google Gemini via LangChain; embeddings via SentenceTransformers (CPU) or Google.
- Vector DB: Qdrant (Docker) with automatic local Chroma fallback.
- Memory: ephemeral per-session store for last location/units.

Flow per turn:
1) UI calls the graph with `{question, chat_history=last 2 messages}`.
2) Router decides route using heuristics (optional LLM routing).
3) Weather node or RAG node answers and returns structured data back to UI.

Code entry points:
- UI: `app.py`
- Agent graph: `src/graph.py`
- Weather client: `src/weather.py`
- RAG pipeline: `src/rag.py`
- Memory: `src/memory.py`
- Config/flags: `src/config.py`

## Requirements
- Python 3.10+
- Qdrant

## Install
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Environment (.env)
Create a `.env` with at least your keys:
```
GOOGLE_API_KEY=...
OPENWEATHER_API_KEY=...
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=neura_docs
DEFAULT_GEMINI_MODEL=gemini-2.0-flash-lite
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GOOGLE_EMBEDDING_MODEL=text-embedding-004
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=NeuraDynamicsAssignment
LANGCHAIN_TRACING_V2=true
WEATHER_CACHE_TTL_SECONDS=300
HTTP_TIMEOUT_SECONDS=10
HTTP_MAX_RETRIES=3
```

Optional feature flags (set to `true` to enable):
```
WEATHER_SLOT_LLM=false          # LLM extracts weather slots
WEATHER_CLARIFICATION_LLM=false # LLM crafts clarifying question
ENABLE_LLM_ROUTING=false        # LLM-based router
RAG_REWRITE_WITH_HISTORY=true   # rewrite follow-ups using history (default behavior)
RAG_HEAD_CHARS=600
RAG_MAX_CONTEXT_CHARS=4000
```

## Run
```bash
streamlit run app.py
```

In the sidebar:
- Paste `GOOGLE_API_KEY`, `OPENWEATHER_API_KEY`.
- Choose embeddings (Sentence or Google).
- Ingest PDFs (upload or “Ingest PDFs from data/”).

Chat usage:
- Weather: “What’s the weather in Paris?” → follow up “What’s the temp there?”
- RAG: “Summarize section 2.” → follow up “And limitations?”

The UI persists full assistant messages (tool badge, weather details, sources) so previous answers remain visible.


## Tests
```bash
pytest -q
```
Highlights:
- `tests/test_history.py`: multi-turn RAG rewrite and weather follow-ups (e.g., “Paris”, “there”).
- `tests/test_weather.py`: parsing, caching, clarification, and error behavior.
- `tests/test_rag.py`: retrieval and context building.

## Troubleshooting
- “there/here” follow-ups don’t resolve:
  - Ensure you asked a city first; the agent reads the last 2 messages and assistant replies to infer the city.
  - Consider enabling `WEATHER_CLARIFICATION_LLM=true` for smarter clarifications.
- Qdrant offline:
  - The app will warn and switch to local Chroma; ingestion still works locally.
- Missing Google key:
  - Provide `GOOGLE_API_KEY` in the sidebar or `.env` for Gemini and Google embeddings (SentenceTransformers works without it).

## Telemetry (LangSmith)
Set `LANGSMITH_API_KEY` and keep `LANGCHAIN_TRACING_V2=true` to emit traces. The app also runs a lightweight auto-eval on answers when available.


<img width="1909" height="955" alt="Screenshot 2025-10-06 231354" src="https://github.com/user-attachments/assets/7d04eafa-2ab0-40df-938e-9d5d23e7bdee" />
<img width="1904" height="1017" alt="Screenshot 2025-10-06 224807" src="https://github.com/user-attachments/assets/bee4b4cf-36fc-428c-b8f6-bcf14e427e23" />
