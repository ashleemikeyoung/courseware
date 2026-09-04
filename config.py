"""
config.py — single source of truth for environment configuration: the .env
file, the Ollama endpoint, and every named model role.

Before this, .env was loaded independently in three places -- rag.py,
writer.py, and orchestrator.py -- and orchestrator.py's was the fragile
version: a bare load_dotenv() with no path searches from the current
working directory, not from wherever this file lives, so it only worked
because orchestrator.py always happened to be launched from
~/Development/RAG. Every model role was also declared wherever the module
that used it happened to live, so "what model handles X, and how do I
change it" had a different answer depending which file you opened, and
ask.py had no answer at all -- it took whatever model its caller happened
to pass in, defaulting to writer.py's DRAFT_MODEL by convention rather than
by anything ask.py itself declared.

Every other module imports OLLAMA_URL and model names from here instead of
reading os.getenv() itself. One .env load, anchored to this file's
location; one place to look for or add a model role.

NOTE: never point any *_MODEL variable at an Ollama "-cloud" model. Those
ship your document chunks off the machine, which defeats the entire point
of a local-only pipeline -- this is the same reason app.py filters
cloud-tagged models out of the UI model picker.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- rag.py: image OCR fallback during indexing -----------------------
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:13b")

# --- writer.py: long-form document generation --------------------------
# Long-form wants a bigger model than fast Q&A does. You generate once and
# then read the result for twenty minutes, so quality beats latency here.
# Using one 32b for both outlining and drafting avoids a model swap mid-run.
OUTLINE_MODEL = os.getenv("OLLAMA_OUTLINE_MODEL", "qwen3:32b")
DRAFT_MODEL = os.getenv("OLLAMA_DRAFT_MODEL", "qwen3:32b")
NOTES_MODEL = os.getenv("OLLAMA_NOTES_MODEL", "gemma4:e4b")

# --- ask.py: single-turn grounded chat ----------------------------------
# Used by both app.py's Ask tab and MCP's ask_local tool, so this is the
# one place that answer quality/speed tradeoff gets tuned for both.
# Defaults to the same model as drafting, but is independently
# configurable -- a smaller, faster model might suit quick chat even when
# long-form drafting still wants something bigger.
ASK_MODEL = os.getenv("OLLAMA_ASK_MODEL", DRAFT_MODEL)

# --- ask.py: Google Search grounding for web-search evidence ------------
# Optional. When empty, ask.py's web search falls back to the existing
# DuckDuckGo-scraping path -- a missing key degrades the feature, it never
# breaks the app. Get a free key (no credit card required) from Google AI
# Studio at aistudio.google.com, then set GOOGLE_API_KEY in .env.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
GOOGLE_SEARCH_MODEL = os.getenv("GOOGLE_SEARCH_MODEL", "gemini-2.5-flash")

# --- orchestrator.py: terminal specialist routing -----------------------
CODER_MODEL = os.getenv("OLLAMA_CODER_MODEL", "qwen2.5-coder:32b")
GENERAL_MODEL = os.getenv("OLLAMA_GENERAL_MODEL", "qwen3:32b")
REASONING_MODEL = os.getenv("OLLAMA_REASONING_MODEL", "qwq:32b")
SYNTHESIS_MODEL = os.getenv("OLLAMA_SYNTHESIS_MODEL", "gemma4:e4b")
ROUTING_MODEL = os.getenv("OLLAMA_ROUTING_MODEL", "gemma4:e4b")
