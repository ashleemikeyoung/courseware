import os
import sys
import json
import time
import re
import threading
import requests
from pathlib import Path
try:
    import readline
except ImportError:
    readline = None

from rag import (
    search,
    scan_documents,
    get_indexed_sources,
    collection,
    DOCUMENTS_FOLDER,
    SCAN_INTERVAL,
    file_hash,
    SUPPORTED_EXTENSIONS,
    ignored_dirs,
)
import citations
import summarize

# Reuses ask.py's pronoun-reference resolution rather than re-implementing
# it here. This used to be a separate, hardcoded-to-"Tye" regex check (see
# _document_list_answer below) that quietly diverged from the real fix
# once ask.py's version was generalized -- MCP sessions (which route
# through ask.py, not this terminal loop) got the fix, terminal sessions
# didn't. One implementation now, so the two paths can't drift apart again.
from ask import _reference_terms_from_context, ask as ask_with_sources

# OLLAMA_URL and every *_MODEL name come from config.py -- see that
# module's docstring. This also fixes a latent bug: the load_dotenv() this
# replaced was called bare, no path, which searches from the current
# working directory rather than from this file's location -- it only ever
# worked because orchestrator.py happened to always be launched from
# ~/Development/RAG. config.py's load is anchored to its own file path, so
# that's no longer a requirement.
from config import (
    OLLAMA_URL, CODER_MODEL, GENERAL_MODEL, REASONING_MODEL,
    SYNTHESIS_MODEL, ROUTING_MODEL,
)

# Session/episodic logging lives in a sibling folder, not a package, so it
# needs to be added to sys.path before it can be imported. See
# ~/Development/RAG/memory/README.md for what this stores and why it's kept
# separate from chroma_db.
sys.path.insert(0, str(Path(__file__).resolve().parent / "memory"))
from memory_client import (
    start_session,
    find_cached_answer,
    flag_false_positive,
    pii_redaction_enabled,
    set_setting,
)

# pii.py is a sibling module in RAG root, not under memory/ -- no extra
# sys.path entry needed. Import is intentionally NOT wrapped in try/except:
# if Presidio isn't installed, that should surface clearly at startup
# ("ModuleNotFoundError: presidio_analyzer") rather than silently disabling
# redaction and letting someone believe PII protection is active when it
# isn't. See pii.py's docstring for install instructions.
import pii
import projects

BENCHMARK_LOG = os.getenv("BENCHMARK_LOG", "benchmark.jsonl")
ALL_PROJECTS = "all"
CURRENT_PROJECT_SCOPE = ALL_PROJECTS
HISTORY_PATH = Path(os.getenv(
    "ORCHESTRATOR_HISTORY",
    Path.home() / ".rag_orchestrator_history",
))
DOCUMENT_LIST_QUERY_RE = re.compile(
    r"\b(?:provide|give|show|list|find|which|what)\b.*"
    r"\b(?:all|any|the)?\s*(?:documents?|files?|sources?)\b.*"
    r"\b(?:reference|references|referencing|referenced|mention|mentions|"
    r"mentioned|cites?|cited|citing)\b\s+(?:to\s+)?(.+?)[?.!]*$",
    re.IGNORECASE,
)
DOCUMENT_SUMMARY_QUERY_RE = re.compile(
    r"\b(?:summarize|summarise|summary|summaries|recap|overview|synopsis|"
    r"digest|review)\w*\b.*\b(?:documents?|files?|sources?)\b.*"
    r"\b(?:reference|references|referencing|referenced|mention|mentions|"
    r"mentioned|cites?|cited|citing)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Session memory (separate from chroma_db and from BENCHMARK_LOG -- this is
# "what was asked and answered", not "what's relevant" or "how fast was it".
# See ~/Development/RAG/memory/README.md.)
# ---------------------------------------------------------------------------

_session = None
_incognito = False
_last_turn_id = None   # tracks the most recently logged turn, for /flag

# Process-local rolling context for pronoun resolution ("this article",
# "that source") -- NOT memory-db logging, NOT persisted anywhere, just
# enough recent Q&A text for _reference_terms_from_context() to resolve a
# reference against, the same way ask.py folds in recent conversation
# turns. This terminal loop previously had no notion of "recent context"
# at all, which is a big part of why the old hardcoded "Tye" check existed
# in the first place -- there was nothing else to check a reference against.
_recent_turns: list = []
_RECENT_TURNS_KEEP = 4


def _new_session(incognito: bool = False):
    """
    (Re)start the logging session. If memory-db isn't reachable (container
    not running, etc.), this session just goes unlogged -- it never blocks
    the actual Q&A loop, same failure philosophy as log_benchmark() above.
    """
    global _session
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
    try:
        _session = start_session(project=_memory_project(), machine="mac",
                                 mode="qa", incognito=incognito)
    except Exception as e:
        print(f"  [Warning: memory-db unreachable, this session won't be logged: {e}]",
              flush=True)
        _session = None


def _active_project() -> str:
    """Project value to pass into retrieval; None means search all projects."""
    return None if CURRENT_PROJECT_SCOPE == ALL_PROJECTS else CURRENT_PROJECT_SCOPE


def _memory_project() -> str:
    """Memory-db project label for cache/logging."""
    return None if CURRENT_PROJECT_SCOPE == ALL_PROJECTS else CURRENT_PROJECT_SCOPE


def _scope_label() -> str:
    return "all projects" if CURRENT_PROJECT_SCOPE == ALL_PROJECTS else CURRENT_PROJECT_SCOPE


def setup_input_history():
    if readline is None:
        return
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        if HISTORY_PATH.exists():
            readline.read_history_file(str(HISTORY_PATH))
        readline.set_history_length(500)
    except Exception as e:
        print(f"  [Warning: command history unavailable: {e}]", flush=True)


def save_input_history():
    if readline is None:
        return
    try:
        readline.write_history_file(str(HISTORY_PATH))
    except Exception:
        pass


def _log_turn(question: str, answer: str, model: str, sources: list):
    """
    NOTE: sources here are the document filenames get_rag_context() already
    collects, not chroma's full "{rel}::{i}" chunk ids -- search() doesn't
    return chunk ids today, only sources and text. Good enough to answer
    "which files fed this answer"; not precise enough to jump to the exact
    chunk. Threading real chunk ids through would mean patching rag.search()
    to return ids alongside documents/metadatas -- worth doing later, not
    bundled into this change.
    """
    global _last_turn_id
    if _session is None:
        return
    try:
        _last_turn_id = _session.log_turn(question=question, answer=answer,
                                          model=model, chunk_ids=sources)
    except Exception as e:
        print(f"  [Warning: could not log turn to memory-db: {e}]", flush=True)


def _close_session():
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass


def _record_recent_turn(question: str, answer: str):
    """
    Append to the rolling context buffer, skipped entirely in incognito
    mode -- incognito means nothing about the conversation is retained
    beyond the immediate turn, in-process buffer included, not just
    memory-db logging.
    """
    if _incognito:
        return
    _recent_turns.append({"question": question or "", "answer": answer or ""})
    del _recent_turns[:-_RECENT_TURNS_KEEP]


def _recent_context_text() -> str:
    """
    Joined question+answer text from the last few turns, mirroring ask.py's
    `recent_context = " ".join(m.get("content", "") for m in messages[-8:])`
    -- same idea, just built from this loop's own (question, answer) pairs
    instead of a shared messages list.
    """
    parts = []
    for turn in _recent_turns:
        parts.append(turn["question"])
        parts.append(turn["answer"])
    return " ".join(parts)


def _recent_chat_messages() -> list:
    messages = []
    for turn in _recent_turns:
        if turn.get("question"):
            messages.append({"role": "user", "content": turn["question"]})
        if turn.get("answer"):
            messages.append({"role": "assistant", "content": turn["answer"]})
    return messages


def cmd_incognito():
    global _incognito
    _incognito = not _incognito
    _new_session(incognito=_incognito)
    _recent_turns.clear()  # a clean slate either direction, same reasoning
                           # as _record_recent_turn()'s own incognito check
    state = ("ON -- nothing from here forward is being logged" if _incognito
             else "OFF -- logging resumed")
    print(f"\nIncognito: {state}\n")


def cmd_flag():
    """Mark the most recently logged turn as a false positive, with notes."""
    if _last_turn_id is None:
        print("\nNothing to flag yet -- ask a question first.\n")
        return
    notes = input("What was wrong with that answer? ").strip()
    try:
        flag_false_positive(_last_turn_id, notes or None)
        print(f"\nFlagged turn #{_last_turn_id} as a false positive.\n")
    except Exception as e:
        print(f"  [Warning: could not flag turn in memory-db: {e}]\n", flush=True)


def cmd_redact():
    """
    Admin toggle for PII redaction, persisted in memory-db's settings table
    so it applies immediately to writer.py too, not just this process.
    See pii.py's docstring: this is a strong aid, not a guarantee -- treat
    "redaction on" as "known PII gets masked," not as "this is now safe to
    share no matter what."
    """
    try:
        currently_on = pii_redaction_enabled()
        set_setting("pii_redaction", "off" if currently_on else "on")
        state = "OFF" if currently_on else "ON"
        print(f"\nPII redaction: {state} "
              f"(applies to answers here and documents saved by writer.py)\n")
    except Exception as e:
        print(f"  [Warning: could not update PII redaction setting: {e}]\n", flush=True)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

class PhaseTimer:
    """Tracks timing and stats for each phase of a question."""

    def __init__(self, question: str):
        self.question = question
        self.session_start = time.time()
        self.phases = []
        self._phase_start = None
        self._current_phase = None

    def start_phase(self, name: str):
        self._current_phase = name
        self._phase_start = time.time()

    def end_phase(self, stats: dict = None):
        if self._phase_start is None:
            return
        elapsed = time.time() - self._phase_start
        entry = {
            "phase": self._current_phase,
            "elapsed_s": round(elapsed, 3),
        }
        if stats:
            entry.update(stats)
        self.phases.append(entry)
        self._phase_start = None
        self._current_phase = None
        return elapsed

    def total_elapsed(self):
        return time.time() - self.session_start

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"BENCHMARK SUMMARY",
            f"Question: {self.question[:80]}{'...' if len(self.question) > 80 else ''}",
            f"{'='*60}",
        ]
        total = self.total_elapsed()
        for p in self.phases:
            elapsed = p['elapsed_s']
            pct = (elapsed / total * 100) if total > 0 else 0
            line = f"  {p['phase']:<25} {elapsed:>6.2f}s  ({pct:>4.1f}%)"

            extras = []
            if "tokens_per_second" in p:
                extras.append(f"tok/s: {p['tokens_per_second']:.1f}")
            if "total_tokens" in p:
                extras.append(f"tokens: {p['total_tokens']}")
            if "prompt_tokens" in p:
                extras.append(f"prompt: {p['prompt_tokens']}")
            if "response_tokens" in p:
                extras.append(f"response: {p['response_tokens']}")
            if "chunks_retrieved" in p:
                extras.append(f"chunks: {p['chunks_retrieved']}")
            if "chunk_chars" in p:
                extras.append(f"chars: {p['chunk_chars']}")
            if "model" in p:
                extras.append(f"model: {p['model']}")
            if "task_type" in p:
                extras.append(f"type: {p['task_type']}")

            if extras:
                line += f"  [{', '.join(extras)}]"
            lines.append(line)

        lines.append(f"  {'TOTAL':<25} {total:>6.2f}s")
        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "question": self.question,
            "total_elapsed_s": round(self.total_elapsed(), 3),
            "routing_model": ROUTING_MODEL,
            "synthesis_model": SYNTHESIS_MODEL,
            "phases": self.phases,
        }


def log_benchmark(timer: PhaseTimer):
    """Append benchmark result to the JSONL log file."""
    try:
        with open(BENCHMARK_LOG, "a") as f:
            f.write(json.dumps(timer.to_dict()) + "\n")
    except Exception as e:
        print(f"  [Warning: could not write benchmark log: {e}]", flush=True)


# ---------------------------------------------------------------------------
# Ollama with metrics
# ---------------------------------------------------------------------------

def ask_ollama(prompt: str, model: str, system: str = None, retries: int = 2) -> tuple:
    """
    Send a task to Ollama and return (response_text, metrics_dict).
    Metrics include tokens_per_second, total_tokens, prompt_tokens, response_tokens.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries + 1):
        try:
            start = time.time()
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
                timeout=300,
            )
            response.raise_for_status()
            elapsed = time.time() - start
            data = response.json()
            text = data["message"]["content"]

            # Extract Ollama's native metrics
            metrics = {
                "model": model,
                "elapsed_s": round(elapsed, 3),
            }

            if "eval_count" in data and "eval_duration" in data:
                eval_tokens = data["eval_count"]
                eval_duration_s = data["eval_duration"] / 1e9
                metrics["response_tokens"] = eval_tokens
                metrics["tokens_per_second"] = round(
                    eval_tokens / eval_duration_s if eval_duration_s > 0 else 0, 1
                )

            if "prompt_eval_count" in data:
                metrics["prompt_tokens"] = data["prompt_eval_count"]

            if "prompt_eval_count" in data and "eval_count" in data:
                metrics["total_tokens"] = data["prompt_eval_count"] + data["eval_count"]

            return text, metrics

        except requests.exceptions.Timeout:
            if attempt < retries:
                print(f"  [Timeout, retrying {attempt + 1}/{retries}...]", flush=True)
            else:
                raise

    return "", {}


# ---------------------------------------------------------------------------
# RAG context
# ---------------------------------------------------------------------------

def _document_list_term(question: str) -> str:
    match = DOCUMENT_LIST_QUERY_RE.search(question or "")
    if not match:
        return ""
    return match.group(1).strip(" \t\r\n\"'`“”‘’.?!")


def _document_list_answer(question: str, project: str = None,
                          context: str = "") -> dict:
    """
    "documents that reference X" -> exhaustive indexed-document scan.

    The literal `term` the regex above extracts is often a pronoun ("this
    article", "that source") rather than a real search term -- see ask.py's
    _reference_terms_from_context() docstring for the full history (this
    used to be a hardcoded "Tye" check here, then a generalized fix that
    only ask.py's MCP path benefited from). Resolving it the same way here
    keeps terminal sessions and MCP sessions behaving identically instead
    of the terminal quietly reverting to the old broken behavior.
    """
    term = _document_list_term(question)
    if not term:
        return None

    terms = _reference_terms_from_context(term, context, project=project)
    scope = f" in project '{project}'" if project else ""

    if not terms:
        answer = f"No indexed documents{scope} reference or mention '{term}'."
        return {
            "answer": answer,
            "sources": [],
            "routed_to": "document_list",
            "routing_reason": "exhaustive indexed-document scan",
            "ollama_draft": None,
            "synthesized_by": "none",
        }

    matches = {}
    for search_term in terms:
        for source in summarize.find_documents(search_term, project=project):
            matches.setdefault(source, set()).add(search_term)

    if not matches:
        answer = f"No indexed documents{scope} reference or mention '{term}'."
    else:
        used_terms = ", ".join(terms)
        resolved_note = (f" (resolved to: {used_terms})"
                         if used_terms.lower() != term.lower() else "")
        lines = [
            f"Indexed documents{scope} that reference or mention "
            f"'{term}'{resolved_note}:",
            "",
        ]
        lines.extend(f"- {source}" for source in sorted(matches))
        answer = "\n".join(lines)

    return {
        "answer": answer,
        "sources": sorted(matches),
        "routed_to": "document_list",
        "routing_reason": "exhaustive indexed-document scan",
        "ollama_draft": None,
        "synthesized_by": "none",
    }


def _document_summary_answer(question: str, project: str = None) -> dict:
    if not DOCUMENT_SUMMARY_QUERY_RE.search(question or ""):
        return None

    term = summarize.reference_query_term(question)
    results = summarize.summarize_search(term, project=project)
    sources = [r["source"] for r in results]
    scope = f" in project '{project}'" if project else ""

    if not results:
        answer = f"No indexed documents{scope} reference or mention '{term}'."
    else:
        sections = [
            f"Summaries of indexed documents{scope} that reference or mention '{term}':",
        ]
        for r in results:
            sections.append(f"\n## {r['source']}")
            if "error" in r:
                sections.append(f"Could not summarize: {r['error']}")
                continue
            sections.append(r["summary"])
            if r.get("truncated"):
                sections.append(f"\n(truncated at {r['chars']} characters)")
        answer = "\n".join(sections)

    return {
        "answer": answer,
        "sources": sources,
        "routed_to": "document_summary",
        "routing_reason": "exhaustive indexed-document scan plus whole-file summaries",
        "ollama_draft": None,
        "synthesized_by": "none",
    }


def get_rag_context(question: str, project: str = None) -> tuple:
    """Pull relevant chunks from the vector database, optionally scoped to one project."""
    if collection.count() == 0:
        return "", [], {}

    results = search(question, project=project)
    context = ""
    sources = []

    for i, doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"]
        context += f"[From {source}]\n{doc}\n\n"
        sources.append(source)

    # Citation correlation -- see citations.py's module docstring for the
    # full "Tye" reasoning. This used to be an inline duplicate of the same
    # logic in mcp_server.py; both now call the one shared implementation.
    for hit in citations.topup(question, set(sources), project=project):
        if "_warning" in hit:
            print(f"  [Warning: citation lookup failed: {hit['_warning']}]", flush=True)
            continue
        context += (
            f"[Verified citation record for {hit['source']}]\n"
            f"Title: {hit['title']}\n"
            f"Author(s): {hit['authors']}\n"
            f"Source: {hit['source_line']}\n\n"
        )
        sources.append(hit["source"])

    stats = {
        "chunks_retrieved": len(results["documents"][0]),
        "chunk_chars": len(context),
        "sources": list(set(sources)),
    }

    return context, list(set(sources)), stats


# ---------------------------------------------------------------------------
# Background file watcher
# ---------------------------------------------------------------------------

_watcher_running = False


def start_file_watcher():
    global _watcher_running
    _watcher_running = True

    def watch():
        print(f"[File watcher started, checking every {SCAN_INTERVAL}s]", flush=True)
        while _watcher_running:
            time.sleep(SCAN_INTERVAL)
            folder = Path(DOCUMENTS_FOLDER)
            if not folder.exists():
                continue

            # Recursive and keyed the same way rag.scan_documents() keys its
            # own indexed_sources (relative path from the documents root,
            # e.g. "GCU/foo.pdf"). The original version here used
            # folder.iterdir() (top-level only) keyed by bare filename --
            # every real file lives under a project subfolder, so that
            # never matched get_indexed_sources()'s keys at all. The
            # comparison read as "everything changed" on literally every
            # tick, forcing a full rescan (hashing all files) every
            # SCAN_INTERVAL seconds and reprinting the prompt mid-keystroke.
            current = {
                str(f.relative_to(folder)): file_hash(f)
                for f in folder.rglob("*")
                if f.is_file()
                and f.suffix.lower() in SUPPORTED_EXTENSIONS
                and not any(part in ignored_dirs() or part.startswith(".")
                            for part in f.relative_to(folder).parts[:-1])
            }
            indexed = get_indexed_sources()

            needs_update = (
                set(current.keys()) != set(indexed.keys()) or
                any(current.get(f) != indexed.get(f) for f in current)
            )

            if needs_update:
                print("\n[File watcher detected changes, rescanning...]", flush=True)
                summary = scan_documents(verbose=True)
                print(
                    f"[Rescan complete: "
                    f"{len(summary['new'])} new, "
                    f"{len(summary['updated'])} updated, "
                    f"{len(summary['removed'])} removed]\n"
                    "You: ",
                    end="",
                    flush=True,
                )

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ROUTING_SYSTEM = """You are an orchestrator deciding how to handle a user question.
Reply with JSON only. No markdown. No explanation. No text outside the JSON object.
Use this exact structure:
{
  "task_type": "coding" | "reasoning" | "general" | "direct",
  "subtask_prompt": "the focused prompt to send to the specialist model, or null if direct",
  "reasoning": "one sentence explaining your choice"
}

Routing rules:
- "coding": writing, explaining, debugging, or reviewing code of any kind
- "reasoning": complex multi-step logic, math, analysis, comparisons, planning, or anything that benefits from deep chain-of-thought thinking
- "general": summaries, drafts, structured writing, straightforward Q&A that needs more than a direct answer
- "direct": simple factual questions that can be answered cleanly from the context without specialist help

Return only the JSON object. Nothing else."""

SYNTHESIS_SYSTEM = """You are a helpful assistant that synthesizes work from specialist models
and answers questions from document context. Be accurate, cite document sources by filename,
and present code or structured output clearly. If a specialist model produced a draft,
review it critically, correct anything needed, and deliver the best final version."""


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_question(question: str, context: str, timer: PhaseTimer) -> dict:
    timer.start_phase("routing")

    for attempt in range(3):
        raw, metrics = ask_ollama(
            prompt=f"Document context:\n{context}\n\nUser question: {question}",
            model=ROUTING_MODEL,
            system=ROUTING_SYSTEM,
        )

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            routing = json.loads(cleaned)
            if "task_type" in routing:
                metrics["task_type"] = routing["task_type"]
                timer.end_phase(metrics)
                return routing
        except json.JSONDecodeError:
            pass

        if attempt < 2:
            print(f"  [Routing parse failed, retrying {attempt + 1}/3...]", flush=True)

    print("  [Routing failed after 3 attempts, defaulting to direct]", flush=True)
    timer.end_phase(metrics)
    return {
        "task_type": "direct",
        "subtask_prompt": None,
        "reasoning": "routing parse failed",
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def orchestrate(user_question: str, project: str = None) -> dict:
    timer = PhaseTimer(user_question)

    timer.start_phase("document_summary")
    direct_summary = _document_summary_answer(user_question, project=project)
    if direct_summary is not None:
        timer.end_phase({
            "sources": direct_summary["sources"],
            "chunks_retrieved": 0,
            "chunk_chars": len(direct_summary["answer"]),
        })
        direct_summary["timer"] = timer
        log_benchmark(timer)
        return direct_summary

    timer.start_phase("document_scan")
    direct_list = _document_list_answer(user_question, project=project,
                                        context=_recent_context_text())
    if direct_list is not None:
        timer.end_phase({
            "sources": direct_list["sources"],
            "chunks_retrieved": 0,
            "chunk_chars": len(direct_list["answer"]),
        })
        direct_list["timer"] = timer
        log_benchmark(timer)
        return direct_list

    print("[Answering with shared Ask engine: local retrieval + external pull]", flush=True)
    timer.start_phase("ask")
    messages = _recent_chat_messages() + [{"role": "user", "content": user_question}]
    result = ask_with_sources(
        messages,
        project=project,
        ground=True,
        external_policy="pull",
    )
    metrics = result.get("metrics") or {}
    timer.end_phase({
        "model": metrics.get("model") or SYNTHESIS_MODEL,
        "chunks_retrieved": result.get("passages_offered", 0),
        "chunk_chars": sum(len(e.get("text") or "")
                           for e in (result.get("evidence") or {}).values()),
    })
    sources = sorted({e.get("source") for e in (result.get("evidence") or {}).values()
                      if e.get("source")})
    log_benchmark(timer)

    return {
        "answer": result["text"],
        "sources": sources,
        "routed_to": "shared_ask",
        "routing_reason": "app/orchestrator pull-only external policy",
        "ollama_draft": None,
        "synthesized_by": metrics.get("model") or "ask.ask",
        "timer": timer,
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_rescan():
    print("\nRescanning documents folder...", flush=True)
    summary = scan_documents(verbose=True)
    print(
        f"Done: {len(summary['new'])} new, "
        f"{len(summary['updated'])} updated, "
        f"{len(summary['removed'])} removed, "
        f"{len(summary['unchanged'])} unchanged"
    )
    print(f"Total chunks: {collection.count()}\n", flush=True)


def cmd_status():
    indexed = get_indexed_sources()
    print(f"\nDatabase: {collection.count()} chunks across {len(indexed)} file(s)")
    print(f"Scope:    {_scope_label()}")
    if indexed:
        for filename in sorted(indexed.keys()):
            print(f"  {filename}")
    else:
        print("  (no files indexed)")
    print(f"\nRouting:   {ROUTING_MODEL}")
    print(f"Synthesis: {SYNTHESIS_MODEL}")
    print(f"Coder:     {CODER_MODEL}")
    print(f"Reasoning: {REASONING_MODEL}")
    print(f"General:   {GENERAL_MODEL}")
    print(f"Benchmark: {BENCHMARK_LOG}\n")


def cmd_projects():
    rows = projects.stats(collection)
    print("\nProjects:")
    print("  all  (search across all projects)")
    for row in rows:
        marker = "*" if row["name"] == CURRENT_PROJECT_SCOPE else " "
        print(
            f"{marker} {row['name']}  "
            f"{row['files']} file(s), {row['chunks']} chunk(s)"
        )
    if CURRENT_PROJECT_SCOPE == ALL_PROJECTS:
        print("* current scope: all projects")
    else:
        print(f"\nCurrent scope: {CURRENT_PROJECT_SCOPE}")
    print()


def cmd_project(arg: str = ""):
    global CURRENT_PROJECT_SCOPE
    name = (arg or "").strip()
    if not name:
        print(f"\nCurrent scope: {_scope_label()}")
        print("Use /project <name>, /all, or /unfiled.\n")
        return

    lowered = name.lower()
    if lowered in {"all", "*"}:
        CURRENT_PROJECT_SCOPE = ALL_PROJECTS
    else:
        safe = projects.safe(name)
        available = set(projects.discover())
        if safe not in available:
            print(f"\nUnknown project '{name}'. Use /projects to see available scopes.\n")
            return
        CURRENT_PROJECT_SCOPE = safe

    _new_session(incognito=_incognito)
    _recent_turns.clear()  # a reference to "this article" from the old
                           # scope shouldn't resolve against a different
                           # project's documents
    print(f"\nScope set to: {_scope_label()}\n")


def cmd_all():
    cmd_project("all")


def cmd_unfiled():
    cmd_project(projects.UNFILED)


def cmd_clear():
    confirm = input(
        "This will delete the entire database. Type 'yes' to confirm: "
    ).strip()
    if confirm.lower() == "yes":
        ids = collection.get()["ids"]
        collection.delete(ids=ids)
        _recent_turns.clear()  # old context may reference now-deleted docs
        print("Database cleared. Type /rescan to rebuild.\n")
    else:
        print("Cancelled.\n")


def cmd_benchmark():
    """Show a summary of all benchmark runs from the log file."""
    log_path = Path(BENCHMARK_LOG)
    if not log_path.exists():
        print("\nNo benchmark log found yet. Ask a question first.\n")
        return

    runs = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not runs:
        print("\nBenchmark log is empty.\n")
        return

    print(f"\n{'='*80}")
    print(f"BENCHMARK HISTORY ({len(runs)} runs)")
    print(f"{'='*80}")
    print(f"{'#':<4} {'Routing':<20} {'Synthesis':<20} {'Total':>7} {'tok/s':>7}  Question")
    print(f"{'-'*80}")

    for i, run in enumerate(runs):
        routing_model = run.get("routing_model", "?").split(":")[-1]
        synthesis_model = run.get("synthesis_model", "?").split(":")[-1]
        total = run.get("total_elapsed_s", 0)

        # Find synthesis tok/s
        tps = ""
        for phase in run.get("phases", []):
            if phase.get("phase") == "synthesis" and "tokens_per_second" in phase:
                tps = f"{phase['tokens_per_second']:.0f}"

        question = run.get("question", "")[:40]
        print(f"{i+1:<4} {routing_model:<20} {synthesis_model:<20} {total:>6.1f}s {tps:>7}  {question}")

    print(f"{'='*80}\n")


def cmd_clear_benchmark():
    confirm = input("Clear all benchmark history? Type 'yes' to confirm: ").strip()
    if confirm.lower() == "yes":
        Path(BENCHMARK_LOG).unlink(missing_ok=True)
        print("Benchmark log cleared.\n")
    else:
        print("Cancelled.\n")


def cmd_summarize(term: str):
    """
    Search finds which files are relevant, then each match gets read
    straight off disk and summarized whole -- not from retrieved chunks.
    Same shared pipeline as MCP's summarize_documents tool and app.py's
    /api/summarize route (see summarize.py's module docstring), so this
    command exists mainly so the terminal doesn't need ask_local's
    citation-matching heuristics just to answer "summarize the article by
    X" -- it goes straight to the real document.
    """
    project = _active_project()
    scope = f" in {_scope_label()}" if project else " across all projects"
    print(f"\nSearching for documents matching '{term}'{scope}...", flush=True)
    results = summarize.summarize_search(term, project=project)
    if not results:
        print(f"  No indexed documents matched '{term}'.\n")
        return

    print(f"  {len(results)} matching document(s).\n")
    for r in results:
        print(f"--- {r['source']} ---")
        if "error" in r:
            print(f"  [Error: {r['error']}]\n")
            continue
        print(r["summary"])
        if r.get("truncated"):
            print(f"  (truncated at {r['chars']} characters)")
        print()


COMMANDS = {
    "/rescan": cmd_rescan,
    "/status": cmd_status,
    "/projects": cmd_projects,
    "/all": cmd_all,
    "/unfiled": cmd_unfiled,
    "/clear": cmd_clear,
    "/benchmark": cmd_benchmark,
    "/clearbenchmark": cmd_clear_benchmark,
    "/incognito": cmd_incognito,
    "/flag": cmd_flag,
    "/redact": cmd_redact,
}

HELP_TEXT = """
Commands:
  /rescan          — scan documents folder and update the database
  /status          — show indexed files, chunk counts, and active models
  /projects        — show available project scopes
  /project <name>  — confine searches to one project
  /all             — search across all projects
  /unfiled         — search only loose files at the documents root
  /clear           — wipe the database completely
  /summarize <term> — search the index and summarize every matching
                      document straight off disk, not from retrieved chunks
  /benchmark       — show timing history across all runs
  /clearbenchmark  — clear benchmark history
  /incognito       — toggle session logging off/on for what follows
  /flag            — mark the last answer as wrong, for later review
  /redact          — toggle PII redaction on answers (and writer.py output)
  /help            — show this message
  quit             — exit
"""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_input_history()
    print("\nInitial document scan...", flush=True)
    summary = scan_documents(verbose=True)
    print(
        f"Ready: {len(summary['new'])} new, "
        f"{len(summary['updated'])} updated, "
        f"{len(summary['removed'])} removed, "
        f"{len(summary['unchanged'])} unchanged"
    )
    print(f"Total chunks in database: {collection.count()}")
    print(f"\nRouting via:   {ROUTING_MODEL}")
    print(f"Synthesis via: {SYNTHESIS_MODEL}")
    print(f"Benchmark log: {BENCHMARK_LOG}")
    print(f"Initial scope: {_scope_label()}")

    start_file_watcher()
    _new_session(incognito=False)

    print("\nOrchestrated RAG — fully local")
    print(HELP_TEXT)

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            save_input_history()
            _close_session()
            break

        if not q:
            continue

        if q.lower() in ["quit", "exit"]:
            print("Goodbye.")
            save_input_history()
            _close_session()
            break

        if q.lower() == "/help":
            print(HELP_TEXT)
            continue

        if q.lower().startswith("/summarize"):
            term = q[len("/summarize"):].strip()
            if not term:
                print("\nUsage: /summarize <search term>\n")
            else:
                cmd_summarize(term)
            continue

        if q.lower() in COMMANDS:
            COMMANDS[q.lower()]()
            continue

        if q.lower() == "/project" or q.lower().startswith("/project "):
            cmd_project(q[len("/project"):].strip())
            continue

        # Repeat-question check: show the cached answer and let the person
        # decide whether to re-run fresh, rather than silently reusing it
        # (stale context, an updated index, or a since-flagged answer could
        # all make a fresh run the right call) or silently re-running every
        # time (which is the actual cost this exists to avoid).
        cached = None
        try:
            cached = find_cached_answer(q, project=_memory_project())
        except Exception as e:
            print(f"  [Warning: could not check memory-db for a cached answer: {e}]",
                  flush=True)

        if cached:
            print(f"\n[You asked this before, on {cached['asked_at']}]")
            print(f"\n--- Cached Answer ---\n{cached['answer']}")
            rerun = input("\nRe-run fresh instead? [y/N] ").strip().lower()
            if not rerun.startswith("y"):
                _record_recent_turn(q, cached["answer"])
                continue

        result = orchestrate(q, project=_active_project())
        answer_to_show = result["answer"]
        try:
            if pii_redaction_enabled():
                answer_to_show = pii.redact_text(answer_to_show)
                print("\n[PII redaction is ON -- entities masked below]")
        except Exception as e:
            print(f"  [Warning: PII redaction failed, showing unredacted answer: {e}]",
                  flush=True)

        # Logged and cached as originally generated, not the redacted view --
        # redaction is a display-time transform (see pii.py), so /flag notes
        # and future cache hits still reflect the real underlying answer.
        _log_turn(q, result["answer"], result["synthesized_by"], result["sources"])
        _record_recent_turn(q, result["answer"])

        print(f"\n[Routed to: {result['routed_to']} | Synthesized by: {result['synthesized_by']}]")

        if result["ollama_draft"]:
            print(f"\n--- Specialist Draft ---\n{result['ollama_draft']}")

        print(f"\n--- Final Answer ---\n{answer_to_show}")

        if result["sources"]:
            print(f"\n[Sources: {', '.join(result['sources'])}]")

        print(result["timer"].summary())
