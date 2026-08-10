"""
app.py — local web front end for Ask Ash.

Runs on 127.0.0.1 only. Nothing here reaches the network: no CDN, no web fonts,
no analytics. Same privacy boundary as the rest of the pipeline.

  pip install flask
  python app.py
  open http://127.0.0.1:5111

Long runs are handled as background jobs rather than blocking requests. A draft
takes twenty minutes, which no HTTP request should ever wait for. This is the
same job pattern the MCP server will need when write_document gets exposed as a
tool, since Claude Desktop times tool calls out long before a paper finishes.
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

import writer
import quality
import bench
import ask
import summarize
import projects
import updater
from writer import BASE_DIR, collection

app = Flask(__name__, static_folder=str(BASE_DIR / "static"))
# Jinja ties template auto-reload to app.debug by default, and debug stays
# off below (deliberately -- Flask's debug mode opens an interactive Python
# shell over HTTP on unhandled exceptions, a real risk even bound to
# 127.0.0.1 if anything else on the machine can reach that port). This line
# gets the one piece actually wanted -- index.html re-read from disk on
# every request -- without turning on the debugger. Without it, Jinja
# compiles the template once on first render and keeps serving that cached
# version from memory regardless of edits on disk, until the process
# restarts -- which is why a CSS change (e.g. the logo's height) can appear
# to do nothing no matter how many times the page is reloaded in the
# browser: the server itself is the one holding stale content, not the
# browser's cache. Harmless to leave on permanently -- one file mtime
# check per request, not a real cost.
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Recolour the wordmark for each theme at startup. The source navy scores 1.51
# contrast against the dark background, so a single file would leave the Q
# invisible there. Cheap and idempotent: skips any variant newer than the source.
#
# Wrapped because none of this is load bearing. A missing logo.py, absent
# Pillow, or unreadable source should cost you a wordmark, never the server.
try:
    import logo
    _LOGO_SRC = BASE_DIR / "static" / "logo.png"
    if _LOGO_SRC.exists():
        logo.build(_LOGO_SRC, BASE_DIR / "static")
except Exception as _e:
    print(f"  Logo skipped ({_e}). Everything else runs normally.")

# First run on a machine with no .git yet takes whatever's currently on disk
# as the baseline version -- see updater.py's docstring. Wrapped for the same
# reason the logo block above is: a missing/broken git installation should
# cost you the update feature, never the server itself.
try:
    updater.ensure_repo()
except Exception as _e:
    print(f"  Update tracking unavailable ({_e}). Everything else runs normally.")


def _delayed_restart(delay: float = 0.6):
    """
    Spawn a detached restart helper, then exit this process.

    When app.py is launched by hand from a CLI, os.execv() keeps the server
    tied to that foreground process. The update button then feels like it
    kills the app instead of resuming it.

    Starting the replacement directly from this process is racy, though:
    the old Flask server still owns port 5111 for a moment, so the new
    process can fail to bind and exit. The helper waits after this process
    exits, then launches app.py detached -- the Python equivalent of:

        sleep 1 && python app.py &
    """
    def go():
        time.sleep(delay)
        payload = json.dumps({
            "argv": [sys.executable] + sys.argv,
            "cwd": str(BASE_DIR),
            "wait": 1.25,
            "log": str(BASE_DIR / "app-restart.log"),
        })
        helper = (
            "import json, subprocess, time\n"
            f"cfg = json.loads({payload!r})\n"
            "time.sleep(cfg['wait'])\n"
            "log = open(cfg['log'], 'ab', buffering=0)\n"
            "subprocess.Popen(\n"
            "    cfg['argv'], cwd=cfg['cwd'], stdin=subprocess.DEVNULL,\n"
            "    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,\n"
            ")\n"
        )
        subprocess.Popen(
            [sys.executable, "-c", helper],
            cwd=str(BASE_DIR),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        os._exit(0)
    threading.Thread(target=go, daemon=True).start()


def active(payload=None):
    """
    Resolve the project for this request. The client sends it explicitly on
    every call rather than the server holding session state, because a stale
    server-side "current project" is exactly how you end up drafting a thesis
    section out of a client folder without noticing.
    """
    name = None
    if payload:
        name = payload.get("project")
    name = name or request.args.get("project")
    return writer.set_project(name or projects.UNFILED)


def plan_file():
    return writer.plan_path()


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class Job:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.events = []          # replay buffer for reconnects
        self.subscribers = []
        self.done = False
        self.lock = threading.Lock()

    def emit(self, event: dict):
        with self.lock:
            self.events.append(event)
            if event.get("type") in ("done", "error"):
                self.done = True
            for q in self.subscribers:
                q.put(event)

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            backlog = list(self.events)
            already_done = self.done
            if not already_done:
                self.subscribers.append(q)
        for e in backlog:
            yield e
        if already_done:
            return
        while True:
            e = q.get()
            yield e
            if e.get("type") in ("done", "error"):
                break
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


JOBS: dict = {}


def start_job(fn) -> str:
    job = Job()
    JOBS[job.id] = job

    def run():
        try:
            fn(job.emit)
        except Exception as e:
            traceback.print_exc()
            job.emit({"type": "error", "message": str(e)})

    threading.Thread(target=run, daemon=True).start()
    return job.id


@app.get("/api/jobs/<jid>/stream")
def job_stream(jid):
    job = JOBS.get(jid)
    if not job:
        return jsonify({"error": "no such job"}), 404

    def gen():
        for event in job.subscribe():
            yield f"data: {json.dumps(event)}\n\n"

    return Response(gen(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })


# ---------------------------------------------------------------------------
# Status and setup
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    active()
    have = writer.available_models()
    roles = [
        ("outliner", writer.OUTLINE_MODEL, "OLLAMA_OUTLINE_MODEL"),
        ("drafter", writer.DRAFT_MODEL, "OLLAMA_DRAFT_MODEL"),
        ("notes", writer.NOTES_MODEL, "OLLAMA_NOTES_MODEL"),
    ]
    normalized = {n if ":" in n else f"{n}:latest" for n in have}
    return jsonify({
        "chunks": collection.count(),
        "ollama": bool(have),
        "models": sorted(have),
        "roles": [{"role": r, "model": m, "var": v,
                   "ok": (m if ":" in m else f"{m}:latest") in normalized}
                  for r, m, v in roles],
        # Two separate checks, not one. A process can have the retry guard
        # loaded without having the (later) history sanitizer, and a single
        # combined flag would report "active" for both, hiding exactly the
        # gap that matters when only one of two sequential fixes has been
        # picked up by a restart.
        "ask_retry_guard": hasattr(ask, "_is_degenerate"),
        "ask_history_sanitizer": hasattr(ask, "_sanitize_history"),
        "project": writer.CURRENT_PROJECT,
        "projects": projects.stats(collection),
        "output": str(writer.output_dir()),
        "has_plan": plan_file().exists(),
    })


@app.get("/api/projects")
def api_projects():
    return jsonify({"projects": projects.stats(collection),
                    "root": str(projects.DOCUMENTS_ROOT)})


@app.post("/api/projects")
def api_create_project():
    name = (request.json.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400
    created = projects.create(name)
    return jsonify({"name": created,
                    "path": str(projects.DOCUMENTS_ROOT / created),
                    "projects": projects.stats(collection)})


@app.post("/api/rescan")
def rescan():
    def work(emit):
        from rag import scan_documents
        emit({"type": "stage", "stage": "scanning documents"})
        s = scan_documents(verbose=False)
        emit({"type": "done", "summary": {
            "new": s["new"], "updated": s["updated"], "removed": s["removed"],
            "chunks": collection.count()}})
    return jsonify({"job": start_job(work)})


# ---------------------------------------------------------------------------
# Outline
# ---------------------------------------------------------------------------

@app.get("/api/plan")
def get_plan():
    active()
    PLAN = plan_file()
    if not PLAN.exists():
        return jsonify({"outline": None})
    return jsonify({"outline": json.loads(PLAN.read_text())})


@app.post("/api/plan")
def save_plan():
    active(request.json)
    PLAN = plan_file()
    outline = request.json.get("outline")
    PLAN.write_text(json.dumps(outline, indent=2))
    return jsonify({"saved": str(PLAN)})


@app.delete("/api/plan")
def delete_plan():
    """
    Archive rather than destroy. An outline costs a few minutes of model time
    plus whatever editing you did to it, which is too much to lose to a
    misclick. The slate looks clean either way.
    """
    active()
    PLAN = plan_file()
    if not PLAN.exists():
        return jsonify({"deleted": False})

    try:
        title = json.loads(PLAN.read_text()).get("title") or "outline"
    except (json.JSONDecodeError, OSError):
        title = "outline"

    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())[:50].strip("-") or "outline"
    archive = writer.project_paths()["plans"]
    archive.mkdir(parents=True, exist_ok=True)
    dest = archive / f"{slug}-{datetime.now():%Y%m%d-%H%M%S}.json"
    PLAN.rename(dest)
    return jsonify({"deleted": True, "archived": str(dest)})


@app.get("/api/plans")
def list_plans():
    active()
    archive = writer.project_paths()["plans"]
    if not archive.exists():
        return jsonify({"plans": []})
    out = []
    for p in sorted(archive.glob("*.json"), key=lambda x: -x.stat().st_mtime)[:30]:
        try:
            o = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        out.append({"file": p.name, "title": o.get("title", p.stem),
                    "sections": len(o.get("sections", [])),
                    "modified": int(p.stat().st_mtime)})
    return jsonify({"plans": out})


@app.post("/api/plans/<name>/restore")
def restore_plan(name):
    active()
    PLAN = plan_file()
    src = writer.project_paths()["plans"] / Path(name).name
    if not src.exists():
        return jsonify({"error": "not found"}), 404
    outline = json.loads(src.read_text())
    PLAN.write_text(json.dumps(outline, indent=2))
    return jsonify({"outline": outline})


@app.post("/api/outline")
def make_outline():
    proj = active(request.json)
    topic = (request.json.get("topic") or "").strip()
    brief = (request.json.get("brief") or "").strip()
    if not topic:
        return jsonify({"error": "topic is required"}), 400

    def work(emit):
        emit({"type": "stage", "stage": "surveying your documents"})
        outline = writer.build_outline(topic, brief, project=proj)
        plan_file().write_text(json.dumps(outline, indent=2))
        emit({"type": "done", "outline": outline})

    return jsonify({"job": start_job(work)})


# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------

@app.post("/api/ask")
def api_ask():
    proj = active(request.json)
    body = request.json
    messages = body.get("messages") or []
    ground = bool(body.get("ground", True))
    # Omitting model entirely (rather than falling back to writer.DRAFT_MODEL
    # here) lets ask.ask() apply its own default, config.ASK_MODEL -- one
    # place decides what "no model specified" means, not this route.
    model = (body.get("model") or "").strip() or None
    turn_id = body.get("turn_id")

    if not messages or messages[-1].get("role") != "user":
        return jsonify({"error": "messages must end with a user turn"}), 400
    if not messages[-1].get("content", "").strip():
        return jsonify({"error": "empty message"}), 400

    def work(emit):
        result = ask.ask(
            messages, model, project=proj, ground=ground, turn_id=turn_id,
            on_token=lambda t: emit({"type": "token", "text": t}),
        )
        emit({"type": "done", **result})

    return jsonify({"job": start_job(work)})


@app.post("/api/summarize")
def api_summarize():
    """
    Search finds which files are relevant, then each match gets read
    straight off disk and summarized whole -- not from retrieved chunks.
    Same shared pipeline as MCP's summarize_documents tool and
    orchestrator.py's /summarize command (see summarize.py's module
    docstring), so "search then summarize" behaves identically wherever
    it's offered.
    """
    proj = active(request.json)
    body = request.json
    query = (body.get("query") or "").strip()
    model = (body.get("model") or "").strip() or None
    if not query:
        return jsonify({"error": "query is required"}), 400

    def work(emit):
        emit({"type": "stage", "stage": "searching documents"})
        sources = summarize.find_documents(query, project=proj)
        emit({"type": "found", "sources": sources})
        if not sources:
            emit({"type": "done", "results": []})
            return

        results = []
        for source in sources:
            emit({"type": "document_start", "source": source})
            path = summarize.resolve_path(source)
            try:
                result = summarize.summarize_file(
                    path, model=model,
                    on_token=lambda t, source=source: emit(
                        {"type": "token", "source": source, "text": t}),
                )
                result["source"] = source
            except (FileNotFoundError, ValueError) as e:
                result = {"source": source, "path": str(path), "error": str(e)}
            results.append(result)
            emit({"type": "document_done", "result": result})

        emit({"type": "done", "results": results})

    return jsonify({"job": start_job(work)})


@app.post("/api/draft")
def draft():
    proj = active(request.json)
    outline = request.json.get("outline")
    if not outline or not outline.get("sections"):
        return jsonify({"error": "outline with at least one section required"}), 400
    plan_file().write_text(json.dumps(outline, indent=2))

    def work(emit):
        writer.write_document(outline.get("title", ""), outline=outline,
                              on_event=emit, project=proj)

    return jsonify({"job": start_job(work)})


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@app.post("/api/bench")
def run_bench():
    body = request.json
    active(body)
    outline = body.get("outline")
    models = body.get("models") or []
    limit = body.get("sections")
    if not outline or not models:
        return jsonify({"error": "outline and at least one model required"}), 400
    if limit:
        outline = dict(outline)
        outline["sections"] = outline["sections"][:int(limit)]

    def work(emit):
        emit({"type": "stage", "stage": "gathering shared evidence"})
        registry, per_section = bench.prepare_evidence(outline)
        emit({"type": "evidence", "passages": len(registry.items)})

        results = []
        for m in models:
            emit({"type": "model_start", "model": m})
            try:
                r = bench.run_model(m, outline, per_section, registry)
                results.append(r)
                emit({"type": "model_done", "model": m,
                      "report": r["report"], "sections": r["sections"]})
            except Exception as e:
                emit({"type": "model_failed", "model": m, "message": str(e)})

        run_dir = bench.save_run(results, outline) if results else None
        emit({"type": "done",
              "results": [r["report"] for r in results],
              "drafts": {r["report"]["model"]: r["sections"] for r in results},
              "evidence": {e.marker: {"source": e.source, "start": e.start,
                                      "end": e.end, "text": e.text}
                           for e in registry.items},
              "dir": str(run_dir) if run_dir else None})

    return jsonify({"job": start_job(work)})


@app.get("/api/history")
def history():
    active()
    path = bench.bench_dir() / "history.jsonl"
    if not path.exists():
        return jsonify({"runs": [], "models": []})
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    agg = {}
    for r in rows:
        a = agg.setdefault(r["model"], {
            "model": r["model"], "runs": 0, "score": 0.0, "fabrications": 0,
            "grounding": 0.0, "distinct_3": 0.0, "minutes": 0.0})
        a["runs"] += 1
        a["score"] += r["score"]
        a["grounding"] += r["grounding"]
        a["distinct_3"] += r["distinct_3"]
        a["fabrications"] += r["citations"]["fabricated_count"]
        a["minutes"] += r.get("speed", {}).get("total_min", 0) or 0

    for a in agg.values():
        n = a["runs"]
        a["score"] = round(a["score"] / n, 1)
        a["grounding"] = round(a["grounding"] / n, 3)
        a["distinct_3"] = round(a["distinct_3"] / n, 3)
        a["minutes"] = round(a["minutes"] / n, 1)

    return jsonify({
        "runs": rows[-200:],
        "models": sorted(agg.values(), key=lambda a: -a["score"]),
        "topics": sorted({r.get("title") or "untitled" for r in rows}),
    })


@app.get("/api/documents")
def documents():
    active()
    OUT = writer.output_dir()
    if not OUT.exists():
        return jsonify({"documents": []})
    docs = sorted(OUT.glob("*.md"), key=lambda p: -p.stat().st_mtime)
    return jsonify({"documents": [
        {"name": p.stem, "path": str(p),
         "words": len(p.read_text().split()),
         "modified": int(p.stat().st_mtime)} for p in docs[:50]]})


@app.get("/api/documents/<name>")
def document(name):
    active()
    path = writer.output_dir() / f"{Path(name).stem}.md"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify({"name": path.stem, "markdown": path.read_text()})


# ---------------------------------------------------------------------------
# Updates -- see updater.py for what "update" and "rollback" actually mean
# (a real git commit and a real git checkout, not a hand-rolled scheme).
# Every route here that changes the repo state ends by scheduling a restart,
# never restarting inline -- the response needs to reach the browser first.
# ---------------------------------------------------------------------------

@app.get("/api/update/status")
def api_update_status():
    return jsonify(updater.status())


@app.get("/api/update/history")
def api_update_history():
    return jsonify({"releases": updater.history()})


@app.post("/api/update/apply")
def api_update_apply():
    body = request.json or {}
    message = (body.get("message") or "").strip() or None
    result = updater.apply_update(message)
    _delayed_restart()
    return jsonify({"applied": result, "restarting": True})


@app.post("/api/update/rollback")
def api_update_rollback():
    body = request.json or {}
    commit = (body.get("hash") or "").strip()
    if not commit:
        return jsonify({"error": "hash is required"}), 400
    updater.rollback(commit)
    _delayed_restart()
    return jsonify({"restarting": True})


if __name__ == "__main__":
    print(f"\n  Ask Ash on http://127.0.0.1:5111")
    print(f"  Index: {collection.count()} chunks across "
          f"{len(projects.discover())} project(s)")
    print(f"  Drafter: {writer.DRAFT_MODEL}")
    # This process only ever reads code at import time. If you've just
    # edited a file and restarted, this line is how you tell the restart
    # actually picked it up rather than guessing from behavior. If it says
    # "not loaded" after a restart, the file on disk and the file this
    # process imported are not the same file, check the path.
    print(f"  Ask retry guard: "
          f"{'active' if hasattr(ask, '_is_degenerate') else 'NOT LOADED — update ask.py'}\n")
    app.run(host="127.0.0.1", port=5111, threaded=True, debug=False)
