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
import configparser
from datetime import datetime
from email import policy
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import quote_plus

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
MAIL_CONFIG_PATH = BASE_DIR / ".offlineimaprc"
MAIL_CONFIG_EXAMPLE_PATH = BASE_DIR / "offlineimaprc.example"
DEFAULT_MAILDIR_ROOT = BASE_DIR / "Offline Email"

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

MEMORY_AVAILABLE = False
try:
    sys.path.insert(0, str(BASE_DIR / "memory"))
    from memory_client import (
        clear_ask_conversation,
        get_search_criteria,
        get_setting,
        list_ask_conversations,
        list_retrieval_misses,
        load_ask_conversation,
        new_ask_conversation,
        record_retrieval_miss,
        restore_ask_conversation,
        save_ask_conversation,
        set_setting,
        upsert_search_criterion,
    )
    MEMORY_AVAILABLE = True
except Exception as _e:
    print(f"  Tuning storage unavailable ({_e}). Everything else runs normally.")


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
    if name == projects.ALL:
        return projects.ALL
    return writer.set_project(name or projects.UNFILED)


def plan_file():
    return writer.plan_path()


def _specific_project_required():
    return jsonify({"error": "choose a specific project for this action"}), 400


def _env_path() -> Path:
    return BASE_DIR / ".env"


def _set_env_value(key: str, value: str):
    path = _env_path()
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    prefix = f"{key}="
    updated = False
    out = []
    for line in lines:
        if line.startswith(prefix):
            out.append(f"{key}={value}")
            updated = True
        else:
            out.append(line)
    if not updated:
        out.append(f"{key}={value}")
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    os.environ[key] = value


def _mail_auth_lines() -> list:
    if not MAIL_CONFIG_PATH.exists():
        return []
    lines = []
    for line in MAIL_CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if (
            stripped.startswith("remotepass")
            or stripped.startswith("remotepasseval")
            or stripped.startswith("oauth2_")
        ):
            lines.append(line)
    return lines


def _mail_config_parser() -> configparser.ConfigParser:
    parser = configparser.RawConfigParser()
    parser.optionxform = str
    if MAIL_CONFIG_PATH.exists():
        parser.read(MAIL_CONFIG_PATH)
    return parser


def _mail_settings_from_config() -> dict:
    parser = _mail_config_parser()
    account = parser["general"].get("accounts", "Personal") if parser.has_section("general") else "Personal"
    account = account.split(",", 1)[0].strip() or "Personal"
    account_section = f"Account {account}"
    local_repo = parser[account_section].get("localrepository", "PersonalLocal") if parser.has_section(account_section) else "PersonalLocal"
    remote_repo = parser[account_section].get("remoterepository", "PersonalRemote") if parser.has_section(account_section) else "PersonalRemote"
    local_section = f"Repository {local_repo}"
    remote_section = f"Repository {remote_repo}"
    localfolders = (
        parser[local_section].get("localfolders", str(DEFAULT_MAILDIR_ROOT / account))
        if parser.has_section(local_section) else str(DEFAULT_MAILDIR_ROOT / account)
    )
    folderfilter = (
        parser[remote_section].get("folderfilter", "")
        if parser.has_section(remote_section) else ""
    )
    folders = ""
    match = re.search(r"\[(.*)\]", folderfilter)
    if match:
        folders = ",".join(
            item.strip().strip("\"'")
            for item in match.group(1).split(",")
            if item.strip()
        )
    return {
        "account": account,
        "remotehost": (
            parser[remote_section].get("remotehost", "imap.example.com")
            if parser.has_section(remote_section) else "imap.example.com"
        ),
        "remoteuser": (
            parser[remote_section].get("remoteuser", "you@example.com")
            if parser.has_section(remote_section) else "you@example.com"
        ),
        "ssl": (
            parser[remote_section].get("ssl", "yes")
            if parser.has_section(remote_section) else "yes"
        ),
        "sslcacertfile": (
            parser[remote_section].get("sslcacertfile", "/opt/homebrew/etc/openssl@3/cert.pem")
            if parser.has_section(remote_section) else "/opt/homebrew/etc/openssl@3/cert.pem"
        ),
        "maxage": (
            parser[account_section].get("maxage", "365")
            if parser.has_section(account_section) else "365"
        ),
        "maxsize": (
            parser[account_section].get("maxsize", "2000000")
            if parser.has_section(account_section) else "2000000"
        ),
        "folders": folders or "INBOX,Sent",
        "localfolders": localfolders,
        "maildir_roots": os.getenv("IMAP_MAILDIR_ROOTS", str(DEFAULT_MAILDIR_ROOT)),
        "has_auth": bool(_mail_auth_lines()),
    }


def _write_mail_config(settings: dict):
    account = re.sub(r"[^A-Za-z0-9._-]+", "-", settings["account"]).strip("-.") or "Personal"
    local_repo = f"{account}Local"
    remote_repo = f"{account}Remote"
    folders = [
        f.strip() for f in str(settings.get("folders") or "").replace("\n", ",").split(",")
        if f.strip()
    ]
    folder_list = ", ".join(repr(f) for f in folders[:50])
    auth_lines = _mail_auth_lines()
    localfolders = str(DEFAULT_MAILDIR_ROOT / account)
    config_lines = [
        "[general]",
        f"accounts = {account}",
        f"metadata = {DEFAULT_MAILDIR_ROOT}/.metadata",
        "ui = basic",
        "",
        f"[Account {account}]",
        f"localrepository = {local_repo}",
        f"remoterepository = {remote_repo}",
        f"maxage = {settings['maxage']}",
        f"maxsize = {settings['maxsize']}",
        "",
        f"[Repository {local_repo}]",
        "type = Maildir",
        f"localfolders = {localfolders}",
        "",
        f"[Repository {remote_repo}]",
        "type = IMAP",
        f"remotehost = {settings['remotehost']}",
        f"remoteuser = {settings['remoteuser']}",
        f"ssl = {settings['ssl']}",
        f"sslcacertfile = {settings['sslcacertfile']}",
    ]
    if folders:
        config_lines.append(f"folderfilter = lambda foldername: foldername in [{folder_list}]")
    if auth_lines:
        config_lines.extend(["", *auth_lines])
    else:
        config_lines.extend([
            "",
            "# Add authentication locally, for example:",
            "# remotepass = your-app-password",
            "# remotepasseval = get_password(\"you@example.com\")",
        ])
    MAIL_CONFIG_PATH.write_text("\n".join(config_lines).rstrip() + "\n", encoding="utf-8")
    DEFAULT_MAILDIR_ROOT.mkdir(parents=True, exist_ok=True)
    (DEFAULT_MAILDIR_ROOT / ".metadata").mkdir(parents=True, exist_ok=True)
    Path(localfolders).mkdir(parents=True, exist_ok=True)
    _set_env_value("IMAP_MAILDIR_ROOTS", str(DEFAULT_MAILDIR_ROOT))


def _mail_source_meta(source: str) -> dict:
    if not source.startswith("mail/"):
        return {}
    try:
        data = collection.get(where={"source": source}, include=["metadatas"])
    except Exception:
        return {}
    for meta in data.get("metadatas") or []:
        if meta.get("source") == source:
            return meta
    return {}


def _mail_source_path(source: str) -> Path | None:
    meta = _mail_source_meta(source)
    path_value = meta.get("email_local_path") or ""
    if not path_value:
        return None
    try:
        path = Path(path_value).resolve()
        roots = [
            Path(p.strip()).resolve()
            for p in os.getenv("IMAP_MAILDIR_ROOTS", str(DEFAULT_MAILDIR_ROOT)).split(",")
            if p.strip()
        ]
        if roots and not any(path.is_relative_to(root) for root in roots):
            return None
        return path if path.exists() else None
    except Exception:
        return None


def _mail_attachment_items(source: str, include_payload: bool = False) -> list:
    path = _mail_source_path(source)
    if not path:
        return []
    try:
        msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    except Exception:
        return []

    items = []
    for part_index, part in enumerate(msg.walk()):
        if part.is_multipart():
            continue
        raw_filename = part.get_filename()
        filename = raw_filename or f"attachment-{part_index}"
        content_type = part.get_content_type() or "application/octet-stream"
        disposition = (part.get_content_disposition() or "").lower()
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        if not (raw_filename or disposition == "attachment" or content_type.startswith("image/")):
            continue
        item = {
            "part": part_index,
            "filename": filename,
            "content_type": content_type,
            "size": len(payload),
            "image": content_type.startswith("image/"),
            "url": (
                "/api/mail/attachment"
                f"?source={quote_plus(source)}&part={part_index}"
            ),
        }
        if include_payload:
            item["payload"] = payload
        items.append(item)
    return items


def _attachments_for_evidence(evidence: dict, text: str = "") -> list:
    seen = set()
    attachments = []
    cited = {
        marker for marker in (evidence or {})
        if not text or f"[{marker}]" in text
    }
    for marker, ev in (evidence or {}).items():
        if marker not in cited:
            continue
        source = ev.get("source") or ""
        if not source.startswith("mail/"):
            continue
        for item in _mail_attachment_items(source):
            key = (item["filename"], item["size"])
            if key in seen:
                continue
            seen.add(key)
            attachments.append({**item, "source": source})
    return attachments


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
    proj = active()
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
        "project": proj,
        "projects": projects.stats(collection),
        "output": None if proj == projects.ALL else str(writer.output_dir()),
        "has_plan": False if proj == projects.ALL else plan_file().exists(),
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
            "unchanged": s["unchanged"], "chunks": collection.count()}})
    return jsonify({"job": start_job(work)})


@app.get("/api/mail/config")
def api_mail_config():
    if not MAIL_CONFIG_PATH.exists():
        _write_mail_config(_mail_settings_from_config())
    return jsonify({
        "settings": _mail_settings_from_config(),
        "config_path": str(MAIL_CONFIG_PATH),
        "maildir_root": str(DEFAULT_MAILDIR_ROOT),
        "offlineimap": str(BASE_DIR / ".venv-offlineimap" / "bin" / "offlineimap"),
    })


@app.post("/api/mail/config")
def api_mail_config_save():
    body = request.json or {}
    settings = {
        "account": (body.get("account") or "Personal").strip(),
        "remotehost": (body.get("remotehost") or "").strip(),
        "remoteuser": (body.get("remoteuser") or "").strip(),
        "ssl": "yes" if body.get("ssl") in (True, "yes", "on", "true", "1", 1) else "no",
        "sslcacertfile": (body.get("sslcacertfile") or "").strip(),
        "maxage": str(body.get("maxage") or "").strip(),
        "maxsize": str(body.get("maxsize") or "").strip(),
        "folders": (body.get("folders") or "").strip(),
    }
    if not settings["remotehost"]:
        return jsonify({"error": "IMAP host is required"}), 400
    if not settings["remoteuser"]:
        return jsonify({"error": "email address is required"}), 400
    if not settings["sslcacertfile"]:
        return jsonify({"error": "certificate file path is required"}), 400
    try:
        maxage = int(settings["maxage"])
        maxsize = int(settings["maxsize"])
    except ValueError:
        return jsonify({"error": "age and size limits must be whole numbers"}), 400
    if not 1 <= maxage <= 3650:
        return jsonify({"error": "age limit must be between 1 and 3650 days"}), 400
    if not 10000 <= maxsize <= 50000000:
        return jsonify({"error": "size limit must be between 10 KB and 50 MB"}), 400
    if not settings["folders"]:
        return jsonify({"error": "at least one folder is required"}), 400
    _write_mail_config(settings)
    return jsonify({
        "settings": _mail_settings_from_config(),
        "config_path": str(MAIL_CONFIG_PATH),
    })


@app.post("/api/mail/rescan")
def api_mail_rescan():
    body = request.json or {}
    force = bool(body.get("force", False))

    def work(emit):
        from rag import scan_mailboxes
        emit({"type": "stage", "stage": "scanning mailboxes"})
        s = scan_mailboxes(verbose=False, force=force)
        emit({"type": "done", "summary": {
            "new": s["new"], "updated": s["updated"], "removed": s["removed"],
            "unchanged": s["unchanged"], "roots": s.get("roots", []),
            "maildirs": s.get("maildirs", []), "chunks": collection.count()}})
    return jsonify({"job": start_job(work)})


@app.get("/api/mail/attachments")
def api_mail_attachments():
    source = (request.args.get("source") or "").strip()
    return jsonify({"attachments": _mail_attachment_items(source)})


@app.get("/api/mail/attachment")
def api_mail_attachment():
    source = (request.args.get("source") or "").strip()
    try:
        part_index = int(request.args.get("part") or "-1")
    except ValueError:
        part_index = -1
    for item in _mail_attachment_items(source, include_payload=True):
        if item["part"] != part_index:
            continue
        payload = item.pop("payload")
        headers = {
            "Content-Disposition": f"inline; filename=\"{item['filename']}\"",
            "X-Content-Type-Options": "nosniff",
        }
        return Response(payload, mimetype=item["content_type"], headers=headers)
    return jsonify({"error": "attachment not found"}), 404


# ---------------------------------------------------------------------------
# Retrieval tuning
# ---------------------------------------------------------------------------

CRITERIA_TYPES = {
    "stopword", "low_signal", "source_low_signal", "section_noise",
    "domain_trigger", "domain_term", "document_section", "genre_marker",
    "theme_marker", "subject_stop_label", "ask_route", "genre_alias",
    "source_lookup_stopword", "relation_target", "redaction_profile",
    "redaction_rule", "redaction_protection",
    # Assignment-formatting requirements (Ask screen) -- see ask.py's
    # _has_requirement_trigger / _requirement_lines and DMAIC.md.
    "requirement_trigger", "requirement_text",
}
TUNING_SETTING_KEYS = {
    "rag_chunk_size", "rag_chunk_overlap", "rag_auto_reindex_on_tuning",
    "rag_supported_extensions", "rag_ignored_dirs",
    "rag_topic_min_domain_hits", "rag_external_search_enabled",
    "rag_redaction_profiles",
}


def _memory_required():
    if MEMORY_AVAILABLE:
        return None
    return jsonify({"error": "memory database is not available"}), 503


def _clean_chat_message(message: dict) -> dict:
    if not isinstance(message, dict):
        return {}
    role = message.get("role")
    if role not in {"user", "assistant"}:
        return {}
    return {"role": role, "content": str(message.get("content") or "")}


def _clean_chat_messages(messages: list) -> list:
    cleaned = [_clean_chat_message(m) for m in (messages or [])]
    return [m for m in cleaned if m.get("role") and m.get("content", "").strip()]


@app.get("/api/tuning")
def api_tuning():
    unavailable = _memory_required()
    if unavailable:
        return unavailable
    proj = active()
    return jsonify({
        "criteria": get_search_criteria(enabled_only=False),
        "settings": {
            "rag_chunk_size": get_setting("rag_chunk_size", "500"),
            "rag_chunk_overlap": get_setting("rag_chunk_overlap", "100"),
            "rag_auto_reindex_on_tuning": get_setting(
                "rag_auto_reindex_on_tuning", "off"),
            "rag_external_search_enabled": get_setting(
                "rag_external_search_enabled", "off"),
            "rag_redaction_profiles": get_setting(
                "rag_redaction_profiles", "legal_privileged"),
            "rag_topic_min_domain_hits": get_setting(
                "rag_topic_min_domain_hits", "5"),
            "rag_supported_extensions": get_setting(
                "rag_supported_extensions",
                ".arw,.bmp,.cr2,.cr3,.dng,.docx,.gif,.jpeg,.jpg,.md,.nef,.orf,.pdf,.png,.pptx,.rw2,.tiff,.txt,.xlsx",
            ),
            "rag_ignored_dirs": get_setting(
                "rag_ignored_dirs",
                ".DS_Store,.git,.obsidian,.trash,.venv,.writer,__pycache__,node_modules,venv",
            ),
        },
        "misses": list_retrieval_misses(
            project=None if proj == projects.ALL else proj, limit=50),
    })


@app.post("/api/tuning/criteria")
def api_tuning_criteria():
    unavailable = _memory_required()
    if unavailable:
        return unavailable
    body = request.json or {}
    criteria_type = (body.get("criteria_type") or "").strip().lower()
    group_name = (body.get("group_name") or "").strip().lower()
    # requirement_text rows are full instruction sentences, not lowercase
    # matching keywords -- casing (e.g. "(Author, 2024)", "DOI") is part of
    # the content, so it's preserved rather than lowercased like other
    # criteria types. See memory_client.CASE_PRESERVING_CRITERIA_TYPES.
    raw_term = (body.get("term") or "").strip()
    if criteria_type == "requirement_text":
        term = " ".join(raw_term.split())
    else:
        term = " ".join(raw_term.lower().split())
    notes = (body.get("notes") or "").strip() or None
    enabled = bool(body.get("enabled", True))
    try:
        weight = float(body.get("weight", 1.0))
    except (TypeError, ValueError):
        return jsonify({"error": "weight must be a number"}), 400

    if criteria_type not in CRITERIA_TYPES:
        return jsonify({"error": "unknown criteria type"}), 400
    if not term:
        return jsonify({"error": "term is required"}), 400
    if (
        criteria_type.startswith("domain_")
        or criteria_type in {
            "ask_route", "genre_alias", "redaction_rule",
            "redaction_protection", "requirement_trigger", "requirement_text",
        }
    ) and not group_name:
        return jsonify({"error": f"{criteria_type} criteria need a group"}), 400

    upsert_search_criterion(
        criteria_type, term, group_name=group_name, weight=weight,
        enabled=enabled, notes=notes,
    )
    return jsonify({"criteria": get_search_criteria(enabled_only=False)})


@app.post("/api/tuning/settings")
def api_tuning_settings():
    unavailable = _memory_required()
    if unavailable:
        return unavailable
    body = request.json or {}
    saved = {}
    for key in TUNING_SETTING_KEYS:
        if key not in body:
            continue
        if key in {"rag_auto_reindex_on_tuning", "rag_external_search_enabled"}:
            value = "on" if body.get(key) in (True, "on", "true", "1", 1) else "off"
            saved[key] = value
            set_setting(key, value)
            continue
        if key == "rag_redaction_profiles":
            raw = str(body.get(key) or "")
            values = [
                item.strip().lower()
                for item in raw.replace("\n", ",").split(",")
                if item.strip()
            ]
            value = ",".join(dict.fromkeys(values))
            saved[key] = value
            set_setting(key, value)
            continue
        if key in {"rag_supported_extensions", "rag_ignored_dirs"}:
            raw = str(body.get(key) or "")
            values = [
                item.strip().lower()
                for item in raw.replace("\n", ",").split(",")
                if item.strip()
            ]
            if key == "rag_supported_extensions":
                values = [v if v.startswith(".") else f".{v}" for v in values]
            value = ",".join(dict.fromkeys(values))
            if not value:
                return jsonify({"error": f"{key} cannot be empty"}), 400
            saved[key] = value
            set_setting(key, value)
            continue
        try:
            value = int(body[key])
        except (TypeError, ValueError):
            return jsonify({"error": f"{key} must be a whole number"}), 400
        if key == "rag_chunk_size" and not 150 <= value <= 2000:
            return jsonify({"error": "chunk size must be between 150 and 2000"}), 400
        if key == "rag_chunk_overlap" and not 0 <= value <= 500:
            return jsonify({"error": "chunk overlap must be between 0 and 500"}), 400
        if key == "rag_topic_min_domain_hits" and not 1 <= value <= 100:
            return jsonify({"error": "minimum topic hits must be between 1 and 100"}), 400
        saved[key] = str(value)
        set_setting(key, str(value))
    return jsonify({"settings": saved})


@app.post("/api/tuning/reindex")
def api_tuning_reindex():
    unavailable = _memory_required()
    if unavailable:
        return unavailable
    proj = active(request.json or {})

    def work(emit):
        from rag import scan_documents
        emit({"type": "stage", "stage": "re-indexing with current settings"})
        s = scan_documents(verbose=False, force=True, overwrite_profiles=True)
        emit({"type": "done", "summary": {
            "new": s["new"], "updated": s["updated"], "removed": s["removed"],
            "unchanged": s["unchanged"], "chunks": collection.count(),
            "project": proj,
        }})

    return jsonify({"job": start_job(work)})


@app.post("/api/tuning/misses")
def api_tuning_misses():
    unavailable = _memory_required()
    if unavailable:
        return unavailable
    body = request.json or {}
    proj = active(body)
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    miss_id = record_retrieval_miss(
        proj,
        query,
        expected_source=(body.get("expected_source") or "").strip() or None,
        actual_source=(body.get("actual_source") or "").strip() or None,
        notes=(body.get("notes") or "").strip() or None,
    )
    return jsonify({
        "id": miss_id,
        "misses": list_retrieval_misses(project=proj, limit=50),
    })


# ---------------------------------------------------------------------------
# Outline
# ---------------------------------------------------------------------------

@app.get("/api/plan")
def get_plan():
    if active() == projects.ALL:
        return jsonify({"outline": None})
    PLAN = plan_file()
    if not PLAN.exists():
        return jsonify({"outline": None})
    return jsonify({"outline": json.loads(PLAN.read_text())})


@app.post("/api/plan")
def save_plan():
    if active(request.json) == projects.ALL:
        return _specific_project_required()
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
    if active() == projects.ALL:
        return _specific_project_required()
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
    if active() == projects.ALL:
        return jsonify({"plans": []})
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
    if active() == projects.ALL:
        return _specific_project_required()
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
    if proj == projects.ALL:
        return _specific_project_required()
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

@app.get("/api/ask/conversation")
def api_ask_conversation():
    unavailable = _memory_required()
    if unavailable:
        return unavailable
    proj = active()
    return jsonify(load_ask_conversation(proj))


@app.get("/api/ask/conversations")
def api_ask_conversations():
    unavailable = _memory_required()
    if unavailable:
        return unavailable
    proj = active()
    return jsonify({"conversations": list_ask_conversations(proj)})


@app.post("/api/ask/conversation")
def api_save_ask_conversation():
    unavailable = _memory_required()
    if unavailable:
        return unavailable
    proj = active(request.json)
    body = request.json or {}
    messages = _clean_chat_messages(body.get("messages") or [])
    try:
        turn_seq = int(body.get("turn_seq") or 0)
    except (TypeError, ValueError):
        turn_seq = 0
    save_ask_conversation(proj, messages, turn_seq=turn_seq)
    return jsonify({"ok": True})


@app.post("/api/ask/conversation/new")
def api_new_ask_conversation():
    unavailable = _memory_required()
    if unavailable:
        return unavailable
    proj = active(request.json)
    return jsonify(new_ask_conversation(proj))


@app.post("/api/ask/conversation/restore")
def api_restore_ask_conversation():
    unavailable = _memory_required()
    if unavailable:
        return unavailable
    proj = active(request.json)
    body = request.json or {}
    restored = restore_ask_conversation(proj, int(body.get("id") or 0))
    if restored is None:
        return jsonify({"error": "conversation not found"}), 404
    return jsonify(restored)


@app.post("/api/ask/conversation/clear")
def api_clear_ask_conversation():
    unavailable = _memory_required()
    if unavailable:
        return unavailable
    proj = active(request.json)
    clear_ask_conversation(proj)
    return jsonify({"ok": True})


@app.post("/api/ask")
def api_ask():
    proj = active(request.json)
    body = request.json
    messages = body.get("messages") or []
    ground_value = body.get("ground", True)
    if isinstance(ground_value, str):
        ground = ground_value.strip().lower() not in {"0", "false", "no", "off"}
    else:
        ground = bool(ground_value)
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
            external_policy="pull",
        )
        result["attachments"] = _attachments_for_evidence(
            result.get("evidence") or {}, result.get("text") or "")
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
            try:
                result = summarize.summarize_source(
                    source, model=model,
                    on_token=lambda t, source=source: emit(
                        {"type": "token", "source": source, "text": t}),
                )
                result["source"] = source
            except (FileNotFoundError, ValueError) as e:
                path = source if source.startswith("mail/") else str(
                    summarize.resolve_path(source))
                result = {"source": source, "path": str(path), "error": str(e)}
            results.append(result)
            emit({"type": "document_done", "result": result})

        emit({"type": "done", "results": results})

    return jsonify({"job": start_job(work)})


@app.post("/api/draft")
def draft():
    proj = active(request.json)
    if proj == projects.ALL:
        return _specific_project_required()
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
