"""
memory_client.py -- thin wrapper around libsql_client for logging RAG
sessions to memory-db.

This is intentionally the *only* place that writes to the sessions/turns/
quality_scores tables, so the incognito guarantee lives in one function
instead of being re-implemented (and potentially forgotten) at every call
site. If incognito=True, start_session() returns a session object whose
log_turn()/close() calls are no-ops -- nothing touches the network, nothing
touches the database. Not "logged then filtered out later," never sent.

Usage from orchestrator.py or writer.py:

    from memory_client import start_session

    session = start_session(project="GCU", machine="mac", mode="qa",
                             incognito=False)
    ...
    session.log_turn(question=q, answer=a, model="qwen3:32b",
                      chunk_ids=[m["source"] + "::" + str(i) for i, m in ...])
    ...
    session.close()

Nothing in this file is wired into orchestrator.py or writer.py yet --
that's a separate, deliberate step so each call site can decide what's
worth logging rather than everything being captured by default.
"""

import json
import mimetypes
import socket
import uuid

import libsql_client

from memory_config import LIBSQL_URL, LIBSQL_AUTH_TOKEN


class _NullSession:
    """Returned when incognito=True. Every method is a no-op."""

    def log_turn(self, *args, **kwargs):
        pass

    def close(self):
        pass


class _Session:
    def __init__(self, client, session_id):
        self._client = client
        self._session_id = session_id

    def log_turn(self, question: str, answer: str = None, model: str = None,
                 chunk_ids: list = None) -> int:
        result = self._client.execute(
            "INSERT INTO turns (session_id, question, answer, model, chunk_ids) "
            "VALUES (?, ?, ?, ?, ?)",
            [self._session_id, question, answer, model,
             json.dumps(chunk_ids) if chunk_ids else None],
        )
        return result.last_insert_rowid

    def log_quality(self, turn_id: int, fabrication: float = None,
                     grounding: float = None, self_repetition: float = None,
                     cross_section_bleed: float = None,
                     length_adherence: float = None):
        self._client.execute(
            "INSERT INTO quality_scores "
            "(turn_id, fabrication, grounding, self_repetition, "
            " cross_section_bleed, length_adherence) VALUES (?, ?, ?, ?, ?, ?)",
            [turn_id, fabrication, grounding, self_repetition,
             cross_section_bleed, length_adherence],
        )

    def close(self):
        self._client.execute(
            "UPDATE sessions SET ended_at = datetime('now') WHERE id = ?",
            [self._session_id],
        )
        self._client.close()


def start_session(project: str, machine: str, mode: str, incognito: bool = False):
    """
    project: the rag.py project folder this session relates to, or None
    machine: 'mac' | 'alice'
    mode:    'qa' | 'draft' | 'bench'
    incognito: if True, returns a _NullSession -- nothing is written,
               nothing is sent over the network, full stop.
    """
    if incognito:
        return _NullSession()

    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    result = client.execute(
        "INSERT INTO sessions (project, machine, mode, incognito) VALUES (?, ?, ?, 0)",
        [project, machine, mode],
    )
    return _Session(client, result.last_insert_rowid)


def recent_turns(n: int = 10, project: str = None):
    """Convenience read helper for a quick 'what have I been asking' check."""
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        if project:
            result = client.execute(
                "SELECT t.asked_at, s.project, t.question, t.answer "
                "FROM turns t JOIN sessions s ON s.id = t.session_id "
                "WHERE s.project = ? ORDER BY t.asked_at DESC LIMIT ?",
                [project, n],
            )
        else:
            result = client.execute(
                "SELECT t.asked_at, s.project, t.question, t.answer "
                "FROM turns t JOIN sessions s ON s.id = t.session_id "
                "ORDER BY t.asked_at DESC LIMIT ?",
                [n],
            )
        return [dict(zip(result.columns, row)) for row in result.rows]
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Ask conversation state -- durable visible chat for the local web UI
# ---------------------------------------------------------------------------

def load_ask_conversation(project: str):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        result = client.execute(
            "SELECT messages, turn_seq, updated_at, cleared_at "
            "FROM ask_conversations WHERE project = ?",
            [project or ""],
        )
        if not result.rows:
            return {
                "messages": [],
                "turn_seq": 0,
                "updated_at": None,
                "cleared_at": None,
            }
        row = dict(zip(result.columns, result.rows[0]))
        try:
            messages = json.loads(row.get("messages") or "[]")
        except Exception:
            messages = []
        return {
            "messages": messages if isinstance(messages, list) else [],
            "turn_seq": int(row.get("turn_seq") or 0),
            "updated_at": row.get("updated_at"),
            "cleared_at": row.get("cleared_at"),
        }
    finally:
        client.close()


def save_ask_conversation(project: str, messages: list, turn_seq: int = 0):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "INSERT INTO ask_conversations "
            "(project, messages, turn_seq, updated_at, cleared_at) "
            "VALUES (?, ?, ?, datetime('now'), NULL) "
            "ON CONFLICT(project) DO UPDATE SET "
            "messages=excluded.messages, turn_seq=excluded.turn_seq, "
            "updated_at=datetime('now'), cleared_at=NULL",
            [project or "", json.dumps(messages or []), int(turn_seq or 0)],
        )
    finally:
        client.close()


def _ask_conversation_title(messages: list) -> str:
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        title = " ".join(str(message.get("content") or "").split())
        if title:
            return title[:80]
    return "Untitled conversation"


def new_ask_conversation(project: str):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        result = client.execute(
            "SELECT messages, turn_seq FROM ask_conversations WHERE project = ?",
            [project or ""],
        )
        archived_id = None
        if result.rows:
            row = dict(zip(result.columns, result.rows[0]))
            try:
                messages = json.loads(row.get("messages") or "[]")
            except Exception:
                messages = []
            messages = messages if isinstance(messages, list) else []
            if any(str(m.get("content") or "").strip()
                   for m in messages if isinstance(m, dict)):
                inserted = client.execute(
                    "INSERT INTO ask_conversation_history "
                    "(project, title, messages, turn_seq) VALUES (?, ?, ?, ?)",
                    [project or "", _ask_conversation_title(messages),
                     json.dumps(messages), int(row.get("turn_seq") or 0)],
                )
                archived_id = inserted.last_insert_rowid
        client.execute(
            "INSERT INTO ask_conversations "
            "(project, messages, turn_seq, updated_at, cleared_at) "
            "VALUES (?, '[]', 0, datetime('now'), NULL) "
            "ON CONFLICT(project) DO UPDATE SET "
            "messages='[]', turn_seq=0, updated_at=datetime('now'), "
            "cleared_at=NULL",
            [project or ""],
        )
        return {"ok": True, "archived_id": archived_id}
    finally:
        client.close()


def list_ask_conversations(project: str, limit: int = 30):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        result = client.execute(
            "SELECT id, title, turn_seq, created_at, archived_at "
            "FROM ask_conversation_history WHERE project = ? "
            "ORDER BY archived_at DESC LIMIT ?",
            [project or "", int(limit or 30)],
        )
        return [dict(zip(result.columns, row)) for row in result.rows]
    finally:
        client.close()


def restore_ask_conversation(project: str, conversation_id: int):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        result = client.execute(
            "SELECT messages, turn_seq FROM ask_conversation_history "
            "WHERE project = ? AND id = ?",
            [project or "", int(conversation_id)],
        )
        if not result.rows:
            return None
        row = dict(zip(result.columns, result.rows[0]))
        messages = row.get("messages") or "[]"
        turn_seq = int(row.get("turn_seq") or 0)
        client.execute(
            "INSERT INTO ask_conversations "
            "(project, messages, turn_seq, updated_at, cleared_at) "
            "VALUES (?, ?, ?, datetime('now'), NULL) "
            "ON CONFLICT(project) DO UPDATE SET "
            "messages=excluded.messages, turn_seq=excluded.turn_seq, "
            "updated_at=datetime('now'), cleared_at=NULL",
            [project or "", messages, turn_seq],
        )
        try:
            parsed = json.loads(messages)
        except Exception:
            parsed = []
        return {"messages": parsed if isinstance(parsed, list) else [],
                "turn_seq": turn_seq}
    finally:
        client.close()


def clear_ask_conversation(project: str):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "INSERT INTO ask_conversations "
            "(project, messages, turn_seq, updated_at, cleared_at) "
            "VALUES (?, '[]', 0, datetime('now'), datetime('now')) "
            "ON CONFLICT(project) DO UPDATE SET "
            "messages='[]', turn_seq=0, updated_at=datetime('now'), "
            "cleared_at=datetime('now')",
            [project or ""],
        )
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Verified citations -- durable ground truth, independent of sessions/turns
# and independent of whatever chroma_db's ranking does or doesn't surface.
# See schema.sql for the reasoning. Plain module-level functions, not part
# of the _Session class, since these aren't tied to any one Q&A session --
# a citation gets recorded once and is meant to outlive every session that
# will ever look it up.
# ---------------------------------------------------------------------------

def record_citation(source: str, title: str = None, authors: str = None,
                    source_line: str = None, publication_year: int = None,
                    verified_how: str = "manual") -> int:
    """
    source is rag.py's relative path, e.g. "GCU/EBSCO-FullText-07_26_2026.pdf" --
    matching that format is what lets this line up with chunk_ids elsewhere.
    Upserts on source (ON CONFLICT), so re-verifying the same file just
    refreshes the record rather than erroring or duplicating.
    """
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        result = client.execute(
            "INSERT INTO citations (source, title, authors, source_line, "
            " publication_year, verified_how) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source) DO UPDATE SET "
            " title=excluded.title, authors=excluded.authors, "
            " source_line=excluded.source_line, "
            " publication_year=excluded.publication_year, "
            " verified_how=excluded.verified_how, "
            " verified_at=datetime('now')",
            [source, title, authors, source_line, publication_year, verified_how],
        )
        return result.last_insert_rowid
    finally:
        client.close()


def find_citation(author: str = None, source: str = None):
    """
    Look up a verified citation by author (substring match) or exact source
    path. This is the fallback a query like "articles by Tye" should check
    when chroma's own ranking comes up empty -- a direct, un-ranked lookup
    against known-good facts instead of hoping the vector search finds them.
    """
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        if source:
            result = client.execute(
                "SELECT * FROM citations WHERE source = ?", [source])
        elif author:
            result = client.execute(
                "SELECT * FROM citations WHERE authors LIKE ?", [f"%{author}%"])
        else:
            return []
        return [dict(zip(result.columns, row)) for row in result.rows]
    finally:
        client.close()


def search_citations(query_words: list, project_prefix: str = None):
    """
    Search verified citations across source path, title, authors, and source
    line. This is intentionally lexical: citation records are short, curated
    facts, so exact title/author words should outrank Chroma similarity.
    """
    words = [w.lower() for w in query_words if w and len(w) >= 2]
    if not words:
        return []

    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        fields = ["source", "title", "authors", "source_line"]
        clauses = []
        params = []
        for word in words:
            like = f"%{word}%"
            clauses.append("(" + " OR ".join(
                [f"lower(coalesce({field}, '')) LIKE ?" for field in fields]
            ) + ")")
            params.extend([like] * len(fields))

        sql = "SELECT * FROM citations WHERE (" + " OR ".join(clauses) + ")"
        if project_prefix:
            sql += " AND source LIKE ?"
            params.append(f"{project_prefix}%")

        result = client.execute(sql, params)
        rows = [dict(zip(result.columns, row)) for row in result.rows]
        rows.sort(
            key=lambda row: sum(
                (row.get(field) or "").lower().count(word)
                for word in words
                for field in fields
            ),
            reverse=True,
        )
        return rows
    finally:
        client.close()


def flag_false_positive(turn_id: int, notes: str = None):
    """
    Mark a turn's answer as wrong, for later review. Broader than
    record_retrieval_gap -- this covers any bad answer, not just the
    specific "the right source existed and search() missed it" case.
    """
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "UPDATE turns SET flagged_false_positive = 1, flag_notes = ?, "
            "flagged_at = datetime('now') WHERE id = ?",
            [notes, turn_id],
        )
    finally:
        client.close()


def flagged_turns(n: int = 20):
    """Review helper: everything flagged as a false positive, most recent first."""
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        result = client.execute(
            "SELECT id, asked_at, question, answer, flag_notes, flagged_at "
            "FROM turns WHERE flagged_false_positive = 1 "
            "ORDER BY flagged_at DESC LIMIT ?",
            [n],
        )
        return [dict(zip(result.columns, row)) for row in result.rows]
    finally:
        client.close()


def find_cached_answer(question: str, project: str = None):
    """
    Exact-match lookup (case/whitespace-insensitive) for a question already
    asked and answered. Returns the most recent match, or None. Deliberately
    NOT fuzzy/semantic -- a near-miss match returning a stale answer for a
    subtly different question is worse than just re-running the pipeline.
    Excludes anything flagged as a false positive, so a known-bad cached
    answer never gets silently resurfaced.
    """
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        normalized = " ".join(question.strip().lower().split())
        if project:
            result = client.execute(
                "SELECT t.id, t.asked_at, t.question, t.answer, t.model "
                "FROM turns t JOIN sessions s ON s.id = t.session_id "
                "WHERE lower(trim(t.question)) = ? AND s.project = ? "
                "AND t.flagged_false_positive = 0 "
                "ORDER BY t.asked_at DESC LIMIT 1",
                [normalized, project],
            )
        else:
            result = client.execute(
                "SELECT id, asked_at, question, answer, model FROM turns "
                "WHERE lower(trim(question)) = ? AND flagged_false_positive = 0 "
                "ORDER BY asked_at DESC LIMIT 1",
                [normalized],
            )
        rows = [dict(zip(result.columns, row)) for row in result.rows]
        return rows[0] if rows else None
    finally:
        client.close()


def record_retrieval_gap(query: str, expected_source: str, notes: str = None):
    """
    Log that rag.search() missed a real, indexed source for this query.
    expected_source must already exist in citations (that's the point --
    this only makes sense once the real source has actually been verified).
    """
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "INSERT INTO retrieval_gaps (query, expected_source, notes) "
            "VALUES (?, ?, ?)",
            [query, expected_source, notes],
        )
    finally:
        client.close()


def record_query_quality(project: str, question: str, intent: str = None,
                         define: dict = None, measure: dict = None,
                         analyze: dict = None, improve: dict = None,
                         control: dict = None):
    """
    Log one Define/Measure/Analyze/Improve/Control event for the query path.

    This is intentionally separate from draft quality scoring. Query quality is
    about whether the system chose the right source-of-truth path and escalated
    when confidence was weak, not whether generated prose was stylish.
    """
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "INSERT INTO query_quality_events "
            "(project, question, intent, define_json, measure_json, "
            " analyze_json, improve_json, control_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                project,
                question,
                intent,
                json.dumps(define or {}),
                json.dumps(measure or {}),
                json.dumps(analyze or {}),
                json.dumps(improve or {}),
                json.dumps(control or {}),
            ],
        )
    finally:
        client.close()


# ---------------------------------------------------------------------------
# El Roi file identity and scan/diff foundation.
# ---------------------------------------------------------------------------

EL_ROI_NAMESPACE = uuid.UUID("3ab230c8-87f6-4d08-a0c2-9f6b55dff73a")


def _stable_id(kind: str, value: str) -> str:
    return str(uuid.uuid5(EL_ROI_NAMESPACE, f"{kind}:{value}"))


def _new_id() -> str:
    return str(uuid.uuid4())


def _server_name(name: str = None) -> str:
    return (name or socket.gethostname() or "local").strip()


def ensure_file_server(name: str = None, base_url: str = None,
                       machine: str = None) -> str:
    name = _server_name(name)
    server_id = _stable_id("file_server", name)
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "INSERT INTO file_servers "
            "(server_id, name, base_url, machine, last_seen_at) "
            "VALUES (?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(server_id) DO UPDATE SET "
            "name=excluded.name, base_url=coalesce(excluded.base_url, base_url), "
            "machine=coalesce(excluded.machine, machine), status='active', "
            "last_seen_at=datetime('now')",
            [server_id, name, base_url, machine],
        )
        return server_id
    finally:
        client.close()


def ensure_storage_root(root_path: str, server_name: str = None,
                        description: str = None, base_url: str = None,
                        machine: str = None) -> str:
    server_id = ensure_file_server(server_name, base_url=base_url, machine=machine)
    normalized = str(root_path)
    storage_root_id = _stable_id("storage_root", f"{server_id}:{normalized}")
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "INSERT INTO storage_roots "
            "(storage_root_id, server_id, root_path, description, observed_at) "
            "VALUES (?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(storage_root_id) DO UPDATE SET "
            "root_path=excluded.root_path, "
            "description=coalesce(excluded.description, description), "
            "status='active', observed_at=datetime('now')",
            [storage_root_id, server_id, normalized, description],
        )
        return storage_root_id
    finally:
        client.close()


def start_file_scan(storage_root_id: str) -> str:
    scan_run_id = _new_id()
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "INSERT INTO file_scan_runs (scan_run_id, storage_root_id) "
            "VALUES (?, ?)",
            [scan_run_id, storage_root_id],
        )
        return scan_run_id
    finally:
        client.close()


def _rowdict(result):
    rows = [dict(zip(result.columns, row)) for row in result.rows]
    return rows[0] if rows else None


def _hash_version(client, hash_value: str):
    result = client.execute(
        "SELECT fv.file_version_id, fv.file_id, fv.version_number "
        "FROM file_hashes fh "
        "JOIN file_versions fv ON fv.file_version_id = fh.file_version_id "
        "WHERE fh.hash_algorithm = 'md5' AND fh.hash_value = ? LIMIT 1",
        [hash_value],
    )
    return _rowdict(result)


def _latest_version_number(client, file_id: str) -> int:
    result = client.execute(
        "SELECT max(version_number) FROM file_versions WHERE file_id = ?",
        [file_id],
    )
    return int(result.rows[0][0] or 0) if result.rows else 0


def _create_file_version(client, file_id: str, version_number: int,
                         relative_path: str, hash_value: str, byte_size: int,
                         modified_at: str, mime_type: str = None) -> str:
    file_version_id = _new_id()
    client.execute(
        "INSERT OR IGNORE INTO files (file_id, status) VALUES (?, 'active')",
        [file_id],
    )
    client.execute(
        "INSERT INTO file_versions "
        "(file_version_id, file_id, version_number, origin_date, "
        " last_modified_at, byte_size, mime_type, original_filename) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            file_version_id, file_id, version_number, modified_at,
            modified_at, byte_size, mime_type, relative_path.split("/")[-1],
        ],
    )
    client.execute(
        "INSERT OR IGNORE INTO file_hashes "
        "(file_version_id, hash_algorithm, hash_value) VALUES (?, 'md5', ?)",
        [file_version_id, hash_value],
    )
    return file_version_id


def record_file_observation(storage_root_id: str, relative_path: str,
                            hash_value: str, byte_size: int,
                            modified_at: str, scan_run_id: str = None,
                            mime_type: str = None) -> dict:
    """
    Record one scan observation in El Roi and return stable file identity.

    A same-path hash change becomes a new version of the same file. A same-hash
    file discovered at another path reuses the existing file version, giving
    the catalog rename/copy awareness without relying on filenames.
    """
    mime_type = mime_type or mimetypes.guess_type(relative_path)[0] or ""
    storage_object_id = _stable_id(
        "storage_object", f"{storage_root_id}:{relative_path}")
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        existing = _rowdict(client.execute(
            "SELECT fso.file_version_id, fv.file_id, fh.hash_value "
            "FROM file_storage_objects fso "
            "JOIN file_versions fv ON fv.file_version_id = fso.file_version_id "
            "LEFT JOIN file_hashes fh ON fh.file_version_id = fv.file_version_id "
            " AND fh.hash_algorithm = 'md5' "
            "WHERE fso.storage_root_id = ? AND fso.relative_path = ? "
            "LIMIT 1",
            [storage_root_id, relative_path],
        ))

        if existing and existing.get("hash_value") == hash_value:
            file_id = existing["file_id"]
            file_version_id = existing["file_version_id"]
            observed_state = "unchanged"
        elif existing:
            hashed = _hash_version(client, hash_value)
            if hashed:
                file_id = hashed["file_id"]
                file_version_id = hashed["file_version_id"]
                observed_state = "matched_existing_hash"
            else:
                file_id = existing["file_id"]
                version_number = _latest_version_number(client, file_id) + 1
                file_version_id = _create_file_version(
                    client, file_id, version_number, relative_path, hash_value,
                    byte_size, modified_at, mime_type,
                )
                observed_state = "updated"
        else:
            hashed = _hash_version(client, hash_value)
            if hashed:
                file_id = hashed["file_id"]
                file_version_id = hashed["file_version_id"]
            else:
                file_id = _new_id()
                file_version_id = _create_file_version(
                    client, file_id, 1, relative_path, hash_value,
                    byte_size, modified_at, mime_type,
                )
            observed_state = "new"

        client.execute(
            "INSERT INTO file_storage_objects "
            "(storage_object_id, storage_root_id, file_version_id, "
            " relative_path, status, observed_at, missing_at) "
            "VALUES (?, ?, ?, ?, 'active', datetime('now'), NULL) "
            "ON CONFLICT(storage_root_id, relative_path) DO UPDATE SET "
            "file_version_id=excluded.file_version_id, status='active', "
            "observed_at=datetime('now'), missing_at=NULL",
            [storage_object_id, storage_root_id, file_version_id, relative_path],
        )
        client.execute(
            "INSERT OR IGNORE INTO file_paths (file_version_id, path_text) "
            "VALUES (?, ?)",
            [file_version_id, relative_path],
        )
        if scan_run_id:
            client.execute(
                "INSERT OR REPLACE INTO file_scan_observations "
                "(scan_run_id, storage_root_id, relative_path, file_version_id, "
                " hash_value, observed_state) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    scan_run_id, storage_root_id, relative_path,
                    file_version_id, hash_value, observed_state,
                ],
            )
        return {
            "file_id": file_id,
            "file_version_id": file_version_id,
            "storage_root_id": storage_root_id,
            "storage_object_id": storage_object_id,
            "observed_state": observed_state,
        }
    finally:
        client.close()


def mark_missing_storage_objects(storage_root_id: str, current_paths: list,
                                 scan_run_id: str = None) -> list:
    current = set(current_paths or [])
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        result = client.execute(
            "SELECT relative_path, file_version_id FROM file_storage_objects "
            "WHERE storage_root_id = ? AND status = 'active'",
            [storage_root_id],
        )
        missing = [
            {"relative_path": row[0], "file_version_id": row[1]}
            for row in result.rows
            if row[0] not in current
        ]
        for row in missing:
            client.execute(
                "UPDATE file_storage_objects SET status='missing', "
                "missing_at=datetime('now') "
                "WHERE storage_root_id = ? AND relative_path = ?",
                [storage_root_id, row["relative_path"]],
            )
            if scan_run_id:
                client.execute(
                    "INSERT OR REPLACE INTO file_scan_observations "
                    "(scan_run_id, storage_root_id, relative_path, "
                    " file_version_id, observed_state) VALUES (?, ?, ?, ?, 'missing')",
                    [
                        scan_run_id, storage_root_id, row["relative_path"],
                        row["file_version_id"],
                    ],
                )
        return missing
    finally:
        client.close()


def finish_file_scan(scan_run_id: str, summary: dict,
                     status: str = "complete", notes: str = None):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "UPDATE file_scan_runs SET finished_at=datetime('now'), "
            "files_seen=?, files_new=?, files_updated=?, files_unchanged=?, "
            "files_missing=?, status=?, notes=? WHERE scan_run_id=?",
            [
                sum(len(summary.get(k, [])) for k in ("new", "updated", "unchanged")),
                len(summary.get("new", [])),
                len(summary.get("updated", [])),
                len(summary.get("unchanged", [])),
                len(summary.get("removed", [])),
                status,
                notes,
                scan_run_id,
            ],
        )
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Settings -- persistent admin toggles, checked live by any process (see
# schema.sql for why this can't just be a .env var: separate processes
# reading a shared .env once at startup can't be flipped without a restart,
# a shared row here can).
# ---------------------------------------------------------------------------

def get_setting(key: str, default: str = None) -> str:
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        result = client.execute("SELECT value FROM settings WHERE key = ?", [key])
        return result.rows[0][0] if result.rows else default
    finally:
        client.close()


def set_setting(key: str, value: str):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_at=datetime('now')",
            [key, value],
        )
    finally:
        client.close()


def pii_redaction_enabled() -> bool:
    """Convenience wrapper -- the one thing every call site actually needs."""
    return get_setting("pii_redaction", "off") == "on"


# ---------------------------------------------------------------------------
# Search criteria -- retrieval policy maintained in libSQL.
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_CRITERIA = [
    *[
        {"criteria_type": "stopword", "term": term}
        for term in [
            "a", "an", "the", "and", "or", "but", "if", "of", "in", "on",
            "at", "to", "for", "with", "from", "by", "as", "is", "are",
            "was", "were", "be", "been", "being", "do", "does", "did",
            "done", "has", "have", "had", "having", "not", "no", "so",
            "than", "then", "this", "that", "these", "those", "it", "its",
            "it's", "you", "your", "yours", "he", "she", "they", "we", "i",
            "me", "my", "him", "her", "them", "us", "our", "their", "who",
            "what", "when", "where", "why", "how", "which", "can", "could",
            "should", "would", "will", "shall", "about", "into", "over",
            "under", "again", "also", "just", "up", "out", "off", "all",
            "any", "some", "such", "own", "article", "articles", "document",
            "documents", "file", "files", "source", "sources", "talk",
            "talks", "discuss", "discusses", "deal", "deals", "using",
            "use", "uses",
        ]
    ],
    *[
        {"criteria_type": "low_signal", "term": term}
        for term in [
            "ai", "genai", "generative", "artificial", "intelligence",
            "implication", "implications", "impact", "impacts", "effect",
            "effects",
        ]
    ],
    *[
        {"criteria_type": "source_low_signal", "term": term}
        for term in [
            "article", "study", "research", "system", "systems", "service",
            "services", "practice", "quality", "offering",
        ]
    ],
    *[
        {"criteria_type": "section_noise", "term": term}
        for term in [
            "references", "bibliography", "works cited", "doi:",
            "accessed:", "[online]", "available:",
        ]
    ],
    *[
        {"criteria_type": "document_section", "term": term}
        for term in [
            "abstract", "introduction", "literature review", "method",
            "methods", "methodology", "results", "findings", "discussion",
            "conclusion", "recommendations", "implications", "limitations",
            "references",
        ]
    ],
    *[
        {"criteria_type": "genre_marker", "group_name": group, "term": term}
        for group, terms in {
            "chat_export": [
                "chat history", "conversation export", "conversation with claude",
                "working session", "record of a research and writing session",
            ],
            "research_methods_guide": ["sage research methods"],
            "book_chapter": ["doi:", "online isbn"],
            "dissertation_template": [
                "dissertation template", "insert your dissertation title here",
            ],
            "dissertation": [
                "a dissertation presented to", "a dissertation submitted",
                "doctoral dissertation", "doctor of philosophy", "degree of doctor",
                "dissertation committee", "proquest dissertations",
            ],
            "research_book_filename": [
                "an-applied-guide-to-research-designs",
                "an-introduction-to-qualitative-research",
                "constructing-social-research", "doing-quantitative-research",
                "introducing-qualitative-research", "qualitative-data-analysis",
                "qualitative-data-collection-tools",
                "quantitative-research-in-education",
                "research-methods-and-statistics", "research-with-children",
                "social-research-theory-methods",
                "understanding-and-evaluating-research",
                "qualitativeresearchag", "sharanb.merriam",
            ],
            "research_book_head": [
                "library of congress cataloging",
                "all rights reserved. may not be reproduced",
                "sage publications", "sage research methods", "isbn",
            ],
            "coursework": [
                "topic4 dq", "topic5 dq", "topic6 dq", "topic7 dq",
                " dq1", " dq2", "summary of the problem space",
                "population to be studied", "variables (excluding demographics)",
                "discussion question",
            ],
            "coursework_dissertation": ["problem space", "dissertation"],
            "presentation": ["slide 1:", "speaker notes"],
            "spreadsheet": ["sheet:"],
            "academic_section": [
                "abstract", "introduction", "method", "methods", "methodology",
                "results", "findings", "discussion", "conclusion", "references",
            ],
            "academic_filename": [
                "ebsco-fulltext", "s2.0-", "feduc-", "societies-", "jmir_",
                "determinants_of", "div-class-title", "sc-96", "ej",
            ],
            "scholarly_marker": [
                "doi:", "journal", " vol.", " volume ", " issue ", "abstract",
                "keywords", "received", "accepted", "publication year",
                "publisher information", "type original research",
                "original research", "article",
            ],
            "legal_filing": [
                "plaintiff", "defendant", "case no", "court", "pursuant to",
                "complaint", "affidavit", "judgment", "dismissal",
                "certificate of service",
            ],
            "contract_agreement": [
                "settlement agreement", "quitclaim", "contract", "agreement made",
                "executed agreement", "this agreement", "release and settlement",
            ],
            "interview_protocol": ["interview protocol", "interview questions"],
            "literature_review": [
                "systematic review", "scoping review", "review of the literature",
            ],
            "notes": ["meeting notes"],
        }.items()
        for term in terms
    ],
    *[
        {"criteria_type": "theme_marker", "group_name": group, "term": term}
        for group, terms in {
            "ai_adoption": [
                "ai adoption", "adoption of ai",
                "artificial intelligence adoption", "generative ai adoption",
                "adopt generative ai", "ai usage",
            ],
            "training_usability": [
                "training", "ease of use", "perceived usefulness",
            ],
            "technology_context": [
                "ai", "artificial intelligence", "technology", "system",
            ],
            "legal_privilege": [
                "attorney-client privilege", "attorney client privilege",
                "work-product privilege", "work product doctrine",
                "work product privilege", "client confidentiality",
                "legal privilege", "privileged communication",
            ],
            "research_methods": [
                "methodology", "qualitative", "quantitative", "research design",
                "interview protocol", "data collection", "sample size",
            ],
            "risk_governance": [
                "risk governance", "ai governance", "compliance", "legal ethics",
                "confidentiality", "privacy risk", "ethical risk",
                "risk management",
            ],
        }.items()
        for term in terms
    ],
    *[
        {"criteria_type": "subject_stop_label", "term": term}
        for term in [
            "description", "abstract", "source", "publisher information",
        ]
    ],
    *[
        {"criteria_type": "genre_alias", "group_name": group, "term": term}
        for group, terms in {
            "academic article": [
                "article", "articles", "academic article", "academic articles",
            ],
            "book chapter": ["book chapter", "book chapters"],
            "presentation": ["presentation", "presentations"],
            "chat export": ["chat export", "chat exports"],
            "coursework": ["coursework"],
            "dissertation draft": ["dissertation draft", "dissertation drafts"],
            "dissertation": ["dissertation", "dissertations", "thesis"],
            "legal filing": ["legal filing", "legal filings"],
            "contract/agreement": [
                "contract", "contracts", "agreement", "agreements",
            ],
            "interview protocol": ["interview protocol", "interview protocols"],
            "research methods guide": [
                "research methods guide", "research methods guides",
            ],
        }.items()
        for term in terms
    ],
    *[
        {"criteria_type": "ask_route", "group_name": group, "term": term}
        for group, terms in {
            "summarize": ["summarize", "summary"],
            "abstract_filter": [
                "abstract", "abstracts", "with abstract", "has abstract",
                "have abstract", "having abstract", "contains abstract",
                "include abstract",
            ],
            "topic_filter": [
                "about", "on", "deals with", "dealing with", "related to",
                "concerning", "covers", "covering", "discusses", "discussing",
            ],
            "annotated_bibliography": [
                "bibliography", "annotated bibliography",
            ],
            "document_metadata": [
                "author", "authors", "who wrote", "who authored",
                "written by", "document type", "document types", "genre",
                "genres", "subject", "subject matter", "topic", "topics",
                "theme", "themes", "metadata",
            ],
            "content_search": ["which", "what", "find", "show", "identify"],
            "app_command": [
                "clear", "reset", "wipe", "reindex", "re-index", "rescan",
                "refresh index",
            ],
            "content_question": [
                "argues", "covers", "discusses", "says", "explain",
                "summarize", "summary", "compare", "contrast", "synthesize",
                "analyze",
            ],
            "redaction_request": [
                "redact", "redacted", "redaction", "de-identify",
                "deidentify", "remove pii", "remove personal information",
            ],
            "coder_request": [
                "code:", "coder:", "write code:", "edit code:",
                "implement:", "patch:",
            ],
            "contextual_search": [
                "this", "that", "these", "those", "same", "subject",
                "matter", "above", "it",
            ],
            "all_documents": ["all", "every", "each"],
            "web_search": [
                "web search", "internet search", "online search",
                "google search", "external search", "search the web",
                "search the internet", "search online", "search google",
            ],
        }.items()
        for term in terms
    ],
    *[
        {"criteria_type": "relation_target", "term": term}
        for term in [
            "concerning", "regarding", "about", "related to", "dealing with",
            "involving", "mentioning", "referencing",
        ]
    ],
    *[
        {"criteria_type": "source_lookup_stopword", "term": term}
        for term in [
            "about", "above", "additional", "additionally", "also", "another",
            "appears", "article", "articles", "because", "being", "could",
            "discuss", "discusses", "discussing", "document", "documents",
            "file", "files", "following", "found", "from", "into", "legal",
            "like", "mentions", "one", "other", "provides", "settings",
            "several", "source", "sources", "specific", "specifically",
            "that", "their", "there", "these", "this", "those", "use",
            "uses", "using", "which", "with", "dealing", "subject",
            "concerning",
        ]
    ],
    *[
        {"criteria_type": "redaction_profile", "term": term}
        for term in [
            "legal_privileged",
            "education_ferpa",
            "medical_hipaa",
            "military_confidential",
        ]
    ],
    *[
        {"criteria_type": "redaction_rule", "group_name": group, "term": term}
        for group, terms in {
            "legal_privileged": [
                "person", "email", "phone", "ssn", "credit_card",
                "street_address", "highway_address", "city_state_zip",
                "zip_code",
            ],
            "education_ferpa": [
                "person", "email", "phone", "ssn", "street_address",
                "zip_code", "student_id", "date",
            ],
            "medical_hipaa": [
                "person", "email", "phone", "ssn", "credit_card",
                "street_address", "zip_code", "date", "medical_record_number",
            ],
            "military_confidential": [
                "person", "email", "phone", "ssn", "street_address",
                "zip_code", "rank_serial", "unit_identifier",
            ],
        }.items()
        for term in terms
    ],
    *[
        {"criteria_type": "redaction_protection", "group_name": group, "term": term}
        for group, terms in {
            "legal_privileged": [
                "legal_authority", "short_statute", "court_name", "case_number",
            ],
            "education_ferpa": [
                "institution_name", "course_code",
            ],
            "medical_hipaa": [
                "medical_authority", "facility_type",
            ],
            "military_confidential": [
                "public_military_authority",
            ],
        }.items()
        for term in terms
    ],
    *[
        {"criteria_type": "domain_trigger", "group_name": "healthcare", "term": term}
        for term in [
            "healthcare", "health", "medical", "clinical", "patient",
            "patients", "hospital", "hospitals", "medicine",
        ]
    ],
    *[
        {"criteria_type": "domain_term", "group_name": "healthcare", "term": term}
        for term in [
            "healthcare", "health care", "medical", "clinical", "patient",
            "patients", "hospital", "hospitals", "medicine", "physician",
            "physicians", "nurse", "nurses", "care delivery",
        ]
    ],
    *[
        {"criteria_type": "domain_trigger", "group_name": "legal", "term": term}
        for term in [
            "legal", "law", "lawyer", "lawyers", "attorney", "attorneys",
            "privilege", "confidentiality",
        ]
    ],
    *[
        {"criteria_type": "domain_term", "group_name": "legal", "term": term}
        for term in [
            "legal", "law", "lawyer", "lawyers", "attorney", "attorneys",
            "privilege", "confidentiality", "jurimetrics", "court",
        ]
    ],
    *[
        {"criteria_type": "domain_trigger", "group_name": "education", "term": term}
        for term in [
            "education", "educational", "student", "students", "teacher",
            "teachers", "school", "schools", "university",
        ]
    ],
    *[
        {"criteria_type": "domain_term", "group_name": "education", "term": term}
        for term in [
            "education", "educational", "student", "students", "teacher",
            "teachers", "school", "schools", "university", "learning",
            "academic",
        ]
    ],
]


def upsert_search_criterion(criteria_type: str, term: str, group_name: str = "",
                            weight: float = 1.0, enabled: bool = True,
                            notes: str = None):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "INSERT INTO search_criteria "
            "(criteria_type, group_name, term, weight, enabled, notes) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(criteria_type, group_name, term) DO UPDATE SET "
            "weight=excluded.weight, enabled=excluded.enabled, "
            "notes=excluded.notes, updated_at=datetime('now')",
            [criteria_type, group_name or "", term.lower(), weight,
             1 if enabled else 0, notes],
        )
    finally:
        client.close()


def seed_default_search_criteria() -> int:
    inserted = 0
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        for row in DEFAULT_SEARCH_CRITERIA:
            result = client.execute(
                "INSERT OR IGNORE INTO search_criteria "
                "(criteria_type, group_name, term, weight, enabled, notes) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                [
                    row["criteria_type"],
                    row.get("group_name", ""),
                    row["term"].lower(),
                    row.get("weight", 1.0),
                    "seeded_default",
                ],
            )
            if getattr(result, "rows_affected", 0):
                inserted += result.rows_affected
        return inserted
    finally:
        client.close()


def get_search_criteria(criteria_type: str = None, enabled_only: bool = True):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        sql = (
            "SELECT id, criteria_type, group_name, term, weight, enabled, notes, "
            "updated_at FROM search_criteria"
        )
        clauses = []
        params = []
        if criteria_type:
            clauses.append("criteria_type = ?")
            params.append(criteria_type)
        if enabled_only:
            clauses.append("enabled = 1")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY criteria_type, group_name, term"
        result = client.execute(sql, params)
        return [dict(zip(result.columns, row)) for row in result.rows]
    finally:
        client.close()


def record_retrieval_miss(project: str, query: str, expected_source: str = None,
                          actual_source: str = None, notes: str = None,
                          status: str = "open"):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        result = client.execute(
            "INSERT INTO retrieval_misses "
            "(project, query, expected_source, actual_source, notes, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [project, query, expected_source, actual_source, notes, status],
        )
        return result.last_insert_rowid
    finally:
        client.close()


def list_retrieval_misses(project: str = None, status: str = None,
                          limit: int = 50):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        clauses = []
        params = []
        if project:
            clauses.append("project = ?")
            params.append(project)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        result = client.execute(
            "SELECT id, recorded_at, project, query, expected_source, "
            "actual_source, notes, status FROM retrieval_misses"
            + where + " ORDER BY recorded_at DESC LIMIT ?",
            params + [limit],
        )
        return [dict(zip(result.columns, row)) for row in result.rows]
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Documents -- ingest-time synopses. See schema.sql for the "context aware
# storage" reasoning: chroma_db stays chunk-level, this is document-level,
# consulted alongside chroma's own search rather than instead of it.
# ---------------------------------------------------------------------------

def record_synopsis(source: str, synopsis: str, word_count: int = None,
                    model: str = None):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "INSERT INTO documents (source, synopsis, word_count, model) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(source) DO UPDATE SET "
            " synopsis=excluded.synopsis, word_count=excluded.word_count, "
            " model=excluded.model, indexed_at=datetime('now')",
            [source, synopsis, word_count, model],
        )
    finally:
        client.close()



def record_document_upload_profile(source: str, synopsis: str, word_count: int = None,
                                   model: str = None, source_hash: str = None,
                                   chars: int = None, file_type: str = None,
                                   project: str = None, label: str = None,
                                   sections_found: dict = None,
                                   genres: list = None, themes: list = None,
                                   authors: list = None,
                                   subject_terms: list = None,
                                   upload_state: str = "project_file",
                                   file_id: str = None,
                                   file_version_id: str = None,
                                   storage_root_id: str = None):
    """
    Upsert upload-style metadata for a project document.

    Any file saved under the documents root is treated like an uploaded source:
    it gets durable document-level metadata in libSQL, while Chroma keeps the
    chunk embeddings. JSON fields stay as text for libSQL portability.
    """
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        try:
            client.execute(
                "INSERT INTO documents "
                "(source, synopsis, word_count, model, source_hash, chars, file_type, "
                " project, label, sections_found, genres, themes, authors, "
                " subject_terms, upload_state, file_id, file_version_id, "
                " storage_root_id, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                " datetime('now')) "
                "ON CONFLICT(source) DO UPDATE SET "
                " synopsis=excluded.synopsis, word_count=excluded.word_count, "
                " model=excluded.model, source_hash=excluded.source_hash, "
                " chars=excluded.chars, file_type=excluded.file_type, "
                " project=excluded.project, label=excluded.label, "
                " sections_found=excluded.sections_found, genres=excluded.genres, "
                " themes=excluded.themes, authors=excluded.authors, "
                " subject_terms=excluded.subject_terms, "
                " upload_state=excluded.upload_state, "
                " file_id=excluded.file_id, "
                " file_version_id=excluded.file_version_id, "
                " storage_root_id=excluded.storage_root_id, "
                " indexed_at=datetime('now'), updated_at=datetime('now')",
                [source, synopsis, word_count, model, source_hash, chars, file_type,
                 project, label, json.dumps(sections_found or {}),
                 json.dumps(genres or []), json.dumps(themes or []),
                 json.dumps(authors or []), json.dumps(subject_terms or []),
                 upload_state, file_id, file_version_id, storage_root_id],
            )
        except Exception as e:
            if "no such column" not in str(e).lower():
                raise
            client.execute(
                "INSERT INTO documents "
                "(source, synopsis, word_count, model, source_hash, chars, file_type, "
                " project, label, sections_found, genres, themes, upload_state, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now')) "
                "ON CONFLICT(source) DO UPDATE SET "
                " synopsis=excluded.synopsis, word_count=excluded.word_count, "
                " model=excluded.model, source_hash=excluded.source_hash, "
                " chars=excluded.chars, file_type=excluded.file_type, "
                " project=excluded.project, label=excluded.label, "
                " sections_found=excluded.sections_found, genres=excluded.genres, "
                " themes=excluded.themes, upload_state=excluded.upload_state, "
                " indexed_at=datetime('now'), updated_at=datetime('now')",
                [source, synopsis, word_count, model, source_hash, chars, file_type,
                 project, label, json.dumps(sections_found or {}),
                 json.dumps(genres or []), json.dumps(themes or []), upload_state],
            )
    finally:
        client.close()


def link_document_file_identity(source: str, file_id: str = None,
                                file_version_id: str = None,
                                storage_root_id: str = None):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "UPDATE documents SET file_id=?, file_version_id=?, "
            "storage_root_id=?, updated_at=datetime('now') WHERE source=?",
            [file_id, file_version_id, storage_root_id, source],
        )
    finally:
        client.close()


def search_document_uploads(query: str = None, project: str = None,
                            genre: str = None, theme: str = None,
                            exclude_genres: list = None,
                            limit: int = 50):
    """Query upload-style document metadata from libSQL."""
    if project == "__all__":
        project = None
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        clauses = []
        params = []
        has_new_metadata_columns = True
        if query:
            like = f"%{query.lower()}%"
            fields = [
                "source", "label", "synopsis", "genres", "themes",
                "authors", "subject_terms",
            ]
            clauses.append("(" + " OR ".join(
                f"lower(coalesce({field}, '')) LIKE ?" for field in fields
            ) + ")")
            params.extend([like] * len(fields))
        if project:
            clauses.append("project = ?")
            params.append(project)
        if genre:
            clauses.append("lower(coalesce(genres, '')) LIKE ?")
            params.append(f"%{genre.lower()}%")
        if theme:
            clauses.append("lower(coalesce(themes, '')) LIKE ?")
            params.append(f"%{theme.lower()}%")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = (
            "SELECT source, project, label, file_type, chars, word_count, "
            "source_hash, sections_found, genres, themes, synopsis, "
            "authors, subject_terms, upload_state, file_id, file_version_id, "
            "storage_root_id, indexed_at, updated_at "
            "FROM documents"
            + where + " ORDER BY coalesce(updated_at, indexed_at) DESC LIMIT ?"
        )
        params.append(limit)
        try:
            result = client.execute(sql, params)
        except Exception as e:
            if "no such column" not in str(e).lower():
                raise
            has_new_metadata_columns = False
            clauses = []
            params = []
            if query:
                like = f"%{query.lower()}%"
                fields = ["source", "label", "synopsis", "genres", "themes"]
                clauses.append("(" + " OR ".join(
                    f"lower(coalesce({field}, '')) LIKE ?" for field in fields
                ) + ")")
                params.extend([like] * len(fields))
            if project:
                clauses.append("project = ?")
                params.append(project)
            if genre:
                clauses.append("lower(coalesce(genres, '')) LIKE ?")
                params.append(f"%{genre.lower()}%")
            if theme:
                clauses.append("lower(coalesce(themes, '')) LIKE ?")
                params.append(f"%{theme.lower()}%")
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            sql = (
                "SELECT source, project, label, file_type, chars, word_count, "
                "source_hash, sections_found, genres, themes, synopsis, "
                "upload_state, indexed_at, updated_at FROM documents"
                + where + " ORDER BY coalesce(updated_at, indexed_at) DESC LIMIT ?"
            )
            params.append(limit)
            result = client.execute(sql, params)
        rows = [dict(zip(result.columns, row)) for row in result.rows]
        for row in rows:
            json_fields = [
                ("sections_found", {}),
                ("genres", []),
                ("themes", []),
            ]
            if has_new_metadata_columns:
                json_fields.extend([("authors", []), ("subject_terms", [])])
            for field, default in json_fields:
                try:
                    row[field] = json.loads(row[field]) if row.get(field) else default
                except Exception:
                    row[field] = default
        if genre:
            rows = [r for r in rows if genre.lower() in [g.lower() for g in r.get("genres", [])]]
        if theme:
            rows = [r for r in rows if theme.lower() in [t.lower() for t in r.get("themes", [])]]
        excluded = {g.lower() for g in (exclude_genres or [])}
        if excluded:
            rows = [
                r for r in rows
                if not excluded.intersection({g.lower() for g in r.get("genres", [])})
            ]
        return rows
    finally:
        client.close()

def get_synopsis(source: str):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        result = client.execute("SELECT * FROM documents WHERE source = ?", [source])
        rows = [dict(zip(result.columns, row)) for row in result.rows]
        return rows[0] if rows else None
    finally:
        client.close()


def search_synopses(query_words: list, project_prefix: str = None):
    """
    Document-level correlation check: which indexed files' synopses mention
    any of these words. This is the "correlate against libSQL for more
    context" piece -- meant to be checked ALONGSIDE chroma_db's chunk-level
    search() results at query time, not as a replacement for it. Simple
    substring matching, not semantic -- synopses are short enough that this
    is a reasonable, cheap, fully-local first pass.
    """
    if not query_words:
        return []
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        clauses = " OR ".join(["lower(synopsis) LIKE ?"] * len(query_words))
        params = [f"%{w.lower()}%" for w in query_words]
        sql = f"SELECT source, synopsis FROM documents WHERE ({clauses})"
        if project_prefix:
            sql += " AND source LIKE ?"
            params.append(f"{project_prefix}%")
        result = client.execute(sql, params)
        return [dict(zip(result.columns, row)) for row in result.rows]
    finally:
        client.close()


def get_document_summary(source: str, source_hash: str, model: str,
                         max_chars: int):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        result = client.execute(
            "SELECT source, source_hash, model, max_chars, chars, truncated, "
            "summary, generated_at, last_used_at "
            "FROM document_summaries "
            "WHERE source = ? AND source_hash = ? AND model = ? AND max_chars = ? "
            "LIMIT 1",
            [source, source_hash, model, max_chars],
        )
        rows = [dict(zip(result.columns, row)) for row in result.rows]
        if not rows:
            return None
        client.execute(
            "UPDATE document_summaries SET last_used_at = datetime('now') "
            "WHERE source = ? AND source_hash = ? AND model = ? AND max_chars = ?",
            [source, source_hash, model, max_chars],
        )
        return rows[0]
    finally:
        client.close()


def record_document_summary(source: str, source_hash: str, model: str,
                            max_chars: int, chars: int, truncated: bool,
                            summary: str):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "INSERT INTO document_summaries "
            "(source, source_hash, model, max_chars, chars, truncated, summary) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source, source_hash, model, max_chars) DO UPDATE SET "
            " chars=excluded.chars, truncated=excluded.truncated, "
            " summary=excluded.summary, generated_at=datetime('now'), "
            " last_used_at=datetime('now')",
            [source, source_hash, model, max_chars, chars,
             1 if truncated else 0, summary],
        )
    finally:
        client.close()


def get_bibliography_entry(source: str, source_hash: str, model: str,
                           style: str, max_chars: int):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        result = client.execute(
            "SELECT source, source_hash, model, style, max_chars, chars, "
            "truncated, reference, annotation, generated_at, last_used_at "
            "FROM bibliography_entries "
            "WHERE source = ? AND source_hash = ? AND model = ? "
            "AND style = ? AND max_chars = ? LIMIT 1",
            [source, source_hash, model, style, max_chars],
        )
        rows = [dict(zip(result.columns, row)) for row in result.rows]
        if not rows:
            return None
        client.execute(
            "UPDATE bibliography_entries SET last_used_at = datetime('now') "
            "WHERE source = ? AND source_hash = ? AND model = ? "
            "AND style = ? AND max_chars = ?",
            [source, source_hash, model, style, max_chars],
        )
        return rows[0]
    finally:
        client.close()


def record_bibliography_entry(source: str, source_hash: str, model: str,
                              style: str, max_chars: int, chars: int,
                              truncated: bool, reference: str,
                              annotation: str):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "INSERT INTO bibliography_entries "
            "(source, source_hash, model, style, max_chars, chars, truncated, "
            " reference, annotation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source, source_hash, model, style, max_chars) "
            "DO UPDATE SET chars=excluded.chars, truncated=excluded.truncated, "
            " reference=excluded.reference, annotation=excluded.annotation, "
            " generated_at=datetime('now'), last_used_at=datetime('now')",
            [source, source_hash, model, style, max_chars, chars,
             1 if truncated else 0, reference, annotation],
        )
    finally:
        client.close()


# ---------------------------------------------------------------------------
# PII scans -- summary only (entity types + count), never the matched text
# itself. See schema.sql and pii.py for why.
# ---------------------------------------------------------------------------

def record_pii_scan(source: str, entity_types: list, finding_count: int):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        client.execute(
            "INSERT INTO pii_scans (source, entity_types, finding_count) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(source) DO UPDATE SET "
            " entity_types=excluded.entity_types, "
            " finding_count=excluded.finding_count, "
            " scanned_at=datetime('now')",
            [source, json.dumps(entity_types), finding_count],
        )
    finally:
        client.close()


def get_pii_scan(source: str):
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        result = client.execute("SELECT * FROM pii_scans WHERE source = ?", [source])
        rows = [dict(zip(result.columns, row)) for row in result.rows]
        return rows[0] if rows else None
    finally:
        client.close()
