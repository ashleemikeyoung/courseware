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
                                   upload_state: str = "project_file"):
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
                " subject_terms, upload_state, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now')) "
                "ON CONFLICT(source) DO UPDATE SET "
                " synopsis=excluded.synopsis, word_count=excluded.word_count, "
                " model=excluded.model, source_hash=excluded.source_hash, "
                " chars=excluded.chars, file_type=excluded.file_type, "
                " project=excluded.project, label=excluded.label, "
                " sections_found=excluded.sections_found, genres=excluded.genres, "
                " themes=excluded.themes, authors=excluded.authors, "
                " subject_terms=excluded.subject_terms, "
                " upload_state=excluded.upload_state, "
                " indexed_at=datetime('now'), updated_at=datetime('now')",
                [source, synopsis, word_count, model, source_hash, chars, file_type,
                 project, label, json.dumps(sections_found or {}),
                 json.dumps(genres or []), json.dumps(themes or []),
                 json.dumps(authors or []), json.dumps(subject_terms or []),
                 upload_state],
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


def search_document_uploads(query: str = None, project: str = None,
                            genre: str = None, theme: str = None,
                            exclude_genres: list = None,
                            limit: int = 50):
    """Query upload-style document metadata from libSQL."""
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
            "authors, subject_terms, upload_state, indexed_at, updated_at "
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
