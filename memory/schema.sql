-- ~/Development/RAG/memory/schema.sql
--
-- Session/episodic memory for the RAG stack. This is deliberately separate
-- from chroma_db: Chroma answers "what's relevant" (chunks + embeddings),
-- this answers "what happened" (questions asked, answers given, which
-- chunks actually got used, how they scored). Chunk references here are
-- loose foreign keys into Chroma's own "{rel}::{i}" ids -- chunk text is
-- never duplicated into this database, only the ids.
--
-- Applied automatically by init_schema.py once the server is up.

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at    TEXT,
    project     TEXT,                  -- matches rag.py's project folder name
    machine     TEXT NOT NULL,         -- 'mac' | 'alice'
    mode        TEXT NOT NULL,         -- 'qa' | 'draft' | 'bench'
    incognito   INTEGER NOT NULL DEFAULT 0   -- 1 = caller asked not to log; see note below
);

-- A single question/answer exchange within a session. writer.py's
-- plan-then-draft loop logs one row per section here too, with mode
-- carried from the parent session.
--
-- "Synthesis of multiple texts" (per today's design conversation) doesn't
-- get its own column -- it's already answerable from chunk_ids: any turn
-- whose chunk_ids JSON array has more than one entry drew from multiple
-- sources. Query it directly, e.g.:
--   SELECT * FROM turns WHERE json_array_length(chunk_ids) > 1;
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    asked_at    TEXT NOT NULL DEFAULT (datetime('now')),
    question    TEXT NOT NULL,
    answer      TEXT,
    model       TEXT,                  -- e.g. 'qwen3:32b'
    chunk_ids   TEXT                   -- JSON array of chroma "{rel}::{i}" ids retrieved
);

-- False-positive flagging, added on top of the original turns table via
-- ALTER TABLE rather than by editing the CREATE TABLE above -- "CREATE
-- TABLE IF NOT EXISTS" is a full no-op once the table already exists, so
-- putting new columns in that column list would silently never apply them
-- to a database that's already been initialized once (which this one has).
-- init_schema.py tolerates re-running these against a database where
-- they're already present.
--
-- Distinct from retrieval_gaps below -- a flagged turn is "the answer was
-- wrong," a retrieval gap is specifically "the right source existed and
-- search() didn't find it." A wrong answer can happen for other reasons
-- (bad synthesis, stale context, etc.), so this is broader.
ALTER TABLE turns ADD COLUMN flagged_false_positive INTEGER NOT NULL DEFAULT 0;
ALTER TABLE turns ADD COLUMN flag_notes TEXT;
ALTER TABLE turns ADD COLUMN flagged_at TEXT;

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_asked_at ON turns(asked_at);
CREATE INDEX IF NOT EXISTS idx_turns_question ON turns(question);

-- One row per turn, populated from quality.py's scoring pass.
CREATE TABLE IF NOT EXISTS quality_scores (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id             INTEGER NOT NULL REFERENCES turns(id),
    fabrication         REAL,
    grounding           REAL,
    self_repetition     REAL,
    cross_section_bleed REAL,
    length_adherence    REAL,
    scored_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_quality_turn ON quality_scores(turn_id);

-- NOTE on incognito: the `incognito` column above exists so a session can be
-- *marked*, not so it can be logged anyway. The client library (memory_client.py)
-- is expected to skip the write entirely when incognito=True is passed in --
-- rows should never land here from an incognito session in the first place.
-- The column is a safety net for auditing (proving nothing incognito got
-- captured), not a flag to filter on at read time.


-- ---------------------------------------------------------------------------
-- Verified citations -- durable ground truth, independent of chroma_db's
-- ranking. One row per source file, established by hand (or by an agent
-- doing real extraction) whenever a document's actual title/author gets
-- confirmed the hard way, e.g. by reading the PDF directly because
-- rag.search() failed to surface it for an author-name query. This exists
-- to be looked up directly -- "who wrote X" doesn't have to survive
-- Chroma's keyword scoring if it's already sitting here.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS citations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source            TEXT NOT NULL UNIQUE,  -- rag.py's relative path, e.g. "GCU/EBSCO-FullText-07_26_2026.pdf"
    title             TEXT,
    authors           TEXT,
    source_line       TEXT,   -- journal/publisher citation line, e.g. "Jurimetrics ... Vol. 64 Issue 3, p309"
    publication_year  INTEGER,
    verified_at       TEXT NOT NULL DEFAULT (datetime('now')),
    verified_how      TEXT    -- e.g. 'direct_pdf_read', 'ebsco_header', 'manual'
);

CREATE INDEX IF NOT EXISTS idx_citations_authors ON citations(authors);

-- One row per time a real, indexed source existed but rag.search() didn't
-- surface it for a given query. This is the evidence trail for eventually
-- fixing search()'s term-rarity weighting -- without logging *when* and
-- *for what query* it fails, that fix has nothing concrete to check itself
-- against.
CREATE TABLE IF NOT EXISTS retrieval_gaps (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    query            TEXT NOT NULL,
    expected_source  TEXT NOT NULL REFERENCES citations(source),
    noticed_at       TEXT NOT NULL DEFAULT (datetime('now')),
    notes            TEXT
);

CREATE INDEX IF NOT EXISTS idx_gaps_source ON retrieval_gaps(expected_source);


-- ---------------------------------------------------------------------------
-- Themes -- scaffolding only for now. Schema exists so document-theme
-- matches can be recorded manually starting today; the automatic part
-- (checking newly-scanned documents against known themes on ingest, likely
-- via an LLM classification call hooked into rag.scan_documents()'s
-- summary['new'] list) is a separate, not-yet-built design -- see
-- conversation from 2026-08-03. Where themes actually come from (manually
-- named vs. auto-extracted from writer.py outlines/theses) is still an
-- open question; this table doesn't presuppose an answer.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS themes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT,
    identified_at   TEXT NOT NULL DEFAULT (datetime('now')),
    source_turn_id  INTEGER REFERENCES turns(id)  -- optional: the turn/document that first surfaced this theme
);

CREATE TABLE IF NOT EXISTS document_themes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,   -- rag.py relative path
    theme_id    INTEGER NOT NULL REFERENCES themes(id),
    matched_at  TEXT NOT NULL DEFAULT (datetime('now')),
    notes       TEXT
);

CREATE INDEX IF NOT EXISTS idx_doc_themes_source ON document_themes(source);
CREATE INDEX IF NOT EXISTS idx_doc_themes_theme ON document_themes(theme_id);


-- ---------------------------------------------------------------------------
-- Settings -- small persistent key/value store for admin-level toggles that
-- need to apply consistently across separate processes (orchestrator.py,
-- writer.py, and anything else that reads it) without a restart. First use:
-- pii_redaction on/off. A .env var couldn't do this -- each process reads
-- its own .env once at startup, so flipping an env var wouldn't affect a
-- process that's already running. A shared row in libSQL, checked live at
-- the moment redaction matters, does.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);


-- ---------------------------------------------------------------------------
-- Documents -- one row per indexed source, holding an LLM-generated
-- synopsis. This is the "context aware storage" piece: as files get
-- ingested, a synopsis lands here so it can be checked alongside chroma_db's
-- chunk-level search at query time -- document-level relevance ("is this
-- WHOLE file about the right thing") as a complement to chunk-level
-- relevance ("is this SPECIFIC passage about the right thing"). RAG +
-- Ollama remain the primary engine; this is consulted for extra context or
-- correlation, not a replacement retrieval path.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS documents (
    source        TEXT PRIMARY KEY,   -- rag.py's relative path, e.g. "GCU/foo.pdf"
    synopsis      TEXT,
    word_count    INTEGER,
    model         TEXT,               -- which Ollama model generated the synopsis
    indexed_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Generated summaries -- reusable working memory for expensive whole-file
-- summaries. Unlike `documents.synopsis`, these are user-facing summaries
-- produced by summarize.py. They are keyed by source hash so editing a file
-- automatically misses the cache and records a fresh summary.

-- Upload-style document metadata. Any file saved under documents/<project>/ is
-- treated as an uploaded document for indexing and discovery: durable source,
-- extracted-text stats, label, sections, lightweight genres/themes, and state.
-- Kept as ALTER TABLE statements so existing memory databases upgrade in place.
ALTER TABLE documents ADD COLUMN source_hash TEXT;
ALTER TABLE documents ADD COLUMN chars INTEGER;
ALTER TABLE documents ADD COLUMN file_type TEXT;
ALTER TABLE documents ADD COLUMN project TEXT;
ALTER TABLE documents ADD COLUMN label TEXT;
ALTER TABLE documents ADD COLUMN sections_found TEXT; -- JSON object: section -> line number
ALTER TABLE documents ADD COLUMN genres TEXT;          -- JSON array
ALTER TABLE documents ADD COLUMN themes TEXT;          -- JSON array
ALTER TABLE documents ADD COLUMN authors TEXT;         -- JSON array
ALTER TABLE documents ADD COLUMN subject_terms TEXT;   -- JSON array
ALTER TABLE documents ADD COLUMN upload_state TEXT NOT NULL DEFAULT 'project_file';
ALTER TABLE documents ADD COLUMN updated_at TEXT;

CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project);
CREATE INDEX IF NOT EXISTS idx_documents_label ON documents(label);
CREATE INDEX IF NOT EXISTS idx_documents_updated ON documents(updated_at);

CREATE TABLE IF NOT EXISTS document_summaries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source         TEXT NOT NULL,
    source_hash    TEXT NOT NULL,
    model          TEXT NOT NULL,
    max_chars      INTEGER NOT NULL,
    chars          INTEGER,
    truncated      INTEGER NOT NULL DEFAULT 0,
    summary        TEXT NOT NULL,
    generated_at   TEXT NOT NULL DEFAULT (datetime('now')),
    last_used_at   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source, source_hash, model, max_chars)
);

CREATE INDEX IF NOT EXISTS idx_document_summaries_source
    ON document_summaries(source);
CREATE INDEX IF NOT EXISTS idx_document_summaries_generated
    ON document_summaries(generated_at);

CREATE TABLE IF NOT EXISTS bibliography_entries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source         TEXT NOT NULL,
    source_hash    TEXT NOT NULL,
    model          TEXT NOT NULL,
    style          TEXT NOT NULL,
    max_chars      INTEGER NOT NULL,
    chars          INTEGER,
    truncated      INTEGER NOT NULL DEFAULT 0,
    reference      TEXT NOT NULL,
    annotation     TEXT NOT NULL,
    generated_at   TEXT NOT NULL DEFAULT (datetime('now')),
    last_used_at   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source, source_hash, model, style, max_chars)
);

CREATE INDEX IF NOT EXISTS idx_bibliography_entries_source
    ON bibliography_entries(source);
CREATE INDEX IF NOT EXISTS idx_bibliography_entries_generated
    ON bibliography_entries(generated_at);


-- ---------------------------------------------------------------------------
-- PII scans -- deliberately a SUMMARY only. This table records what kinds
-- of PII a document contains and how many instances, never the actual PII
-- text itself -- storing the real names/SSNs/emails found would just create
-- a second local copy of sensitive data, working against the reason for
-- redaction in the first place. Actual redaction is computed live from the
-- real document text at the moment it's needed (see pii.py), using this
-- table only to know at a glance which files need it and what settings.pii_redaction
-- being "on" should apply to.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS pii_scans (
    source         TEXT PRIMARY KEY,   -- rag.py's relative path
    entity_types   TEXT,               -- JSON array, e.g. ["PERSON","EMAIL_ADDRESS"]
    finding_count  INTEGER NOT NULL DEFAULT 0,
    scanned_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
