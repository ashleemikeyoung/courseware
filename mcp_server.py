"""
RAG MCP Server
Exposes your local document index to Claude Desktop via MCP.
Claude Desktop calls search_documents and get_document_list as tools,
everything runs locally, no extra API charges.

Setup:
  pip install mcp
  Add to Claude Desktop config (see bottom of this file for the JSON snippet)
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

# Import from your existing rag.py
import projects
from rag import (
    search,
    scan_documents,
    scan_mailboxes,
    get_indexed_sources,
    collection,
    DOCUMENTS_FOLDER,
    ingest_content,
    read_indexed_source_text,
)
import citations

# orchestrator.py, ask.py, and summarize.py are intentionally NOT imported
# up here at module load time. All three pull in pii.py (which
# hard-requires presidio, no try/except by design -- see orchestrator.py's
# own comment) and memory_client, and this server's read-only search tools
# have worked fine without either. Importing any of them eagerly would mean
# a missing presidio install or an unreachable memory-db takes down
# search_documents/get_document_list/rescan_documents too, not just the
# tool that actually needs it. Each is imported lazily inside its own
# handler instead, so a failure stays scoped to one tool.

# ---------------------------------------------------------------------------
# MCP Server setup
# ---------------------------------------------------------------------------

server = Server(
    "rag-server",
    instructions=(
        "This is ElRoi's built-in local RAG and document storage server. "
        "Use these tools by default for Ashlee's local documents, indexed "
        "files, worksets, lessons, courseware, and document storage. If the "
        "user asks to connect to the RAG MCP server, connect to MCP, use RAG, "
        "search documents, read a file, rescan documents, check whether a file "
        "is indexed, list available projects, or answer from saved documents, "
        "call these tools directly. Do not ask for SSH, FTP, hostnames, ports, "
        "or credentials unless the user explicitly says they mean an external "
        "network server. Project scopes may be top-level folders, nested "
        "document folders, or unique folder names discovered from the indexed "
        "document paths."
    ),
)


def _project_scope(project: str | None) -> str | None:
    """Normalize a user-supplied project/folder scope."""
    scope = (project or "").strip().strip("/")
    if not scope or scope == projects.ALL:
        return None
    return scope


def _scope_root(scope: str | None) -> str | None:
    if not scope:
        return None
    return Path(scope).parts[0] if Path(scope).parts else scope


def _scope_search_root(scope: str | None) -> str | None:
    """Best top-level project to search before applying the final scope filter."""
    scope = _project_scope(scope)
    if not scope:
        return None
    parts = Path(scope).parts
    if len(parts) > 1:
        return parts[0]
    indexed = get_indexed_sources()
    if any(projects.project_of(source) == scope for source in indexed):
        return scope
    roots = {
        projects.project_of(source)
        for source in indexed
        if scope in Path(source).parts
    }
    return next(iter(roots)) if len(roots) == 1 else None


def _source_in_scope(source: str, scope: str | None) -> bool:
    """Accept top-level projects, nested folders, and unique folder aliases."""
    scope = _project_scope(scope)
    if not scope:
        return True
    source_norm = source.strip("/")
    scope_norm = scope.strip("/")
    if "/" not in scope_norm:
        return (
            projects.project_of(source_norm) == scope_norm
            or scope_norm in Path(source_norm).parts
        )
    return source_norm == scope_norm or source_norm.startswith(f"{scope_norm}/")


def _filter_indexed_by_scope(indexed: dict, scope: str | None) -> dict:
    scope = _project_scope(scope)
    if not scope:
        return indexed
    return {k: v for k, v in indexed.items() if _source_in_scope(k, scope)}


# ---------------------------------------------------------------------------
# Tool: search_documents
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_documents",
            description=(
                "Search the private local document index for information relevant "
                "to a question or topic. Returns the most relevant chunks from the "
                "indexed files along with their source paths. Use this whenever "
                "the user asks about something that might be in their documents, "
                "notes, contracts, or personal files. Documents are organised into "
                "projects, which are folders. Pass a project to keep the search "
                "inside it, which matters when the same word means different "
                "things in different projects. If the user asks about one named "
                "file, or search results from a file look incomplete, call "
                "read_document next to inspect the raw extracted text. Call "
                "list_projects first if you are not sure which one the user means."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question to look up in the documents",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 3, max 10)",
                        "default": 3,
                    },
                    "project": {
                        "type": "string",
                        "description": (
                            "Restrict the search to one project folder. Omit to "
                            "search across every project."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_document_list",
            description=(
                "Get a list of all documents currently indexed in the local RAG system, "
                "including filenames and total chunk counts. Use this to tell the user "
                "what files are available to search, then use read_document when the "
                "user needs the contents of a specific listed file."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": (
                            "Restrict the list to one project folder. Omit to "
                            "list documents across every project."
                        ),
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="list_projects",
            description=(
                "List the document projects available locally, with how many "
                "files and indexed chunks each holds. Projects are folders under "
                "the documents root. Use this to tell the user what is available, "
                "or to resolve which project they mean before searching."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="rescan_documents",
            description=(
                "Trigger a rescan of the documents folder to pick up any new, "
                "changed, or deleted files. Use this when the user says they added "
                "new files and wants them indexed."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="rescan_mailboxes",
            description=(
                "Index local Maildir mailboxes synced from IMAP by OfflineIMAP "
                "or offlineimap3. Configure mailbox roots with IMAP_MAILDIR_ROOTS "
                "as a comma-separated list of Maildir/account directories, then "
                "call this tool to add or refresh email messages in the same "
                "semantic index used by search_documents and ask_local."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "roots": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional absolute Maildir root paths. If omitted, "
                            "IMAP_MAILDIR_ROOTS is used."
                        ),
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Re-index messages even if their content hash has not changed.",
                        "default": False,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="ingest_content",
            description=(
                "Save text content into the local documents folder and index it "
                "immediately, so it becomes searchable right away and survives "
                "future rescans. Use this when the user attaches or pastes a file "
                "in this chat that they want added to the RAG index -- extract "
                "the text yourself first, then pass it here with a filename. "
                "The content is saved as a .md file under the documents folder, "
                "so it is treated exactly like any other file you'd drop in by hand."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "A descriptive filename for the content, e.g. "
                            "'q3-board-notes.md'. The extension is normalized to .md."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content to save and index.",
                    },
                    "project": {
                        "type": "string",
                        "description": (
                            "Project folder to file this under. Omit to file it "
                            "under 'unfiled'."
                        ),
                    },
                },
                "required": ["filename", "content"],
            },
        ),
        Tool(
            name="ask_local",
            description=(
                "Answer a question grounded in the document index, with inline "
                "citation markers like [C1] pointing back to specific passages. "
                "Retrieval, grounding, and generation all run locally through "
                "Ollama -- the same underlying function the web app's Ask tab "
                "uses, so this gives the same quality of answer Claude Desktop "
                "would if you asked in the browser. Use this when the user wants "
                "an actual synthesized answer or summary drawn from one or more "
                "documents, not just raw search results. Slower than "
                "search_documents since it's a full generation call."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "The question to answer, or the summary/synthesis "
                            "request."
                        ),
                    },
                    "project": {
                        "type": "string",
                        "description": (
                            "Restrict retrieval to one project folder. Omit to "
                            "search across every project."
                        ),
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="read_document",
            description=(
                "Read the raw extracted text of one indexed local document directly "
                "from disk, using the same file extractor used by indexing. Use this "
                "when the user names a file, asks you to inspect a document's full "
                "contents, asks whether a saved file is blank or incomplete, or when "
                "search_documents only returns narrow chunks such as rubrics or tables. "
                "This is not a Chroma chunk search; it opens the source document and "
                "returns its extracted text. For long documents, use start and "
                "max_chars to page through the text."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "Indexed source path or filename, e.g. "
                            "'GCU/Topic5 DQ1.docx' or 'Topic5 DQ1.docx'."
                        ),
                    },
                    "project": {
                        "type": "string",
                        "description": (
                            "Restrict filename resolution to one project folder. "
                            "Omit to resolve across every project."
                        ),
                    },
                    "start": {
                        "type": "integer",
                        "description": "Character offset to start reading from. Default 0.",
                        "default": 0,
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": (
                            "Maximum characters to return. Default 20000, max 100000."
                        ),
                        "default": 20000,
                    },
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="summarize_documents",
            description=(
                "Search the index for documents matching a term, then read and "
                "summarize each matching file directly off disk -- the full "
                "document text, not retrieved chunks. Use this for 'summarize "
                "the article by X' or 'what's in the files about Y' style "
                "requests, where ask_local's chunk-based grounding might only "
                "see fragments. Slower than ask_local when several files match, "
                "since each one gets its own full-document summarization pass, "
                "but the summary for each file is built from everything in it."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search term identifying the document(s) to "
                            "summarize, e.g. an author name, title fragment, "
                            "or topic."
                        ),
                    },
                    "project": {
                        "type": "string",
                        "description": (
                            "Restrict the search to one project folder. Omit to "
                            "search across every project."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="create_workset",
            description=(
                "Create or reopen a named attachment workset. A workset is a "
                "temporary bundle of files/text for Claude-style attachment "
                "analysis: summarize each item, then synthesize across exactly "
                "that bundle without adding it to the permanent documents index."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short workset name, e.g. 'august-articles'.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional note about what this bundle is for.",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="list_worksets",
            description=(
                "List attachment worksets and the files currently in each bundle."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="ingest_workset_file",
            description=(
                "Copy a local file into an attachment workset and extract its text "
                "for full-document summarization/synthesis. Supports the same file "
                "types as the RAG extractor, with pdftotext-first extraction for PDFs. "
                "This does not add the file to the permanent Chroma documents index."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workset": {
                        "type": "string",
                        "description": "Name of the workset to add this file to.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the local file to ingest.",
                    },
                },
                "required": ["workset", "path"],
            },
        ),
        Tool(
            name="ingest_workset_text",
            description=(
                "Add already-extracted text to an attachment workset. Use this when "
                "the client has attachment text but not a local file path."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workset": {
                        "type": "string",
                        "description": "Name of the workset to add this text to.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Display filename for the text item.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full extracted text/content.",
                    },
                },
                "required": ["workset", "filename", "content"],
            },
        ),
        Tool(
            name="summarize_workset",
            description=(
                "Summarize every document in an attachment workset from full extracted "
                "text. Long documents are summarized in parts and then combined. "
                "Summaries are saved in the workset manifest for later synthesis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workset": {
                        "type": "string",
                        "description": "Name of the workset to summarize.",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Regenerate summaries even if cached.",
                        "default": False,
                    },
                },
                "required": ["workset"],
            },
        ),
        Tool(
            name="synthesize_workset",
            description=(
                "Synthesize across all documents in an attachment workset. Produces "
                "shared themes, tensions, unique contributions, gaps, and next steps, "
                "citing workset document IDs like [D1]."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workset": {
                        "type": "string",
                        "description": "Name of the workset to synthesize.",
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "Optional synthesis question or focus, e.g. 'compare "
                            "methods and findings'."
                        ),
                    },
                },
                "required": ["workset"],
            },
        ),
        Tool(
            name="query_document_registry",
            description=(
                "Query libSQL document metadata for saved project files treated as uploaded documents. "
                "Searches labels, synopses, genres, and themes without using Chroma."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for."},
                    "project": {"type": "string", "description": "Optional project folder."},
                    "genre": {"type": "string", "description": "Optional genre filter."},
                    "theme": {"type": "string", "description": "Optional theme filter."},
                    "exclude_genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional genres to omit, e.g. ['chat export'].",
                    },
                    "limit": {"type": "integer", "default": 20},
                },
                "required": [],
            },
        ),
        Tool(
            name="lesson",
            description=(
                "Build a grounded corpus on any subject from open courseware, "
                "index it as a project, and draft an explainer from it. Leads "
                "with MIT OpenCourseWare and descends to peer university "
                "courseware, open textbooks, and primary literature (arXiv, "
                "DOAJ) only when MIT comes back thin. Everything is fetched "
                "from openly licensed sources, given a provenance header, and "
                "indexed, so afterwards the subject is permanently searchable "
                "and citable offline. Use this when the user wants to learn or "
                "teach a subject the index does not cover yet, rather than "
                "answering from your own knowledge. Slow: a full run fetches "
                "and extracts a dozen documents and then drafts from them, so "
                "expect minutes, not seconds."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": (
                            "What to learn, in plain words, e.g. 'stochastic "
                            "dominance' or 'Bayesian hierarchical models'."
                        ),
                    },
                    "project": {
                        "type": "string",
                        "description": (
                            "Project folder to build into. Omit to use a slug "
                            "of the subject."
                        ),
                    },
                    "max_docs": {
                        "type": "integer",
                        "default": 10,
                        "description": "Documents to index per tier (capped at 12).",
                    },
                    "tiers": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["mit", "peer", "textbook", "literature"],
                        },
                        "description": (
                            "Pin an explicit tier list, in order, skipping the "
                            "coverage check. Omit to start at MIT and descend "
                            "only if coverage is thin. Pass ['literature'] for a "
                            "peer-reviewed-only pass."
                        ),
                    },
                    "explainer": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "False to build and index the corpus without "
                            "drafting anything."
                        ),
                    },
                },
                "required": ["subject"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:

    if name == "search_documents":
        return await handle_search(arguments)

    elif name == "get_document_list":
        return await handle_document_list(arguments or {})

    elif name == "list_projects":
        return await handle_projects()

    elif name == "rescan_documents":
        return await handle_rescan()

    elif name == "rescan_mailboxes":
        return await handle_rescan_mailboxes(arguments or {})

    elif name == "ingest_content":
        return await handle_ingest_content(arguments or {})

    elif name == "ask_local":
        return await handle_ask_local(arguments or {})

    elif name == "read_document":
        return await handle_read_document(arguments or {})

    elif name == "summarize_documents":
        return await handle_summarize_documents(arguments or {})

    elif name == "create_workset":
        return await handle_create_workset(arguments or {})

    elif name == "list_worksets":
        return await handle_list_worksets()

    elif name == "ingest_workset_file":
        return await handle_ingest_workset_file(arguments or {})

    elif name == "ingest_workset_text":
        return await handle_ingest_workset_text(arguments or {})

    elif name == "summarize_workset":
        return await handle_summarize_workset(arguments or {})

    elif name == "synthesize_workset":
        return await handle_synthesize_workset(arguments or {})

    elif name == "query_document_registry":
        return await handle_query_document_registry(arguments or {})

    elif name == "lesson":
        return await handle_lesson(arguments or {})

    return CallToolResult(
        content=[TextContent(type="text", text=f"Unknown tool: {name}")]
    )


async def handle_query_document_registry(arguments: dict) -> CallToolResult:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent / "memory"))
        from memory_client import search_document_uploads
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Could not load document registry: {e}")]
        )
    try:
        rows = search_document_uploads(
            query=(arguments.get("query") or "").strip() or None,
            project=(arguments.get("project") or "").strip() or None,
            genre=(arguments.get("genre") or "").strip() or None,
            theme=(arguments.get("theme") or "").strip() or None,
            exclude_genres=arguments.get("exclude_genres") or [],
            limit=int(arguments.get("limit", 20) or 20),
        )
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"documents": rows}, indent=2))]
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Document registry query error: {e}")]
        )


async def handle_search(arguments: dict) -> CallToolResult:
    query = arguments.get("query", "").strip()
    n_results = min(int(arguments.get("n_results", 3)), 10)
    project = _project_scope(arguments.get("project"))

    if not query:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: query cannot be empty")]
        )

    if collection.count() == 0:
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="No documents are indexed yet. Ask Claude to run rescan_documents after adding files to the documents folder."
            )]
        )

    try:
        search_project = _scope_search_root(project)
        search_limit = n_results if project == search_project else 50
        results = search(query, n_results=search_limit, project=search_project)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        if project and project != search_project:
            scoped = [
                (doc, meta)
                for doc, meta in zip(docs, metas)
                if _source_in_scope(meta.get("source", ""), project)
            ][:n_results]
            docs = [doc for doc, _meta in scoped]
            metas = [meta for _doc, meta in scoped]

        # Citation correlation, via the shared helper in citations.py -- see
        # that module's docstring for the full reasoning. This used to be
        # an inline duplicate of the same logic in orchestrator.py; both
        # now call the one shared implementation.
        seen_sources = {m.get("source") for m in metas}
        citation_blocks = []
        for hit in citations.topup(query, seen_sources, project=search_project):
            if "_warning" in hit:
                citation_blocks.append(f"[Warning: citation lookup failed: {hit['_warning']}]")
                continue
            if project and not _source_in_scope(hit.get("source", ""), project):
                continue
            citation_blocks.append(
                "--- Verified citation record (from memory-db, not chroma_db) ---\n"
                f"Source: {hit['source']}\n"
                f"Title: {hit['title']}\n"
                f"Author(s): {hit['authors']}\n"
                f"Citation: {hit['source_line']}\n"
            )

        if not docs and not citation_blocks:
            return CallToolResult(
                content=[TextContent(type="text", text="No relevant documents found for that query.")]
            )

        scope = f" in project '{project}'" if project else " across all projects"
        output_parts = [
            f"Found {len(docs)} relevant chunk(s) for: '{query}'{scope}\n"]

        for i, (doc, meta) in enumerate(zip(docs, metas)):
            source = meta.get("source", "unknown")
            output_parts.append(f"--- Result {i + 1} (from {source}) ---")
            output_parts.append(doc)
            output_parts.append("")

        output_parts.extend(citation_blocks)

        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(output_parts))]
        )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Search error: {str(e)}")]
        )


async def handle_projects() -> CallToolResult:
    stats = projects.stats(collection)
    if not stats:
        return CallToolResult(content=[TextContent(
            type="text",
            text=("No projects yet. Make a folder under "
                  f"{projects.DOCUMENTS_ROOT} and run rescan_documents."))])
    lines = [f"Projects under {projects.DOCUMENTS_ROOT}:\n"]
    for p in stats:
        lines.append(f"  {p['name']:<24} {p['files']:>4} files, "
                     f"{p['chunks']:>6} indexed chunks")
    return CallToolResult(content=[TextContent(type="text", text="\n".join(lines))])


async def handle_document_list(arguments: dict) -> CallToolResult:
    project = _project_scope(arguments.get("project"))
    indexed = _filter_indexed_by_scope(get_indexed_sources(), project)

    if not indexed:
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="No documents are currently indexed. Add files to the documents folder and run rescan_documents."
            )]
        )

    total_chunks = collection.count()
    lines = [
        f"Indexed documents ({len(indexed)} file(s), {total_chunks} total chunks):\n"
    ]
    for filename in sorted(indexed.keys()):
        lines.append(f"  {filename}")

    lines.append(f"\nDocuments folder: {Path(DOCUMENTS_FOLDER).resolve()}")

    return CallToolResult(
        content=[TextContent(type="text", text="\n".join(lines))]
    )


async def handle_rescan() -> CallToolResult:
    try:
        summary = scan_documents(verbose=False)
        lines = [
            "Rescan complete:",
            f"  New files indexed:    {len(summary['new'])}",
            f"  Updated files:        {len(summary['updated'])}",
            f"  Removed files:        {len(summary['removed'])}",
            f"  Unchanged files:      {len(summary['unchanged'])}",
            f"  Total chunks now:     {collection.count()}",
        ]
        if summary["new"]:
            lines.append(f"\nNew files: {', '.join(summary['new'])}")
        if summary["updated"]:
            lines.append(f"Updated files: {', '.join(summary['updated'])}")
        if summary["removed"]:
            lines.append(f"Removed files: {', '.join(summary['removed'])}")

        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(lines))]
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Rescan error: {str(e)}")]
        )


async def handle_rescan_mailboxes(arguments: dict) -> CallToolResult:
    try:
        roots = arguments.get("roots") or None
        force = bool(arguments.get("force", False))
        summary = scan_mailboxes(roots=roots, verbose=False, force=force)
        lines = [
            "Mailbox rescan complete:",
            f"  Mailbox folders scanned: {len(summary.get('maildirs', []))}",
            f"  New messages indexed:     {len(summary['new'])}",
            f"  Updated messages:         {len(summary['updated'])}",
            f"  Removed messages:         {len(summary['removed'])}",
            f"  Unchanged messages:       {len(summary['unchanged'])}",
            f"  Total chunks now:         {collection.count()}",
        ]
        if not summary.get("maildirs"):
            roots = ", ".join(summary.get("roots", [])) or "(none configured)"
            lines.append(f"\nNo Maildir folders found under: {roots}")
        if summary["new"]:
            lines.append(f"\nNew messages: {', '.join(summary['new'][:20])}")
            if len(summary["new"]) > 20:
                lines.append(f"... and {len(summary['new']) - 20} more")
        if summary["updated"]:
            lines.append(f"Updated messages: {', '.join(summary['updated'][:20])}")
            if len(summary["updated"]) > 20:
                lines.append(f"... and {len(summary['updated']) - 20} more")
        if summary["removed"]:
            lines.append(f"Removed messages: {', '.join(summary['removed'][:20])}")
            if len(summary["removed"]) > 20:
                lines.append(f"... and {len(summary['removed']) - 20} more")
        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(lines))]
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Mailbox rescan error: {str(e)}")]
        )


async def handle_ingest_content(arguments: dict) -> CallToolResult:
    filename = (arguments.get("filename") or "").strip()
    content = arguments.get("content") or ""
    project = (arguments.get("project") or "").strip() or None

    if not filename:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: filename cannot be empty")]
        )
    if not content.strip():
        return CallToolResult(
            content=[TextContent(type="text", text="Error: content cannot be empty")]
        )

    try:
        rel, chunk_count = ingest_content(filename, content, project=project)
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=(
                    f"Saved and indexed '{rel}' ({chunk_count} chunks).\n"
                    f"Total chunks in index: {collection.count()}"
                ),
            )]
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Ingest error: {str(e)}")]
        )


async def handle_lesson(arguments: dict) -> CallToolResult:
    """
    lesson.py is imported here rather than at module load, following the same
    rule as orchestrator/ask/summarize above: it reaches rag.py, writer.py and
    the network, and none of that should be able to take down search_documents.

    build_lesson() is synchronous and slow -- it fetches a dozen documents,
    extracts them, and then runs a long local generation. Called directly it
    would block this server's event loop for minutes, so it goes to a worker
    thread and the coroutine awaits that instead.
    """
    subject = (arguments.get("subject") or "").strip()
    if not subject:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: subject cannot be empty")]
        )

    try:
        import lesson as lesson_mod
    except Exception as e:
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"lesson module unavailable: {type(e).__name__}: {e}")]
        )

    try:
        result = await asyncio.to_thread(
            lesson_mod.build_lesson,
            subject,
            project=(arguments.get("project") or "").strip() or None,
            max_docs=int(arguments.get("max_docs") or 10),
            tiers=arguments.get("tiers") or None,
            explainer=arguments.get("explainer", True),
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(
                type="text", text=f"Lesson error: {type(e).__name__}: {e}")]
        )

    lines = [
        f"Lesson built for '{result['subject']}' in project '{result['project']}'.",
        f"Indexed {result['indexed']} documents, skipped {result['skipped']}.",
        f"Tiers used: {', '.join(result['tiers_used']) or 'none'}.",
    ]
    if result.get("manifest"):
        lines.append(f"Manifest: {result['manifest']}")
    if result.get("explainer"):
        lines.append(f"Explainer: {result['explainer']}")
    else:
        lines.append("Explainer: not written (skipped, or the local model was "
                     "unavailable). The corpus is still indexed.")
    lines.append(f"Total chunks in index: {collection.count()}")

    indexed = [r for r in result["rows"] if r["status"] == "indexed"]
    if indexed:
        lines.append("")
        lines.append("Indexed:")
        for r in indexed:
            lines.append(f"  [{r['tier']}] {r['title']}")
            lines.append(f"        {r['url']}")

    skipped = [r for r in result["rows"] if r["status"] != "indexed"]
    if skipped:
        lines.append("")
        lines.append("Skipped:")
        for r in skipped:
            lines.append(f"  {r['title']} -- {r['status']}")

    return CallToolResult(content=[TextContent(type="text", text="\n".join(lines))])


async def handle_ask_local(arguments: dict) -> CallToolResult:
    question = (arguments.get("question") or "").strip()
    project = _project_scope(arguments.get("project"))

    if not question:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: question cannot be empty")]
        )

    if collection.count() == 0:
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="No documents are indexed yet. Ask Claude to run rescan_documents (or ingest_content) first."
            )]
        )

    # Lazy import -- see the comment by the top-level rag import block for
    # why this isn't pulled in until this specific tool is called. ask.py
    # itself imports writer.py, which is where the pii/memory_client
    # dependency actually lives.
    try:
        import ask as ask_module
    except Exception as e:
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=(
                    f"Could not load ask.py: {e}\n"
                    "This usually means a dependency (presidio, memory-db, etc.) "
                    "isn't set up. search_documents still works without it."
                ),
            )]
        )

    try:
        # The same function app.py's Ask tab calls (POST /api/ask), so MCP
        # and the web UI answer a question the same way: grounded in the
        # index, cited inline with [C1]-style markers, with the same
        # degenerate-response retry. This used to call orchestrator.py's
        # orchestrate() instead, a separate and older implementation that
        # routes to specialist coder/reasoning models but has none of
        # ask.py's citation or retry handling -- the two interfaces could
        # give genuinely different answers to the same question. Now they
        # share one implementation, so they can't drift apart.
        #
        # No model is passed -- ask.ask() defaults to config.ASK_MODEL on
        # its own now, so this tool doesn't need to know or care what that
        # default is, same as app.py no longer does.
        search_project = _scope_search_root(project)
        result = ask_module.ask(
            [{"role": "user", "content": question}],
            project=search_project,
            ground=True,
        )
        if project and project != search_project and result.get("evidence"):
            result["evidence"] = {
                key: value
                for key, value in result["evidence"].items()
                if _source_in_scope(value.get("source", ""), project)
            }
        lines = [result["text"]]
        if result["evidence"]:
            sources = sorted({e["source"] for e in result["evidence"].values()})
            lines.append(f"\nSources: {', '.join(sources)}")
        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(lines))]
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Ask error: {str(e)}")]
        )


def _resolve_indexed_source(source_query: str, project: str = None):
    indexed = _filter_indexed_by_scope(get_indexed_sources(), project)

    needle = (source_query or "").strip()
    needle_lower = needle.lower()
    if not needle_lower:
        return None, []

    for source in indexed:
        if source.lower() == needle_lower:
            return source, []

    exact_name = [source for source in indexed
                  if Path(source).name.lower() == needle_lower]
    if len(exact_name) == 1:
        return exact_name[0], []
    if len(exact_name) > 1:
        return None, sorted(exact_name)

    contained = [source for source in indexed if needle_lower in source.lower()]
    if len(contained) == 1:
        return contained[0], []
    if len(contained) > 1:
        return None, sorted(contained)

    try:
        import summarize
        detected = summarize.detect_file_reference(needle, project=project)
    except Exception:
        detected = None
    if detected:
        return detected, []

    return None, []


async def handle_read_document(arguments: dict) -> CallToolResult:
    source_query = (arguments.get("source") or "").strip()
    project = _project_scope(arguments.get("project"))
    start = max(int(arguments.get("start", 0) or 0), 0)
    max_chars = int(arguments.get("max_chars", 20000) or 20000)
    max_chars = min(max(max_chars, 1), 100000)

    if not source_query:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: source cannot be empty")]
        )

    source, candidates = _resolve_indexed_source(source_query, project=project)
    if not source:
        if candidates:
            lines = [
                f"More than one indexed document matched '{source_query}'.",
                "Use one exact source path:",
                "",
            ]
            lines.extend(f"  {candidate}" for candidate in candidates[:25])
            if len(candidates) > 25:
                lines.append(f"  ... and {len(candidates) - 25} more")
            return CallToolResult(
                content=[TextContent(type="text", text="\n".join(lines))]
            )
        scope = f" in project '{project}'" if project else ""
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"No indexed document matched '{source_query}'{scope}.",
            )]
        )

    if source.startswith("mail/"):
        path = source
        text = read_indexed_source_text(source) or ""
    else:
        try:
            import rag as rag_module
            import summarize
            path = summarize.resolve_path(source)
            text = rag_module.load_file(path) or ""
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Read error: {str(e)}")]
            )

    total_chars = len(text)
    end = min(start + max_chars, total_chars)
    excerpt = text[start:end]
    truncated = end < total_chars
    if start >= total_chars and total_chars:
        excerpt = ""

    lines = [
        f"Source: {source}",
        f"Path: {path}",
        f"Characters extracted: {total_chars}",
        f"Returned range: {start}-{end}",
        f"Truncated: {'yes' if truncated else 'no'}",
        "",
        "--- Document text ---",
        excerpt,
    ]
    if not text.strip():
        lines.append(
            "\n[No text was extracted. The file may be blank, scanned without OCR, "
            "or in a format the extractor cannot read.]"
        )

    return CallToolResult(
        content=[TextContent(type="text", text="\n".join(lines))]
    )


async def handle_summarize_documents(arguments: dict) -> CallToolResult:
    query = (arguments.get("query") or "").strip()
    project = (arguments.get("project") or "").strip() or None

    if not query:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: query cannot be empty")]
        )

    # Lazy import -- same reasoning as handle_ask_local. summarize.py
    # imports writer.py (for ask_ollama_long), which is where the
    # pii/memory_client dependency actually lives.
    try:
        import summarize
    except Exception as e:
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=(
                    f"Could not load summarize.py: {e}\n"
                    "This usually means a dependency (presidio, memory-db, etc.) "
                    "isn't set up. search_documents still works without it."
                ),
            )]
        )

    try:
        # The same function orchestrator.py's /summarize command and
        # app.py's /api/summarize route call -- search finds which files
        # are relevant (rag.search() plus the citations.py top-up, same
        # signal search_documents uses), then each matched file gets read
        # straight off disk and summarized whole, not from retrieved
        # chunks. One implementation, so "search then summarize" behaves
        # the same regardless of which interface asked for it.
        results = summarize.summarize_search(query, project=project)
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Search error: {str(e)}")]
        )

    if not results:
        scope = f" in project '{project}'" if project else ""
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"No indexed documents matched '{query}'{scope}.",
            )]
        )

    parts = [f"Found and summarized {len(results)} document(s) for '{query}':\n"]
    for r in results:
        parts.append(f"--- {r['source']} ---")
        if "error" in r:
            parts.append(f"[Could not summarize: {r['error']}]\n")
            continue
        note = f" (truncated at {r['chars']} chars)" if r.get("truncated") else ""
        parts.append(f"{r['summary']}{note}\n")

    return CallToolResult(
        content=[TextContent(type="text", text="\n".join(parts))]
    )


def _json_result(payload: dict) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, indent=2))]
    )


def _load_worksets():
    try:
        import worksets
        return worksets, None
    except Exception as e:
        return None, CallToolResult(
            content=[TextContent(type="text", text=f"Could not load worksets.py: {e}")]
        )


async def handle_create_workset(arguments: dict) -> CallToolResult:
    worksets, error = _load_worksets()
    if error:
        return error
    name = (arguments.get("name") or "").strip()
    description = (arguments.get("description") or "").strip()
    if not name:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: name cannot be empty")]
        )
    try:
        return _json_result(worksets.create_workset(name, description=description))
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Create workset error: {e}")]
        )


async def handle_list_worksets() -> CallToolResult:
    worksets, error = _load_worksets()
    if error:
        return error
    try:
        rows = worksets.list_worksets()
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"List worksets error: {e}")]
        )
    if not rows:
        return CallToolResult(
            content=[TextContent(type="text", text="No worksets found.")]
        )
    return _json_result({"worksets": rows})


async def handle_ingest_workset_file(arguments: dict) -> CallToolResult:
    worksets, error = _load_worksets()
    if error:
        return error
    name = (arguments.get("workset") or "").strip()
    path = (arguments.get("path") or "").strip()
    if not name or not path:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: workset and path are required.")]
        )
    try:
        doc = worksets.ingest_file(name, path)
        return _json_result({"ingested": doc})
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Workset file ingest error: {e}")]
        )


async def handle_ingest_workset_text(arguments: dict) -> CallToolResult:
    worksets, error = _load_worksets()
    if error:
        return error
    name = (arguments.get("workset") or "").strip()
    filename = (arguments.get("filename") or "").strip()
    content = arguments.get("content") or ""
    if not name or not filename:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: workset and filename are required.")]
        )
    if not content.strip():
        return CallToolResult(
            content=[TextContent(type="text", text="Error: content cannot be empty.")]
        )
    try:
        doc = worksets.ingest_text(name, filename, content)
        return _json_result({"ingested": doc})
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Workset text ingest error: {e}")]
        )


async def handle_summarize_workset(arguments: dict) -> CallToolResult:
    worksets, error = _load_worksets()
    if error:
        return error
    name = (arguments.get("workset") or "").strip()
    force = bool(arguments.get("force", False))
    if not name:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: workset cannot be empty.")]
        )
    try:
        result = worksets.summarize_workset(name, force=force)
        return _json_result(result)
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Summarize workset error: {e}")]
        )


async def handle_synthesize_workset(arguments: dict) -> CallToolResult:
    worksets, error = _load_worksets()
    if error:
        return error
    name = (arguments.get("workset") or "").strip()
    question = (arguments.get("question") or "").strip()
    if not name:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: workset cannot be empty.")]
        )
    try:
        result = worksets.synthesize_workset(name, question=question)
        lines = [
            f"Workset: {result['name']}",
            "",
            result["synthesis"],
            "",
            "Documents:",
        ]
        for doc in result["documents"]:
            lines.append(f"  [{doc['doc_id']}] {doc['label']} ({doc['filename']})")
        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(lines))]
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Synthesize workset error: {e}")]
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    print("RAG MCP Server starting...", file=sys.stderr)
    print(f"Documents folder: {Path(DOCUMENTS_FOLDER).resolve()}", file=sys.stderr)
    print(f"Projects:         {', '.join(projects.discover()) or 'none'}", file=sys.stderr)
    print(f"Chunks in index:  {collection.count()}", file=sys.stderr)
    print("Waiting for an MCP client to connect...\n", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())


# ---------------------------------------------------------------------------
# Claude Desktop config snippet
# ---------------------------------------------------------------------------
#
# Add this to your Claude Desktop config file at:
# ~/Library/Application Support/Claude/claude_desktop_config.json
#
# {
#   "mcpServers": {
#     "rag": {
#       "command": "/opt/anaconda3/envs/rag/bin/python",
#       "args": ["/Users/ash/Development/RAG/mcp_server.py"],
#       "cwd": "/Users/ash/Development/RAG"
#     }
#   }
# }
#
# If you have other MCP servers already configured, add the "rag" block
# inside your existing "mcpServers" object alongside them.
#
# After saving the config, quit and relaunch Claude Desktop.
# You should see a hammer icon in the chat input confirming tools are loaded.
