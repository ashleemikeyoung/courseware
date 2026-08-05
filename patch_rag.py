"""
patch_rag.py — make rag.py project-aware. Run once.

    python patch_rag.py           # show what would change
    python patch_rag.py --apply   # back up and apply

Three changes, all surgical:

  1. scan_documents walks subfolders. It currently uses iterdir(), so every
     project folder you make is invisible to it.

  2. Chunk IDs and the "source" metadata become paths relative to the documents
     root, so 'thesis/notes.md::0' rather than 'notes.md::0'. Without this,
     thesis/notes.md and client/notes.md collide on the same IDs and the second
     one indexed silently overwrites the first.

  3. search() and _semantic_search() accept an optional project, filtering by
     metadata so retrieval stays inside one project.

Everything else in rag.py is untouched. Your MD5 change detection, loaders,
chunker, and the stdout redirect all keep working exactly as they do now.

After applying, RESCAN. Existing chunks carry the old bare-filename IDs and no
project metadata, so they will not be found by a project-scoped search. The
patcher can clear the collection for you so the rescan rebuilds cleanly.
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

RAG = Path(__file__).parent / "rag.py"

EDITS = [
    (
        "recursive file walk",
        """    current_files = {
        f.name: file_hash(f)
        for f in folder_path.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    }""",
        """    # Walk subfolders. Each top-level folder under the documents root is a
    # project; files loose at the root belong to "unfiled". Keys are paths
    # relative to the root, which makes them unique across projects.
    current_files = {
        str(f.relative_to(folder_path)): file_hash(f)
        for f in folder_path.rglob("*")
        if f.is_file()
        and f.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(part in _IGNORED_DIRS or part.startswith(".")
                    for part in f.relative_to(folder_path).parts[:-1])
    }""",
    ),
    (
        "project-qualified chunk ids and metadata",
        """    ids = [f"{file.name}::{i}" for i in range(len(chunks))]
    embeddings = embedder.encode(chunks).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[
            {"source": file.name, "file_hash": current_hash}
            for _ in chunks
        ],
    )""",
        """    # Source is the path relative to the documents root, so two projects can
    # both hold a notes.md without their chunk IDs colliding.
    try:
        rel = str(file.resolve().relative_to(Path(DOCUMENTS_FOLDER)))
    except ValueError:
        rel = file.name
    project = _project_of(rel)

    ids = [f"{rel}::{i}" for i in range(len(chunks))]
    embeddings = embedder.encode(chunks).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[
            {"source": rel, "project": project,
             "filename": file.name, "file_hash": current_hash}
            for _ in chunks
        ],
    )""",
    ),
    (
        "index_file receives relative paths",
        """    for filename, current_hash in current_files.items():
        file = folder_path / filename""",
        """    for filename, current_hash in current_files.items():
        file = folder_path / filename  # filename is now a relative path""",
    ),
    (
        "project filter on semantic search",
        '''def _semantic_search(question: str, n_results: int = 5):
    """Pure semantic vector search."""
    question_embedding = embedder.encode([question])[0]
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=min(n_results, collection.count()),
    )
    return results''',
        '''def _semantic_search(question: str, n_results: int = 5, project: str = None):
    """Pure semantic vector search, optionally scoped to one project."""
    question_embedding = embedder.encode([question])[0]
    kwargs = {
        "query_embeddings": [question_embedding.tolist()],
        "n_results": min(n_results, collection.count()),
    }
    if project:
        kwargs["where"] = {"project": project}
    results = collection.query(**kwargs)
    return results''',
    ),
    (
        "project filter on combined search",
        """def search(question: str, n_results: int = 5):""",
        """def search(question: str, n_results: int = 5, project: str = None):""",
    ),
    (
        "scope filename matching to the project",
        """    filename_matches = []
    for i, meta in enumerate(all_data["metadatas"]):
        source = meta.get("source", "")
        source_stem = Path(source).stem.lower()""",
        """    filename_matches = []
    for i, meta in enumerate(all_data["metadatas"]):
        if project and meta.get("project") != project:
            continue
        source = meta.get("source", "")
        source_stem = Path(source).stem.lower()""",
    ),
    (
        "pass the project through the blended path",
        """        semantic = _semantic_search(question, n_results=max(1, n_results - 2))""",
        """        semantic = _semantic_search(question, n_results=max(1, n_results - 2),
                                    project=project)""",
    ),
    (
        "pass the project through the fallback path",
        """    return _semantic_search(question, n_results)""",
        """    return _semantic_search(question, n_results, project=project)""",
    ),
    (
        "helpers",
        """def index_file(file: Path, current_hash: str) -> int:""",
        '''_IGNORED_DIRS = {".git", ".obsidian", "__pycache__", "node_modules",
                 ".venv", "venv", ".trash", ".writer"}


def _project_of(rel_path: str) -> str:
    """Top folder under the documents root, or 'unfiled' for loose files."""
    parts = Path(rel_path).parts
    return parts[0] if len(parts) > 1 else "unfiled"


def index_file(file: Path, current_hash: str) -> int:''',
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--clear-index", action="store_true",
                    help="also wipe the collection so the next rescan rebuilds clean")
    args = ap.parse_args()

    if not RAG.exists():
        print(f"  No rag.py at {RAG}")
        return 1

    src = RAG.read_text()

    if "_project_of" in src:
        print("  rag.py already looks patched. Nothing to do.")
        return 0

    missing = []
    for label, old, _ in EDITS:
        if src.count(old) != 1:
            missing.append((label, src.count(old)))

    if missing:
        print("  Cannot patch safely. These sections did not match exactly once:\n")
        for label, n in missing:
            print(f"    {label}  (found {n} times, expected 1)")
        print("\n  Your rag.py has diverged from the version I read. Nothing was")
        print("  changed. Send me the current file and I will redo the patch.")
        return 1

    out = src
    for label, old, new in EDITS:
        out = out.replace(old, new)
        print(f"  ok  {label}")

    if not args.apply:
        print("\n  All 9 edits matched. Re-run with --apply to write them.")
        return 0

    backup = RAG.with_suffix(f".py.bak-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(RAG, backup)
    RAG.write_text(out)
    print(f"\n  Patched. Backup at {backup.name}")

    if args.clear_index:
        sys.path.insert(0, str(RAG.parent))
        import rag
        n = rag.collection.count()
        if n:
            rag.collection.delete(ids=rag.collection.get()["ids"])
        print(f"  Cleared {n} old chunks.")

    print("\n  Now rescan so everything picks up project metadata:")
    print("      python rag.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
