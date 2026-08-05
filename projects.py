"""
projects.py — projects are folders under the documents root.

    ~/Development/RAG/
      documents/
        thesis/            <- project "thesis"
          sources/...         (nesting is fine, the project is the TOP folder)
        client-acme/       <- project "client-acme"
        loose-notes.pdf    <- no folder, so it lands in "unfiled"
      projects/
        thesis/
          plan.json           active outline
          plans/              discarded outlines
          output/             finished documents
          bench/              comparison history

Why generated work lives in projects/ and not inside documents/:

    If drafts were written into documents/thesis/, the next rescan would index
    them, and the system would start retrieving its own output as if it were a
    source. Citations would then trace back to text the model wrote, which looks
    exactly like grounding while being the opposite of it. Keeping the two trees
    apart makes that impossible rather than merely unlikely.

A project name is the folder name. Nothing is registered or configured, so
creating a project means making a folder and rescanning.
"""

import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

_docs_env = os.getenv("DOCUMENTS_FOLDER")
DOCUMENTS_ROOT = (Path(_docs_env).resolve() if _docs_env
                  else BASE_DIR / "documents")
PROJECTS_ROOT = Path(os.getenv("PROJECTS_FOLDER", BASE_DIR / "projects"))

# Files sitting loose at the documents root, belonging to no folder.
UNFILED = "unfiled"

# Folders that are plumbing, not projects.
IGNORED = {".git", ".obsidian", "__pycache__", ".DS_Store", "node_modules",
           ".venv", "venv", ".trash", ".writer"}


def is_project_dir(p: Path) -> bool:
    return (p.is_dir() and p.name not in IGNORED
            and not p.name.startswith("."))


def discover() -> list:
    """Project names, from the folder structure. Cheap enough to call freely."""
    if not DOCUMENTS_ROOT.exists():
        return []
    names = sorted(p.name for p in DOCUMENTS_ROOT.iterdir() if is_project_dir(p))
    if any(f.is_file() and not f.name.startswith(".")
           for f in DOCUMENTS_ROOT.iterdir()):
        names.append(UNFILED)
    return names


def project_of(rel_path: str) -> str:
    """
    Map a documents-root-relative path to its project.

      'thesis/ch1/draft.docx' -> 'thesis'
      'loose.pdf'             -> 'unfiled'

    This is the single definition of that mapping. rag.py, the retriever, and
    the UI all defer to it, so the answer cannot drift between them.
    """
    parts = Path(rel_path).parts
    return parts[0] if len(parts) > 1 else UNFILED


def source_root(name: str) -> Path:
    """Where a project's documents live."""
    return DOCUMENTS_ROOT if name == UNFILED else DOCUMENTS_ROOT / name


def safe(name: str) -> str:
    """Folder names come from disk, but they also arrive over HTTP. Sanitize."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-.")
    return cleaned or UNFILED


def paths(name: str) -> dict:
    """Per-project working directories, created on demand."""
    root = PROJECTS_ROOT / safe(name)
    p = {
        "root": root,
        "plan": root / "plan.json",
        "plans": root / "plans",
        "output": root / "output",
        "bench": root / "bench",
        "history": root / "bench" / "history.jsonl",
        "sources": source_root(name),
    }
    return p


def ensure(name: str) -> dict:
    p = paths(name)
    for key in ("root", "plans", "output", "bench"):
        p[key].mkdir(parents=True, exist_ok=True)
    return p


def create(name: str) -> str:
    """Make a new project: a source folder plus its working directories."""
    n = safe(name)
    (DOCUMENTS_ROOT / n).mkdir(parents=True, exist_ok=True)
    ensure(n)
    return n


def stats(collection=None) -> list:
    """Projects with file and chunk counts, for a picker."""
    from rag import SUPPORTED_EXTENSIONS

    by_project = {}
    if collection is not None and collection.count():
        got = collection.get(include=["metadatas"])
        for meta in got["metadatas"]:
            proj = meta.get("project") or project_of(meta.get("source", ""))
            by_project[proj] = by_project.get(proj, 0) + 1

    out = []
    for name in discover():
        root = source_root(name)
        if name == UNFILED:
            files = [f for f in root.iterdir()
                     if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
        else:
            files = [f for f in root.rglob("*")
                     if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

        p = paths(name)
        out.append({
            "name": name,
            "files": len(files),
            "chunks": by_project.get(name, 0),
            "documents": len(list(p["output"].glob("*.md"))) if p["output"].exists() else 0,
            "has_plan": p["plan"].exists(),
            "path": str(root),
        })
    return out
