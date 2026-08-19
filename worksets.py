"""
worksets.py -- transient attachment bundles for full-document synthesis.

The durable RAG index answers "what is in my saved knowledge base?" Worksets
answer a different question: "given these attachments together, summarize and
synthesize just this bundle."
"""

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config import ASK_MODEL


BASE_DIR = Path(__file__).resolve().parent
WORKSETS_DIR = BASE_DIR / "worksets"
FILES_DIR = "files"
EXTRACTED_DIR = "extracted"
MANIFEST = "manifest.json"

def _load_rag():
    # Import only when file extraction needs the durable RAG stack. This keeps
    # light workset operations from loading the embedding model and Chroma.
    import rag
    return rag


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


SECTION_KEYWORDS = [
    "abstract",
    "introduction",
    "literature review",
    "method",
    "methods",
    "methodology",
    "results",
    "findings",
    "discussion",
    "conclusion",
    "recommendations",
    "implications",
    "limitations",
    "references",
]

_BOILERPLATE_MARKERS = (
    "instructions",
    "this form",
    "must be completed",
    "submission date",
    "approval",
    "electronic submission",
    "signature page",
)


@dataclass
class WorksetDoc:
    doc_id: str
    filename: str
    original_path: str = ""
    stored_path: str = ""
    extracted_path: str = ""
    label: str = ""
    chars: int = 0
    hash: str = ""
    has_text_layer: bool | None = None
    sections_found: dict = field(default_factory=dict)
    summary: str = ""
    summary_metrics: dict = field(default_factory=dict)
    error: str = ""


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", (name or "").strip()).strip(".-")
    return cleaned[:80] or "workset"


def _workset_path(name: str) -> Path:
    return WORKSETS_DIR / _safe_name(name)


def _manifest_path(name: str) -> Path:
    return _workset_path(name) / MANIFEST


def _load_manifest(name: str) -> dict:
    path = _manifest_path(name)
    if not path.exists():
        raise FileNotFoundError(f"No workset named '{name}'.")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["docs"] = [WorksetDoc(**doc) for doc in data.get("docs", [])]
    return data


def _save_manifest(data: dict):
    path = _manifest_path(data["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(data)
    serializable["docs"] = [asdict(doc) for doc in data.get("docs", [])]
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _public_manifest(data: dict) -> dict:
    docs = data.get("docs", [])
    return {
        "name": data["name"],
        "description": data.get("description", ""),
        "doc_count": len(docs),
        "documents": [
            {
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "label": doc.label,
                "chars": doc.chars,
                "sections_found": doc.sections_found,
                "has_summary": bool(doc.summary),
                "error": doc.error,
            }
            for doc in docs
        ],
    }


def create_workset(name: str, description: str = "") -> dict:
    safe = _safe_name(name)
    root = _workset_path(safe)
    (root / FILES_DIR).mkdir(parents=True, exist_ok=True)
    (root / EXTRACTED_DIR).mkdir(parents=True, exist_ok=True)
    if _manifest_path(safe).exists():
        data = _load_manifest(safe)
        if description.strip() and description.strip() != data.get("description"):
            data["description"] = description.strip()
            _save_manifest(data)
    else:
        data = {"name": safe, "description": description.strip(), "docs": []}
        _save_manifest(data)
    return _public_manifest(data)


def list_worksets() -> list[dict]:
    if not WORKSETS_DIR.exists():
        return []
    rows = []
    for manifest in sorted(WORKSETS_DIR.glob(f"*/{MANIFEST}")):
        try:
            rows.append(_public_manifest(_load_manifest(manifest.parent.name)))
        except Exception:
            continue
    return rows


def _next_doc_id(docs: list[WorksetDoc]) -> str:
    return f"D{len(docs) + 1}"


def _unique_dest(root: Path, filename: str) -> Path:
    safe = _safe_name(Path(filename).name)
    dest = root / FILES_DIR / safe
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    i = 2
    while True:
        candidate = dest.with_name(f"{stem}-{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def _has_text_layer(pdf_path: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["pdffonts", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return len(result.stdout.strip().splitlines()) > 2


def _extract_pdf_text(pdf_path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        if result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.SubprocessError):
        pass
    return _load_rag().load_file(pdf_path)


def extract_text(path: Path) -> tuple[str, bool | None]:
    if path.suffix.lower() == ".pdf":
        return _extract_pdf_text(path), _has_text_layer(path)
    return _load_rag().load_file(path), None


def derive_label(path: Path, text: str) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            meta = PdfReader(str(path)).metadata or {}
            title = (meta.title or "").strip()
            author = (meta.author or "").strip()
            if title and len(title) > 5 and "http" not in title.lower():
                return title + (f" ({author})" if author else "")
        except Exception:
            pass

    lines = text.splitlines()[:100]
    for line in lines:
        match = re.match(r"(?:Thesis\s+)?Title\s*:\s*(.+)", line.strip(), re.I)
        if match and len(match.group(1)) > 10:
            return match.group(1).strip()[:180]

    caps_block = []
    for line in lines:
        clean = line.strip()
        if len(clean) > 8 and clean.isupper():
            caps_block.append(clean)
        elif caps_block:
            break
    if caps_block:
        return " ".join(caps_block)[:180]

    for line in lines:
        clean = " ".join(line.strip().split())
        if len(clean) > 20 and not any(m in clean.lower() for m in _BOILERPLATE_MARKERS):
            return clean[:180]

    return path.stem.replace("_", " ").replace("-", " ").strip() or "Untitled document"


def find_key_sections(text: str) -> dict:
    hits = {}
    for i, line in enumerate(text.splitlines()):
        clean = re.sub(r"[^a-z ]+", "", line.strip().lower())
        clean = " ".join(clean.split())
        for keyword in SECTION_KEYWORDS:
            if clean == keyword or clean.startswith(keyword + " "):
                hits.setdefault(keyword, i)
    return hits


def ingest_file(workset: str, file_path: str) -> dict:
    data = create_workset(workset)
    data = _load_manifest(data["name"])
    source = Path(file_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"No such file: {source}")
    root = _workset_path(data["name"])
    dest = _unique_dest(root, source.name)
    shutil.copy2(source, dest)
    text, has_text_layer = extract_text(dest)
    doc = _record_document(
        data, dest, text, original_path=str(source), has_text_layer=has_text_layer)
    _save_manifest(data)
    return asdict(doc)


def ingest_text(workset: str, filename: str, content: str) -> dict:
    data = create_workset(workset)
    data = _load_manifest(data["name"])
    root = _workset_path(data["name"])
    name = filename if Path(filename).suffix else f"{filename}.md"
    dest = _unique_dest(root, name)
    dest.write_text(content, encoding="utf-8")
    doc = _record_document(data, dest, content, original_path="", has_text_layer=None)
    _save_manifest(data)
    return asdict(doc)


def _record_document(data: dict, dest: Path, text: str, original_path: str,
                     has_text_layer: bool | None) -> WorksetDoc:
    root = _workset_path(data["name"])
    docs = data["docs"]
    doc_id = _next_doc_id(docs)
    extracted = root / EXTRACTED_DIR / f"{doc_id}.txt"
    extracted.write_text(text or "", encoding="utf-8")
    doc = WorksetDoc(
        doc_id=doc_id,
        filename=dest.name,
        original_path=original_path,
        stored_path=str(dest),
        extracted_path=str(extracted),
        label=derive_label(dest, text or ""),
        chars=len(text or ""),
        hash=_file_hash(dest),
        has_text_layer=has_text_layer,
        sections_found=find_key_sections(text or ""),
        error="" if (text or "").strip() else "No text extracted.",
    )
    docs.append(doc)
    return doc


def _text_windows(text: str, max_chars: int = 22000, overlap: int = 1200) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    windows = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        windows.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return windows


def summarize_document_text(label: str, text: str, model: str = None,
                            max_chars: int = 22000) -> tuple[str, dict]:
    from writer import ask_ollama_long

    model = model or ASK_MODEL
    windows = _text_windows(text, max_chars=max_chars)
    system = (
        "You summarize source documents for later cross-document synthesis. "
        "Use only the supplied text. Preserve concrete findings, claims, "
        "methods, populations, dates, legal holdings, and limitations. Say "
        "when text appears incomplete or garbled."
    )
    if len(windows) == 1:
        return ask_ollama_long(
            f"Document: {label}\n\nText:\n\n{windows[0]}\n\nWrite a structured summary.",
            model,
            system,
            num_ctx=32768,
            num_predict=1000,
            temperature=0.25,
            think=False,
            echo=False,
        )

    partials = []
    metrics = {"parts": len(windows)}
    for i, window in enumerate(windows, start=1):
        summary, part_metrics = ask_ollama_long(
            f"Document: {label}\nPart {i} of {len(windows)}\n\nText:\n\n{window}\n\n"
            "Summarize this part for later synthesis.",
            model,
            system,
            num_ctx=32768,
            num_predict=800,
            temperature=0.25,
            think=False,
            echo=False,
        )
        partials.append(f"Part {i}: {summary}")
        metrics[f"part_{i}"] = part_metrics

    combined, combined_metrics = ask_ollama_long(
        f"Document: {label}\n\nPart summaries:\n\n" + "\n\n".join(partials)
        + "\n\nCombine these into one structured document summary.",
        model,
        system,
        num_ctx=32768,
        num_predict=1200,
        temperature=0.25,
        think=False,
        echo=False,
    )
    metrics["combine"] = combined_metrics
    return combined, metrics


def summarize_workset(name: str, model: str = None, force: bool = False) -> dict:
    data = _load_manifest(name)
    for doc in data["docs"]:
        if doc.summary and not force:
            continue
        text = Path(doc.extracted_path).read_text(encoding="utf-8")
        if not text.strip():
            doc.error = doc.error or "No text extracted."
            continue
        try:
            doc.summary, doc.summary_metrics = summarize_document_text(
                doc.label or doc.filename, text, model=model)
            doc.error = ""
        except Exception as e:
            doc.error = f"Summary failed: {e}"
    _save_manifest(data)
    return _public_manifest(data)


def synthesize_workset(name: str, question: str = "", model: str = None) -> dict:
    from writer import ask_ollama_long

    data = _load_manifest(name)
    if not data["docs"]:
        raise ValueError(f"Workset '{name}' has no documents.")
    if any(not doc.summary and not doc.error for doc in data["docs"]):
        summarize_workset(name, model=model)
        data = _load_manifest(name)

    briefs = []
    for doc in data["docs"]:
        status = f"Error: {doc.error}" if doc.error else doc.summary
        sections = ", ".join(doc.sections_found.keys()) or "none detected"
        briefs.append(
            f"[{doc.doc_id}] {doc.label or doc.filename}\n"
            f"File: {doc.filename}\n"
            f"Characters extracted: {doc.chars}\n"
            f"Sections detected: {sections}\n"
            f"Summary:\n{status}"
        )

    prompt = (
        "Synthesize across this attachment workset. Cite document IDs like [D1] "
        "for claims about a source. Compare the documents directly: shared "
        "themes, disagreements or tensions, unique contributions, gaps, and "
        "practical next steps. Do not invent sources or claims.\n\n"
    )
    if question.strip():
        prompt += f"User synthesis question:\n{question.strip()}\n\n"
    prompt += "Document summaries:\n\n" + "\n\n---\n\n".join(briefs)

    synthesis, metrics = ask_ollama_long(
        prompt,
        model or ASK_MODEL,
        "You are a careful cross-document synthesis analyst.",
        num_ctx=32768,
        num_predict=1800,
        temperature=0.25,
        think=False,
        echo=False,
    )
    return {
        "name": data["name"],
        "question": question,
        "synthesis": synthesis,
        "metrics": metrics,
        "documents": [
            {"doc_id": doc.doc_id, "filename": doc.filename, "label": doc.label}
            for doc in data["docs"]
        ],
    }
