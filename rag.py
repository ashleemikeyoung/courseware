import os
import sys
import json
import hashlib
import math
import re
import subprocess
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Redirect all stdout prints to stderr so MCP stdio channel stays clean
# ---------------------------------------------------------------------------
import builtins
_original_print = builtins.print
def _stderr_print(*args, **kwargs):
    kwargs['file'] = sys.stderr
    _original_print(*args, **kwargs)
builtins.print = _stderr_print

# BASE_DIR, the .env load, OLLAMA_URL, and VISION_MODEL all come from
# config.py now -- see that module's docstring for why (three independent
# .env loads across rag.py/writer.py/orchestrator.py used to exist, one of
# them fragile). Anything path- or model-config-shaped belongs there, not
# redeclared here.
from config import BASE_DIR, OLLAMA_URL, VISION_MODEL

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# Safe at module level: projects.py has no top-level import of rag.py (its
# one dependency on SUPPORTED_EXTENSIONS, in stats(), is a local import
# specifically to avoid a cycle), so this direction is the only one that
# needs to exist and there's nothing to import in a loop with.
import projects

sys.path.insert(0, str(BASE_DIR / "memory"))
try:
    from memory_client import (
        get_synopsis,
        record_synopsis,
        record_document_upload_profile,
        search_citations,
        search_synopses,
    )
    MEMORY_AVAILABLE = True
except Exception as e:
    get_synopsis = None
    record_synopsis = None
    record_document_upload_profile = None
    search_citations = None
    search_synopses = None
    MEMORY_AVAILABLE = False
    print(f"  [Warning] memory-db document registry unavailable: {e}")

# Optional imports
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("  [Warning] python-docx not installed. .docx files will be skipped.")

try:
    import openpyxl
    XLSX_AVAILABLE = True
except ImportError:
    XLSX_AVAILABLE = False
    print("  [Warning] openpyxl not installed. .xlsx files will be skipped.")

try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False
    print("  [Warning] python-pptx not installed. .pptx files will be skipped.")

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("  [Warning] pytesseract or Pillow not installed. Image OCR will be skipped.")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("  [Warning] pdf2image not installed. Image-based PDF OCR will be skipped.")
    print("  Install with: pip install pdf2image && brew install poppler")

try:
    import rawpy
    RAW_AVAILABLE = True
except ImportError:
    RAW_AVAILABLE = False
    print("  [Warning] rawpy not installed. RAW camera files will be skipped.")

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "30"))

# Documents folder — absolute, anchored to BASE_DIR
_docs_env = os.getenv("DOCUMENTS_FOLDER")
if _docs_env:
    DOCUMENTS_FOLDER = str(Path(_docs_env).resolve())
else:
    DOCUMENTS_FOLDER = str(BASE_DIR / "documents")

# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = set(filter(None, [
    ".txt", ".md",
    ".pdf",
    ".docx" if DOCX_AVAILABLE else None,
    ".xlsx" if XLSX_AVAILABLE else None,
    ".pptx" if PPTX_AVAILABLE else None,
    ".jpg", ".jpeg", ".png", ".gif", ".tiff", ".bmp",
    ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2",
    ".dng" if RAW_AVAILABLE else None,
]))

# ---------------------------------------------------------------------------
# Embedding model and database — all paths absolute
# ---------------------------------------------------------------------------

print("Loading embedding model...")
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    local_files_only=True,
)

CHROMA_PATH = str(BASE_DIR / "chroma_db")
db_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = db_client.get_or_create_collection("my_documents")

print(f"Database path:   {CHROMA_PATH}")
print(f"Documents path:  {DOCUMENTS_FOLDER}")
print(f"Chunks in index: {collection.count()}")

# ---------------------------------------------------------------------------
# Vision model
# ---------------------------------------------------------------------------

def describe_image_with_vision(file: Path) -> str:
    try:
        import base64
        with open(file, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": VISION_MODEL,
                "prompt": (
                    "Describe this image in detail. "
                    "Extract any visible text. "
                    "Note any charts, diagrams, tables, or structured data. "
                    "Be thorough so the description can be searched later."
                ),
                "images": [image_data],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        description = response.json().get("response", "").strip()
        if description:
            return f"[Vision description of {file.name}]\n{description}"
        return ""
    except Exception as e:
        print(f"  Warning: vision model failed for {file.name}: {e}")
        return ""

# ---------------------------------------------------------------------------
# RAW camera metadata
# ---------------------------------------------------------------------------

def extract_raw_metadata(file: Path) -> str:
    if not RAW_AVAILABLE:
        return ""
    try:
        metadata_parts = [f"RAW camera file: {file.name}"]
        with rawpy.imread(str(file)) as raw:
            metadata_parts.append(f"Image size: {raw.sizes.width}x{raw.sizes.height}")
            metadata_parts.append(f"Raw type: {raw.raw_type}")

        try:
            result = subprocess.run(
                ["exiftool", "-json", str(file)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                exif_data = json.loads(result.stdout)[0]
                interesting_fields = [
                    "Make", "Model", "LensModel", "FocalLength",
                    "Aperture", "ShutterSpeed", "ISO", "DateTimeOriginal",
                    "GPSLatitude", "GPSLongitude", "GPSAltitude",
                    "ImageDescription", "Artist", "Copyright",
                ]
                for field in interesting_fields:
                    if field in exif_data:
                        metadata_parts.append(f"{field}: {exif_data[field]}")
        except FileNotFoundError:
            print("  [Note] exiftool not found. Install with: brew install exiftool")
        except Exception:
            pass

        return "\n".join(metadata_parts)
    except Exception as e:
        print(f"  Warning: could not process RAW file {file.name}: {e}")
        return ""

# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_file(file: Path) -> str:
    suffix = file.suffix.lower()

    if suffix in [".txt", ".md"]:
        return file.read_text(encoding="utf-8")

    elif suffix == ".pdf":
        try:
            # First try pypdf for text-layer PDFs
            reader = PdfReader(str(file))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

            # If pypdf got nothing, PDF is likely image-based — try OCR
            if not text.strip():
                print(f"  No text layer in {file.name}, trying OCR...")
                if PDF2IMAGE_AVAILABLE and OCR_AVAILABLE:
                    try:
                        images = convert_from_path(str(file))
                        ocr_parts = []
                        for i, image in enumerate(images):
                            page_text = pytesseract.image_to_string(image).strip()
                            if page_text:
                                ocr_parts.append(f"[Page {i+1}]\n{page_text}")
                        text = "\n\n".join(ocr_parts)
                        if text:
                            print(f"  OCR extracted {len(text)} chars from {file.name}")
                        else:
                            print(f"  OCR found no text in {file.name}")
                    except Exception as e:
                        print(f"  OCR failed for {file.name}: {e}")
                elif not PDF2IMAGE_AVAILABLE:
                    print(f"  pdf2image not installed. Run: pip install pdf2image && brew install poppler")
                elif not OCR_AVAILABLE:
                    print(f"  pytesseract not available for OCR fallback")

            return text
        except Exception as e:
            print(f"  Warning: could not read PDF {file.name}: {e}")
            return ""

    elif suffix in [".docx", ".doc"]:
        if not DOCX_AVAILABLE:
            print(f"  Skipping {file.name}: python-docx not installed")
            return ""
        try:
            doc = DocxDocument(str(file))
            parts = []
            for para in doc.paragraphs:
                if para.text.strip():
                    parts.append(para.text)
            for table in doc.tables:
                for row in table.rows:
                    row_text = "\t".join(
                        cell.text for cell in row.cells if cell.text.strip()
                    )
                    if row_text:
                        parts.append(row_text)
            return "\n".join(parts)
        except Exception as e:
            print(f"  Warning: could not read Word file {file.name}: {e}")
            return ""

    elif suffix in [".xlsx", ".xls"]:
        if not XLSX_AVAILABLE:
            print(f"  Skipping {file.name}: openpyxl not installed")
            return ""
        try:
            wb = openpyxl.load_workbook(str(file), data_only=True)
            parts = []
            for sheet in wb.worksheets:
                parts.append(f"Sheet: {sheet.title}")
                for row in sheet.iter_rows(values_only=True):
                    row_text = "\t".join(
                        str(cell) for cell in row if cell is not None
                    )
                    if row_text.strip():
                        parts.append(row_text)
            return "\n".join(parts)
        except Exception as e:
            print(f"  Warning: could not read Excel file {file.name}: {e}")
            return ""

    elif suffix in [".pptx", ".ppt"]:
        if not PPTX_AVAILABLE:
            print(f"  Skipping {file.name}: python-pptx not installed")
            return ""
        try:
            prs = Presentation(str(file))
            parts = []
            for i, slide in enumerate(prs.slides):
                parts.append(f"Slide {i + 1}:")
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        parts.append(shape.text)
                    if shape.has_table:
                        for row in shape.table.rows:
                            row_text = "\t".join(
                                cell.text for cell in row.cells if cell.text.strip()
                            )
                            if row_text:
                                parts.append(row_text)
                if slide.has_notes_slide:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        parts.append(f"Notes: {notes}")
            return "\n".join(parts)
        except Exception as e:
            print(f"  Warning: could not read PowerPoint file {file.name}: {e}")
            return ""

    elif suffix in [".jpg", ".jpeg", ".png", ".gif", ".tiff", ".bmp"]:
        if OCR_AVAILABLE:
            try:
                image = Image.open(str(file))
                ocr_text = pytesseract.image_to_string(image).strip()
                if ocr_text:
                    return f"[OCR extracted from {file.name}]\n{ocr_text}"
            except Exception as e:
                print(f"  Warning: OCR failed for {file.name}: {e}")
        print(f"  Using vision model for {file.name}...")
        return describe_image_with_vision(file)

    elif suffix in [".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".dng"]:
        return extract_raw_metadata(file)

    return ""

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return [c for c in chunks if c.strip()]

# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def file_hash(file: Path) -> str:
    return hashlib.md5(file.read_bytes()).hexdigest()


def get_indexed_sources() -> dict:
    if collection.count() == 0:
        return {}
    results = collection.get(include=["metadatas"])
    indexed = {}
    for meta in results["metadatas"]:
        source = meta.get("source")
        hash_val = meta.get("file_hash")
        if source and hash_val:
            indexed[source] = hash_val
    return indexed


def remove_source(filename: str):
    results = collection.get(include=["metadatas"])
    ids_to_delete = [
        results["ids"][i]
        for i, meta in enumerate(results["metadatas"])
        if meta.get("source") == filename
    ]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(f"  Removed {len(ids_to_delete)} chunks for {filename}")


_IGNORED_DIRS = {".git", ".obsidian", "__pycache__", "node_modules",
                 ".venv", "venv", ".trash", ".writer"}


def _project_of(rel_path: str) -> str:
    """Top folder under the documents root, or 'unfiled' for loose files."""
    parts = Path(rel_path).parts
    return parts[0] if len(parts) > 1 else "unfiled"


def index_file(file: Path, current_hash: str) -> int:
    text = load_file(file)
    if not text.strip():
        print(f"  Skipping {file.name} (empty or unreadable)")
        return 0

    chunks = chunk_text(text)
    if not chunks:
        return 0

    # Source is the path relative to the documents root, so two projects can
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
    )
    record_document_profile(rel, text, project, source_hash=current_hash)
    return len(chunks)


def scan_documents(folder: str = None, verbose: bool = True) -> dict:
    folder_path = Path(folder).resolve() if folder else Path(DOCUMENTS_FOLDER)

    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        if verbose:
            print(f"  Created documents folder at {folder_path}")
        return {"new": [], "updated": [], "removed": [], "unchanged": []}

    # Walk subfolders. Each top-level folder under the documents root is a
    # project; files loose at the root belong to "unfiled". Keys are paths
    # relative to the root, which makes them unique across projects.
    current_files = {
        str(f.relative_to(folder_path)): file_hash(f)
        for f in folder_path.rglob("*")
        if f.is_file()
        and f.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(part in _IGNORED_DIRS or part.startswith(".")
                    for part in f.relative_to(folder_path).parts[:-1])
    }

    indexed = get_indexed_sources()
    summary = {"new": [], "updated": [], "removed": [], "unchanged": []}

    for filename, current_hash in current_files.items():
        file = folder_path / filename  # filename is now a relative path
        if filename not in indexed:
            if verbose:
                print(f"  [+] Indexing new file: {filename}")
            count = index_file(file, current_hash)
            if verbose:
                print(f"      Added {count} chunks")
            summary["new"].append(filename)

        elif indexed[filename] != current_hash:
            if verbose:
                print(f"  [~] Re-indexing changed file: {filename}")
            remove_source(filename)
            count = index_file(file, current_hash)
            if verbose:
                print(f"      Updated with {count} chunks")
            summary["updated"].append(filename)

        else:
            if _needs_document_profile(filename):
                text = load_file(file)
                if text.strip():
                    record_document_profile(
                        filename, text, projects.project_of(filename),
                        source_hash=file_hash(file))
            summary["unchanged"].append(filename)

    for filename in indexed:
        if filename not in current_files:
            if verbose:
                print(f"  [-] Removing deleted file: {filename}")
            remove_source(filename)
            summary["removed"].append(filename)

    return summary


def ingest_content(filename: str, content: str, project: str = None) -> tuple:
    """
    Save arbitrary text content as a file under the documents folder and
    index it immediately -- the one shared way to get ad hoc content (a file
    attached in a Claude Desktop chat, a paste, anything that didn't arrive
    as a file already sitting in the documents folder) into the index.
    Every caller -- mcp_server.py's ingest_content tool today, potentially
    app.py or the CLI later -- goes through this, so "how does content get
    added to the index" only has one answer.

    Content is assumed to already be plain text: callers starting from a
    PDF/DOCX/etc. should extract the text themselves first (Claude Desktop
    already does this for attachments). It's always saved as .md regardless
    of the original filename's extension -- a "report.pdf" full of plain
    text would just fail load_file()'s PDF branch on the next rescan
    otherwise.

    project uses projects.safe() for sanitization, the same rule app.py and
    writer.py apply to a project name arriving from outside the process
    (HTTP body, CLI arg, tool call) -- so a stray "../" or odd character in
    a caller-supplied project name can't do anything unexpected on disk.

    Returns (relative_path, chunk_count).
    """
    proj = projects.safe(project) if project else projects.UNFILED
    safe_stem = Path(filename).stem.strip() or "untitled"

    dest_dir = Path(DOCUMENTS_FOLDER) / proj
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{safe_stem}.md"

    dest_path.write_text(content, encoding="utf-8")

    rel = str(dest_path.resolve().relative_to(Path(DOCUMENTS_FOLDER).resolve()))

    # Same re-index pattern as scan_documents()'s "updated" branch: if this
    # exact path was already indexed (re-ingesting the same filename), drop
    # its old chunks first so they don't linger alongside the new ones.
    if rel in get_indexed_sources():
        remove_source(rel)

    new_hash = file_hash(dest_path)
    chunk_count = index_file(dest_path, new_hash)
    return rel, chunk_count

# ---------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------

def _first_meaningful_lines(text: str, limit: int = 10) -> list:
    lines = []
    for line in text.splitlines():
        clean = " ".join(line.strip().split())
        if clean:
            lines.append(clean)
        if len(lines) >= limit:
            break
    return lines



def _derive_document_label(source: str, text: str) -> str:
    lines = text.splitlines()[:100]
    for line in lines:
        match = re.match(r"(?:Thesis\s+)?Title\s*:\s*(.+)", line.strip(), re.I)
        if match and len(match.group(1)) > 10:
            return match.group(1).strip()[:180]
    for line in lines:
        clean = " ".join(line.strip().split())
        if len(clean) > 20:
            return clean[:180]
    return Path(source).stem.replace("_", " ").replace("-", " ").strip()


def _document_sections(text: str) -> dict:
    keywords = [
        "abstract", "introduction", "literature review", "method", "methods",
        "methodology", "results", "findings", "discussion", "conclusion",
        "recommendations", "implications", "limitations", "references",
    ]
    hits = {}
    for i, line in enumerate(text.splitlines()):
        clean = re.sub(r"[^a-z ]+", "", line.strip().lower())
        clean = " ".join(clean.split())
        for keyword in keywords:
            if clean == keyword or clean.startswith(keyword + " "):
                hits.setdefault(keyword, i)
    return hits


def _document_genres(source: str, text: str) -> list:
    lower = f"{source}\n{text[:18000]}".lower()
    suffix = Path(source).suffix.lower()
    filename = Path(source).name.lower()
    sections = set(_document_sections(text))
    genres = []

    def add(name: str):
        if name not in genres:
            genres.append(name)

    # Container/export formats first. These may discuss many topics, but the
    # file itself is not an article, legal filing, or interview protocol.
    if any(term in lower[:4000] for term in [
        "chat history", "conversation export", "conversation with claude",
        "working session", "record of a research and writing session",
    ]):
        add("chat export")
        return genres

    if "sage research methods" in lower[:4000]:
        add("research methods guide")
        if "doi:" in lower[:4000] or "online isbn" in lower[:4000]:
            add("book chapter")
        return genres

    if "dissertation template" in filename or "insert your dissertation title here" in lower[:4000]:
        add("dissertation template")
        return genres

    coursework_terms = [
        "topic4 dq", "topic5 dq", "topic6 dq", "topic7 dq", " dq1", " dq2",
        "summary of the problem space", "population to be studied",
        "variables (excluding demographics)", "discussion question",
    ]
    if any(term in lower[:6000] or term in filename for term in coursework_terms):
        if "problem space" in lower[:6000] or "dissertation" in lower[:6000]:
            add("dissertation draft")
        else:
            add("coursework")
        return genres

    if suffix == ".pptx" or "slide 1:" in lower or "speaker notes" in lower:
        add("presentation")
        return genres

    if suffix in {".xlsx", ".xls"} or lower.startswith("sheet:"):
        add("spreadsheet")
        return genres

    legal_filing_terms = [
        "plaintiff", "defendant", "case no", "court", "pursuant to",
        "complaint", "affidavit", "judgment", "dismissal", "certificate of service",
    ]
    if sum(1 for term in legal_filing_terms if term in lower) >= 3:
        add("legal filing")

    if any(term in lower for term in [
        "settlement agreement", "quitclaim", "contract", "agreement made",
        "executed agreement",
    ]):
        add("contract/agreement")

    if (
        "interview protocol" in lower[:8000]
        or "interview questions" in lower[:8000]
        or ("participant" in lower[:8000] and "interview" in lower[:8000])
    ):
        add("interview protocol")

    academic_sections = {
        "abstract", "introduction", "method", "methods", "methodology",
        "results", "findings", "discussion", "conclusion", "references",
    }
    if (
        "article" in lower[:2000]
        and len(sections & academic_sections) >= 4
        and ("doi:" in lower[:5000] or "journal" in lower[:5000] or "keywords" in lower[:5000])
    ):
        add("academic article")

    if (
        "literature review" in sections
        or "systematic review" in lower[:8000]
        or "scoping review" in lower[:8000]
        or "review of the literature" in lower[:8000]
    ):
        add("literature review")

    if suffix in {".md", ".txt"} or "meeting notes" in lower[:4000]:
        add("notes")

    return genres or ["document"]

def _document_themes(text: str) -> list:
    lower = text.lower()
    themes = []

    if any(term in lower for term in [
        "ai adoption", "adoption of ai", "artificial intelligence adoption",
        "generative ai adoption", "adopt generative ai", "ai usage",
    ]):
        themes.append("ai adoption")

    if (
        any(term in lower for term in ["training", "ease of use", "perceived usefulness"])
        and any(term in lower for term in ["ai", "artificial intelligence", "technology", "system"])
    ):
        themes.append("training and usability")

    if any(term in lower for term in [
        "attorney-client privilege", "attorney client privilege",
        "work-product privilege", "work product doctrine", "work product privilege",
        "client confidentiality", "legal privilege", "privileged communication",
    ]):
        themes.append("legal privilege")

    if any(term in lower for term in [
        "methodology", "qualitative", "quantitative", "research design",
        "interview protocol", "data collection", "sample size",
    ]):
        themes.append("research methods")

    if any(term in lower for term in [
        "risk governance", "ai governance", "compliance", "legal ethics",
        "confidentiality", "privacy risk", "ethical risk", "risk management",
    ]):
        themes.append("risk and governance")

    return themes


def _document_profile(source: str, text: str, project: str) -> str:
    """
    Cheap document-level profile for memory.documents.

    This is deliberately not an Ollama summary. Indexing should stay fast,
    deterministic, and usable while Ollama is stopped. The profile gives the
    retrieval layer document-level hooks -- filename, project, early headings,
    and lead text -- that chunk vectors alone do not reliably preserve.
    """
    lines = _first_meaningful_lines(text)
    lead = " ".join(text.split()[:220])
    headings = [
        line for line in lines
        if len(line) <= 120 and (
            line.istitle()
            or line.isupper()
            or line.lower().startswith(("abstract", "introduction", "summary"))
        )
    ][:6]
    parts = [
        f"source: {source}",
        f"filename: {Path(source).name}",
        f"project: {project}",
    ]
    if headings:
        parts.append("headings: " + " | ".join(headings))
    if lines:
        parts.append("opening: " + " | ".join(lines[:4]))
    if lead:
        parts.append("lead: " + lead)
    return "\n".join(parts)

def record_document_profile(source: str, text: str, project: str,
                            source_hash: str = None):
    if not MEMORY_AVAILABLE or record_synopsis is None:
        return
    try:
        synopsis = _document_profile(source, text, project)
        if record_document_upload_profile is not None:
            record_document_upload_profile(
                source,
                synopsis,
                word_count=len(text.split()),
                model="local-profile-v1",
                source_hash=source_hash,
                chars=len(text),
                file_type=Path(source).suffix.lower().lstrip("."),
                project=project,
                label=_derive_document_label(source, text),
                sections_found=_document_sections(text),
                genres=_document_genres(source, text),
                themes=_document_themes(text),
                upload_state="project_file",
            )
        else:
            record_synopsis(
                source,
                synopsis,
                word_count=len(text.split()),
                model="local-profile-v1",
            )
    except Exception as e:
        print(f"  [Warning] could not record document profile for {source}: {e}")


def _needs_document_profile(source: str) -> bool:
    if not MEMORY_AVAILABLE or get_synopsis is None:
        return False
    try:
        row = get_synopsis(source)
    except Exception:
        return False
    return not row or row.get("model") != "local-profile-v1"


def backfill_document_profiles(project: str = None, overwrite: bool = False) -> dict:
    """
    Populate memory.documents for files already indexed before document
    profiles existed. This reads source files from disk and records the same
    cheap local profile index_file() records for new/changed files.
    """
    summary = {"profiled": [], "skipped": [], "failed": []}
    if not MEMORY_AVAILABLE or record_synopsis is None:
        summary["failed"].append({
            "source": "*",
            "error": "memory document registry is unavailable",
        })
        return summary

    for source in sorted(get_indexed_sources()):
        if project and projects.project_of(source) != project:
            continue
        try:
            if (not overwrite and get_synopsis is not None
                    and not _needs_document_profile(source)):
                summary["skipped"].append(source)
                continue
            path = Path(DOCUMENTS_FOLDER) / source
            text = load_file(path)
            if not text.strip():
                summary["failed"].append({
                    "source": source,
                    "error": "empty or unreadable",
                })
                continue
            record_document_profile(
                source, text, projects.project_of(source), source_hash=file_hash(path))
            summary["profiled"].append(source)
        except Exception as e:
            summary["failed"].append({"source": source, "error": str(e)})
    return summary


# ---------------------------------------------------------------------------
# Search — hybrid document registry + lexical + semantic retrieval
# ---------------------------------------------------------------------------

def _semantic_search(question: str, n_results: int = 5, project: str = None):
    """Pure semantic vector search, optionally scoped to one project."""
    question_embedding = embedder.encode([question])[0]
    kwargs = {
        "query_embeddings": [question_embedding.tolist()],
        "n_results": min(n_results, collection.count()),
    }
    if project:
        kwargs["where"] = {"project": project}
    results = collection.query(**kwargs)
    return results


def _semantic_candidates(question: str, n_results: int, project: str = None):
    if collection.count() == 0:
        return []
    results = _semantic_search(question, n_results=n_results, project=project)
    out = []
    for rank, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        out.append({
            "document": doc,
            "metadata": dict(meta),
            "score": 1.0 / (rank + 1),
            "signals": {"semantic"},
        })
    return out


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at",
    "to", "for", "with", "from", "by", "as", "is", "are", "was", "were",
    "be", "been", "being", "do", "does", "did", "done", "has", "have",
    "had", "having", "not", "no", "so", "than", "then", "this", "that",
    "these", "those", "it", "its", "it's", "you", "your", "yours", "he",
    "she", "they", "we", "i", "me", "my", "him", "her", "them", "us",
    "our", "their", "who", "what", "when", "where",
    "why", "how", "which", "can", "could", "should", "would", "will",
    "shall", "about", "into", "over", "under", "again", "also", "just",
    "up", "out", "off", "all", "any", "some", "such", "own",
}


def _terms(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def _compact(text: str) -> str:
    return "".join(_terms(text))


def meaningful_words(text: str) -> list:
    """
    Words worth treating as exact-match search terms. Length alone is a bad
    filter here: short proper nouns (a first name, initials, a short case
    name) are exactly the kind of term someone searches for, and a bare
    length cutoff drops them silently. A stopword list is the right tool:
    keep every word that isn't a common English function word, regardless
    of length or case, so a name like "Tye" survives capitalized or not.

    Shared by search()'s filename/content matching and by writer.py's
    gather_evidence(), so both retrieval paths treat a question's meaningful
    terms identically rather than drifting apart over two copies of this.
    """
    words = []
    for clean in _terms(text):
        if clean and len(clean) >= 2 and clean not in _STOPWORDS:
            words.append(clean)
    return words


def _word_count_score(text: str, words: list) -> int:
    text_lower = text.lower()
    return sum(text_lower.count(w) for w in words if len(w) >= 3)


def _source_matches(source: str, words: list, question: str = "") -> int:
    source_lower = source.lower()
    filename_lower = Path(source).name.lower()
    stem_lower = Path(source).stem.lower()
    source_compact = _compact(source)
    query_compact = _compact(question)
    score = 0
    if query_compact and query_compact in source_compact:
        score += 14
    for w in words:
        if w in stem_lower or w in _compact(stem_lower):
            score += 6
        elif w in filename_lower or w in _compact(filename_lower):
            score += 4
        elif w in source_lower:
            score += 2
    return score


def _metadata_source(meta: dict) -> str:
    return meta.get("source", "")


def _candidate_key(doc: str, meta: dict) -> str:
    return f"{_metadata_source(meta)}::{doc[:120]}"


def _add_candidate(candidates: dict, doc: str, meta: dict,
                   score: float, signal: str):
    key = _candidate_key(doc, meta)
    if key not in candidates:
        candidates[key] = {
            "document": doc,
            "metadata": dict(meta),
            "score": 0.0,
            "signals": set(),
        }
    candidates[key]["score"] += score
    candidates[key]["signals"].add(signal)


def _source_chunks(source: str) -> list:
    try:
        got = collection.get(where={"source": source},
                             include=["metadatas", "documents"])
    except Exception:
        return []
    rows = []
    for doc, meta in zip(got["documents"], got["metadatas"]):
        rows.append((doc, meta))
    return rows


def _registry_candidates(words: list, project: str = None) -> list:
    if not MEMORY_AVAILABLE or search_synopses is None or not words:
        return []
    prefix = f"{project}/" if project else None
    try:
        hits = search_synopses(words, project_prefix=prefix)
    except Exception as e:
        print(f"  [Warning] document registry lookup failed: {e}")
        return []

    out = []
    for hit in hits:
        source = hit.get("source")
        synopsis = hit.get("synopsis") or ""
        if not source:
            continue
        chunks = _source_chunks(source)
        if not chunks:
            continue
        ranked = sorted(
            chunks,
            key=lambda row: _word_count_score(row[0], words),
            reverse=True,
        )
        doc, meta = ranked[0]
        doc_with_profile = f"[Document profile]\n{synopsis}\n\n[Matched passage]\n{doc}"
        out.append({
            "document": doc_with_profile,
            "metadata": dict(meta),
            "score": 3.0 + min(_word_count_score(synopsis, words), 10) / 5,
            "signals": {"document_registry"},
        })
    return out


def _citation_candidates(words: list, project: str = None) -> list:
    if not MEMORY_AVAILABLE or search_citations is None or not words:
        return []
    prefix = f"{project}/" if project else None
    try:
        hits = search_citations(words, project_prefix=prefix)
    except Exception as e:
        print(f"  [Warning] citation lookup failed: {e}")
        return []

    out = []
    seen_sources = set()
    for hit in hits:
        source = hit.get("source")
        if not source or source in seen_sources:
            continue
        seen_sources.add(source)
        chunks = _source_chunks(source)
        if not chunks:
            continue
        citation_text = (
            "[Verified citation record, from memory-db not chroma_db]\n"
            f"File: {source}\n"
            f"Title: {hit.get('title')}\n"
            f"Author(s): {hit.get('authors')}\n"
            f"Source: {hit.get('source_line')}"
        )
        ranked = sorted(
            chunks,
            key=lambda row: _word_count_score(row[0], words),
            reverse=True,
        )
        doc, meta = ranked[0]
        out.append({
            "document": f"{citation_text}\n\n[Matched passage]\n{doc}",
            "metadata": dict(meta),
            "score": 12.0 + min(_word_count_score(citation_text, words), 12) / 3,
            "signals": {"citation"},
        })
    return out


def _diversify(ranked: list, n_results: int, max_per_source: int = 2) -> list:
    selected = []
    per_source = {}
    overflow = []
    for item in ranked:
        source = item["metadata"].get("source", "")
        if per_source.get(source, 0) < max_per_source:
            selected.append(item)
            per_source[source] = per_source.get(source, 0) + 1
        else:
            overflow.append(item)
        if len(selected) >= n_results:
            return selected

    for item in overflow:
        selected.append(item)
        if len(selected) >= n_results:
            break
    return selected


def retrieve(question: str, n_results: int = 5, project: str = None) -> list:
    """
    Hybrid retrieval over document-level memory, lexical Chroma content, and
    semantic Chroma vectors. This is the new internal API; search() below
    adapts it back to the long-standing Chroma-like response shape used by
    MCP, Claude Desktop, Codex, summarize.py, and orchestrator.py.
    """
    if collection.count() == 0:
        return []

    words = meaningful_words(question)
    candidates = {}
    all_data = collection.get(include=["metadatas", "documents"])

    for item in _citation_candidates(words, project=project):
        _add_candidate(
            candidates,
            item["document"],
            item["metadata"],
            item["score"],
            "citation",
        )

    for item in _registry_candidates(words, project=project):
        _add_candidate(
            candidates,
            item["document"],
            item["metadata"],
            item["score"],
            "document_registry",
        )

    for i, meta in enumerate(all_data["metadatas"]):
        if project and meta.get("project") != project:
            continue
        doc = all_data["documents"][i]
        source = meta.get("source", "")
        source_score = _source_matches(source, words, question=question)
        content_score = _word_count_score(doc, words)
        if source_score:
            _add_candidate(candidates, doc, meta, 2.0 + source_score / 4, "source")
        if content_score:
            _add_candidate(candidates, doc, meta,
                           1.0 + math.log1p(content_score), "keyword")

    semantic_limit = max(n_results * 3, 12)
    for item in _semantic_candidates(question, semantic_limit, project=project):
        _add_candidate(
            candidates,
            item["document"],
            item["metadata"],
            item["score"],
            "semantic",
        )

    ranked = sorted(
        candidates.values(),
        key=lambda c: (
            c["score"] + 0.35 * max(0, len(c["signals"]) - 1),
            "citation" in c["signals"],
            "document_registry" in c["signals"],
            "source" in c["signals"],
        ),
        reverse=True,
    )
    selected = _diversify(ranked, n_results=n_results)
    for item in selected:
        item["metadata"]["retrieval_signals"] = ",".join(sorted(item["signals"]))
        item["metadata"]["retrieval_score"] = round(item["score"], 4)
    return selected


def search(question: str, n_results: int = 5, project: str = None):
    """
    Compatibility wrapper around retrieve().

    The return shape intentionally matches Chroma's collection.query() shape
    because mcp_server.py, orchestrator.py, summarize.py, Claude Desktop, and
    Codex already depend on it.
    """
    if collection.count() == 0:
        return {"documents": [[]], "metadatas": [[]]}
    results = retrieve(question, n_results=n_results, project=project)
    return {
        "documents": [[r["document"] for r in results]],
        "metadatas": [[r["metadata"] for r in results]],
    }


# ---------------------------------------------------------------------------
# Entry point for standalone use
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nInitial document scan...")
    summary = scan_documents(verbose=True)
    print(
        f"\nReady: {len(summary['new'])} new, "
        f"{len(summary['updated'])} updated, "
        f"{len(summary['removed'])} removed, "
        f"{len(summary['unchanged'])} unchanged"
    )
    print(f"Total chunks in database: {collection.count()}")

    print("\nSearch your documents (or type quit):\n")
    while True:
        q = input("Search: ").strip()
        if not q:
            continue
        if q.lower() in ["quit", "exit"]:
            break
        results = search(q)
        for i, doc in enumerate(results["documents"][0]):
            source = results["metadatas"][0][i]["source"]
            print(f"\n--- Match {i+1} (from {source}) ---")
            print(doc)
