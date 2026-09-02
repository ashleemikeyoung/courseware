import os
import sys
import json
import hashlib
import math
import re
import subprocess
import time
import requests
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
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
        get_search_criteria,
        get_setting,
        ensure_storage_root,
        start_file_scan,
        record_file_observation,
        mark_missing_storage_objects,
        finish_file_scan,
        link_document_file_identity,
    )
    MEMORY_AVAILABLE = True
except Exception as e:
    get_synopsis = None
    record_synopsis = None
    record_document_upload_profile = None
    search_citations = None
    search_synopses = None
    get_search_criteria = None
    get_setting = None
    ensure_storage_root = None
    start_file_scan = None
    record_file_observation = None
    mark_missing_storage_objects = None
    finish_file_scan = None
    link_document_file_identity = None
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
EMAIL_SOURCE_PREFIX = "mail/"

# Documents folder — absolute, anchored to BASE_DIR
_docs_env = os.getenv("DOCUMENTS_FOLDER")
if _docs_env:
    DOCUMENTS_FOLDER = str(Path(_docs_env).resolve())
else:
    DOCUMENTS_FOLDER = str(BASE_DIR / "documents")

DEFAULT_SUPPORTED_EXTENSIONS = set(filter(None, [
    ".txt", ".md",
    ".pdf",
    ".docx" if DOCX_AVAILABLE else None,
    ".xlsx" if XLSX_AVAILABLE else None,
    ".pptx" if PPTX_AVAILABLE else None,
    ".jpg", ".jpeg", ".png", ".gif", ".tiff", ".bmp",
    ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2",
    ".dng" if RAW_AVAILABLE else None,
]))
DEFAULT_IGNORED_DIRS = {
    ".git", ".obsidian", "__pycache__", "node_modules",
    ".venv", "venv", ".trash", ".writer",
}


def _csv_setting(key: str, default_values: set) -> set:
    if not MEMORY_AVAILABLE or get_setting is None:
        return set(default_values)
    try:
        raw = get_setting(key, ",".join(sorted(default_values)))
    except Exception as e:
        print(f"  [Warning] memory setting {key} unavailable: {e}")
        return set(default_values)
    values = {
        item.strip().lower()
        for item in (raw or "").replace("\n", ",").split(",")
        if item.strip()
    }
    return values or set(default_values)


def supported_extensions() -> set:
    return {
        ext if ext.startswith(".") else f".{ext}"
        for ext in _csv_setting("rag_supported_extensions", DEFAULT_SUPPORTED_EXTENSIONS)
    }


def ignored_dirs() -> set:
    return _csv_setting("rag_ignored_dirs", DEFAULT_IGNORED_DIRS)


def email_maildir_roots() -> list:
    """
    Maildirs produced by OfflineIMAP/offlineimap3. Configure with
    IMAP_MAILDIR_ROOTS=/path/to/Maildir,/path/to/another/Maildir.
    """
    raw = os.getenv("IMAP_MAILDIR_ROOTS", "")
    roots = []
    for item in raw.replace("\n", ",").split(","):
        value = item.strip()
        if value:
            roots.append(Path(value).expanduser().resolve())
    return roots


# Backward-compatible export for older call sites. Active scans use the
# functions above so Settings changes apply without editing code.
SUPPORTED_EXTENSIONS = DEFAULT_SUPPORTED_EXTENSIONS

# ---------------------------------------------------------------------------
# Embedding model and vector database — all paths absolute
# ---------------------------------------------------------------------------

class LanceCollection:
    """
    Small Chroma-shaped compatibility wrapper over LanceDB.

    The rest of the app still expects collection.count/get/add/delete/query.
    Keeping those methods here lets us change storage without rewriting every
    retrieval path at once.
    """
    INDEX_MIN_ROWS = 256

    def __init__(self, path: str, name: str, dimension: int):
        try:
            import lancedb
            import pyarrow as pa
        except ImportError as e:
            raise RuntimeError(
                "LanceDB is required for this RAG index. Install it in the "
                "rag environment with: pip install lancedb"
            ) from e

        self.path = path
        self.name = name
        self.dimension = dimension
        self.db = lancedb.connect(path)

        if name in self.db.table_names():
            self.table = self.db.open_table(name)
        else:
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dimension)),
                pa.field("document", pa.string()),
                pa.field("source", pa.string()),
                pa.field("project", pa.string()),
                pa.field("filename", pa.string()),
                pa.field("file_hash", pa.string()),
                pa.field("chunk_index", pa.int64()),
                pa.field("chunk_count", pa.int64()),
                pa.field("word_count", pa.int64()),
                pa.field("char_count", pa.int64()),
                pa.field("source_ext", pa.string()),
                pa.field("source_stem", pa.string()),
                pa.field("metadata_json", pa.string()),
            ])
            self.table = self.db.create_table(name, schema=schema)
        self._indexes_dirty = False

    def count(self) -> int:
        return self.table.count_rows()

    def _quote(self, value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    def _where_expr(self, where: dict = None) -> str:
        if not where:
            return ""
        parts = []
        for key, value in where.items():
            if key not in {
                "id", "source", "project", "filename", "file_hash",
                "chunk_index", "source_ext", "source_stem",
            }:
                continue
            if isinstance(value, (int, float)):
                parts.append(f"{key} = {value}")
            else:
                parts.append(f"{key} = {self._quote(value)}")
        return " AND ".join(parts)

    def _metadata_from_row(self, row: dict) -> dict:
        try:
            meta = json.loads(row.get("metadata_json") or "{}")
        except Exception:
            meta = {}
        for key in (
            "source", "project", "filename", "file_hash", "chunk_index",
            "chunk_count", "word_count", "char_count", "source_ext", "source_stem",
        ):
            if row.get(key) is not None:
                meta[key] = row.get(key)
        return meta

    def _chunk_index_from_id(self, cid: str) -> int:
        try:
            return int(str(cid).rpartition("::")[2])
        except Exception:
            return -1

    def _rows(self, ids: list = None, where: dict = None) -> list:
        if self.count() == 0:
            return []
        expr = self._where_expr(where)
        if ids:
            id_expr = "id IN (" + ", ".join(self._quote(i) for i in ids) + ")"
            expr = f"{expr} AND {id_expr}" if expr else id_expr
        query = self.table
        if expr:
            query = query.search().where(expr)
        else:
            query = query.search()
        return query.limit(self.count()).to_list()

    def get(self, ids: list = None, where: dict = None, include: list = None) -> dict:
        include = include or []
        rows = self._rows(ids=ids, where=where)
        result = {"ids": [row["id"] for row in rows]}
        if "documents" in include:
            result["documents"] = [row.get("document") or "" for row in rows]
        if "metadatas" in include:
            result["metadatas"] = [self._metadata_from_row(row) for row in rows]
        return result

    def add(self, ids: list, embeddings: list, documents: list, metadatas: list):
        if not ids:
            return
        self.delete(ids=ids)
        rows = []
        chunk_count_by_source = {}
        for cid, meta in zip(ids, metadatas):
            source = (meta or {}).get("source") or str(cid).rpartition("::")[0]
            chunk_count_by_source[source] = chunk_count_by_source.get(source, 0) + 1

        for cid, vector, doc, meta in zip(ids, embeddings, documents, metadatas):
            meta = dict(meta or {})
            source = meta.get("source") or str(cid).rpartition("::")[0]
            chunk_index = int(meta.get("chunk_index", self._chunk_index_from_id(cid)))
            source_path = Path(source)
            word_count = len((doc or "").split())
            rows.append({
                "id": cid,
                "vector": vector,
                "document": doc,
                "source": source,
                "project": meta.get("project"),
                "filename": meta.get("filename"),
                "file_hash": meta.get("file_hash"),
                "chunk_index": chunk_index,
                "chunk_count": int(meta.get(
                    "chunk_count", chunk_count_by_source.get(source, 0))),
                "word_count": int(meta.get("word_count", word_count)),
                "char_count": int(meta.get("char_count", len(doc or ""))),
                "source_ext": meta.get("source_ext") or source_path.suffix.lower(),
                "source_stem": meta.get("source_stem") or source_path.stem,
                "metadata_json": json.dumps(meta),
            })
        self.table.add(rows)
        self._indexes_dirty = True

    def delete(self, ids: list):
        if not ids or self.count() == 0:
            return
        for i in range(0, len(ids), 500):
            batch = ids[i:i + 500]
            expr = "id IN (" + ", ".join(self._quote(cid) for cid in batch) + ")"
            self.table.delete(expr)

    def query(self, query_embeddings: list, n_results: int = 5,
              where: dict = None, **kwargs) -> dict:
        if self.count() == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        expr = self._where_expr(where)
        out = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        limit = min(n_results, self.count())
        for vector in query_embeddings:
            q = self.table.search(vector).limit(limit)
            if expr:
                q = q.where(expr, prefilter=True)
            rows = q.to_list()
            out["ids"].append([row["id"] for row in rows])
            out["documents"].append([row.get("document") or "" for row in rows])
            out["metadatas"].append([self._metadata_from_row(row) for row in rows])
            out["distances"].append([row.get("_distance") for row in rows])
        return out

    def text_search(self, text: str, limit: int = 10, where: dict = None) -> list:
        if self.count() == 0 or not text.strip():
            return []
        expr = self._where_expr(where)
        q = self.table.search(
            text, query_type="fts", fts_columns="document"
        ).limit(min(limit, self.count()))
        if expr:
            q = q.where(expr)
        try:
            return q.to_list()
        except Exception:
            return []

    def ensure_indexes(self):
        n = self.count()
        if n == 0:
            return
        try:
            self.table.create_fts_index(
                "document",
                replace=True,
                use_tantivy=False,
            )
        except Exception as e:
            print(f"  [Warning] LanceDB full-text index unavailable: {e}")
        for column in ("source", "project", "filename", "chunk_index"):
            try:
                self.table.create_scalar_index(column, replace=True)
            except Exception as e:
                print(f"  [Warning] LanceDB scalar index unavailable for {column}: {e}")
        if n >= self.INDEX_MIN_ROWS:
            try:
                partitions = max(1, min(64, int(math.sqrt(n))))
                self.table.create_index(
                    metric="cosine",
                    index_type="IVF_FLAT",
                    num_partitions=partitions,
                    replace=True,
                )
            except Exception as e:
                print(f"  [Warning] LanceDB vector index unavailable: {e}")
        self._indexes_dirty = False

print("Loading embedding model...")
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    local_files_only=True,
)

VECTOR_DB_PATH = str(BASE_DIR / "lancedb")
collection = LanceCollection(VECTOR_DB_PATH, "my_documents_v2", 384)

print(f"Vector DB path:  {VECTOR_DB_PATH}")
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

def _memory_int_setting(key: str, default: int) -> int:
    if not MEMORY_AVAILABLE or get_setting is None:
        return default
    try:
        return int(get_setting(key, str(default)))
    except Exception as e:
        print(f"  [Warning] memory setting {key} unavailable: {e}")
        return default


def chunk_text(text, chunk_size=None, overlap=None):
    chunk_size = int(chunk_size or _memory_int_setting("rag_chunk_size", 500))
    overlap = int(overlap if overlap is not None
                  else _memory_int_setting("rag_chunk_overlap", 100))
    overlap = max(0, min(overlap, chunk_size - 1))
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


def _metadata_source_type(meta: dict) -> str:
    source = meta.get("source") or ""
    return meta.get("source_type") or (
        "email" if source.startswith(EMAIL_SOURCE_PREFIX) else "file")


def get_indexed_sources(source_type: str = None) -> dict:
    if collection.count() == 0:
        return {}
    results = collection.get(include=["metadatas"])
    indexed = {}
    for meta in results["metadatas"]:
        if source_type and _metadata_source_type(meta) != source_type:
            continue
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


DOCUMENT_PROFILE_MODEL = "local-profile-v2"


def _project_of(rel_path: str) -> str:
    """Top folder under the documents root, or 'unfiled' for loose files."""
    parts = Path(rel_path).parts
    return parts[0] if len(parts) > 1 else "unfiled"


def _project_scope(project: str):
    return None if project in (None, "", projects.ALL) else project


def _mtime_iso(file: Path) -> str:
    return datetime.fromtimestamp(
        file.stat().st_mtime, tz=timezone.utc).isoformat()


def index_text_source(
    source: str,
    text: str,
    source_hash: str,
    *,
    project: str,
    filename: str,
    source_ext: str,
    source_stem: str,
    source_type: str = "file",
    extra_metadata: dict = None,
) -> int:
    if not text.strip():
        print(f"  Skipping {filename} (empty or unreadable)")
        return 0

    chunks = chunk_text(text)
    if not chunks:
        return 0

    extra_metadata = dict(extra_metadata or {})
    ids = [f"{source}::{i}" for i in range(len(chunks))]
    embeddings = embedder.encode(chunks).tolist()
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[
            {
                **extra_metadata,
                "source": source,
                "project": project,
                "filename": filename,
                "file_hash": source_hash,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "word_count": len(chunk.split()),
                "char_count": len(chunk),
                "source_ext": source_ext,
                "source_stem": source_stem,
                "source_type": source_type,
            }
            for i, chunk in enumerate(chunks)
        ],
    )
    return len(chunks)


def index_file(file: Path, current_hash: str, file_identity: dict = None) -> int:
    text = load_file(file)
    if not text.strip():
        print(f"  Skipping {file.name} (empty or unreadable)")
        return 0

    # Source is the path relative to the documents root, so two projects can
    # both hold a notes.md without their chunk IDs colliding.
    try:
        rel = str(file.resolve().relative_to(Path(DOCUMENTS_FOLDER)))
    except ValueError:
        rel = file.name
    project = _project_of(rel)
    file_identity = dict(file_identity or {})

    count = index_text_source(
        rel,
        text,
        current_hash,
        project=project,
        filename=file.name,
        source_ext=file.suffix.lower(),
        source_stem=file.stem,
        source_type="file",
        extra_metadata={
            "file_id": file_identity.get("file_id"),
            "file_version_id": file_identity.get("file_version_id"),
            "storage_root_id": file_identity.get("storage_root_id"),
            "storage_object_id": file_identity.get("storage_object_id"),
        },
    )
    record_document_profile(
        rel, text, project, source_hash=current_hash,
        file_identity=file_identity)
    return count


def scan_documents(folder: str = None, verbose: bool = True,
                   force: bool = False,
                   overwrite_profiles: bool = False) -> dict:
    folder_path = Path(folder).resolve() if folder else Path(DOCUMENTS_FOLDER)
    active_extensions = supported_extensions()
    active_ignored_dirs = ignored_dirs()
    storage_root_id = None
    scan_run_id = None
    if MEMORY_AVAILABLE and ensure_storage_root is not None:
        try:
            storage_root_id = ensure_storage_root(
                str(folder_path), description="RAG documents root")
            scan_run_id = start_file_scan(storage_root_id)
        except Exception as e:
            print(f"  [Warning] could not start El Roi scan: {e}")

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
        and f.suffix.lower() in active_extensions
        and not any(part in active_ignored_dirs or part.startswith(".")
                    for part in f.relative_to(folder_path).parts[:-1])
    }

    indexed = get_indexed_sources(source_type="file")
    summary = {"new": [], "updated": [], "removed": [], "unchanged": []}
    file_identities = {}

    for filename, current_hash in current_files.items():
        if storage_root_id and record_file_observation is not None:
            file = folder_path / filename
            try:
                file_identities[filename] = record_file_observation(
                    storage_root_id,
                    filename,
                    current_hash,
                    file.stat().st_size,
                    _mtime_iso(file),
                    scan_run_id=scan_run_id,
                    mime_type=None,
                )
                if link_document_file_identity is not None:
                    ident = file_identities[filename]
                    link_document_file_identity(
                        filename,
                        file_id=ident.get("file_id"),
                        file_version_id=ident.get("file_version_id"),
                        storage_root_id=ident.get("storage_root_id"),
                    )
            except Exception as e:
                print(f"  [Warning] could not record El Roi file {filename}: {e}")

    for filename, current_hash in current_files.items():
        file = folder_path / filename  # filename is now a relative path
        if filename not in indexed:
            if verbose:
                print(f"  [+] Indexing new file: {filename}")
            count = index_file(
                file, current_hash, file_identity=file_identities.get(filename))
            if verbose:
                print(f"      Added {count} chunks")
            summary["new"].append(filename)

        elif force or indexed[filename] != current_hash:
            if verbose:
                reason = "forced" if force else "changed"
                print(f"  [~] Re-indexing {reason} file: {filename}")
            remove_source(filename)
            count = index_file(
                file, current_hash, file_identity=file_identities.get(filename))
            if verbose:
                print(f"      Updated with {count} chunks")
            summary["updated"].append(filename)

        else:
            if overwrite_profiles or _needs_document_profile(filename):
                text = load_file(file)
                if text.strip():
                    record_document_profile(
                        filename, text, projects.project_of(filename),
                        source_hash=file_hash(file),
                        file_identity=file_identities.get(filename))
            summary["unchanged"].append(filename)

    if storage_root_id and mark_missing_storage_objects is not None:
        try:
            mark_missing_storage_objects(
                storage_root_id, list(current_files), scan_run_id=scan_run_id)
        except Exception as e:
            print(f"  [Warning] could not mark missing El Roi files: {e}")

    for filename in indexed:
        if filename not in current_files:
            if verbose:
                print(f"  [-] Removing deleted file: {filename}")
            remove_source(filename)
            summary["removed"].append(filename)

    if summary["new"] or summary["updated"] or summary["removed"]:
        collection.ensure_indexes()

    if scan_run_id and finish_file_scan is not None:
        try:
            finish_file_scan(scan_run_id, summary)
        except Exception as e:
            print(f"  [Warning] could not finish El Roi scan: {e}")

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
    file_identity = None
    if MEMORY_AVAILABLE and ensure_storage_root is not None:
        try:
            storage_root_id = ensure_storage_root(
                str(Path(DOCUMENTS_FOLDER).resolve()),
                description="RAG documents root")
            file_identity = record_file_observation(
                storage_root_id,
                rel,
                new_hash,
                dest_path.stat().st_size,
                _mtime_iso(dest_path),
            )
        except Exception as e:
            print(f"  [Warning] could not record El Roi ingest {rel}: {e}")
    chunk_count = index_file(dest_path, new_hash, file_identity=file_identity)
    if chunk_count:
        collection.ensure_indexes()
    return rel, chunk_count


# ---------------------------------------------------------------------------
# Email / Maildir indexing
# ---------------------------------------------------------------------------

def _safe_source_part(value: str, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._@+-]+", "-", (value or "").strip())
    cleaned = cleaned.strip("-.")
    return cleaned[:120] or fallback


def _maildir_name(root: Path, maildir_path: Path) -> str:
    try:
        rel = maildir_path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = Path(maildir_path.name)
    parts = [p for p in rel.parts if p not in ("cur", "new", "tmp")]
    return "/".join(parts) if parts else "INBOX"


def _discover_maildirs(root: Path) -> list:
    if not root.exists():
        return []
    maildirs = []
    for path in [root, *root.rglob("*")]:
        if not path.is_dir():
            continue
        if all((path / name).is_dir() for name in ("cur", "new", "tmp")):
            maildirs.append(path)
    return sorted(set(maildirs))


def _message_body_text(msg) -> tuple:
    body_parts = []
    attachments = []

    def payload_text(part):
        try:
            content = part.get_content()
        except Exception:
            payload = part.get_payload(decode=True)
            if payload is None:
                return ""
            charset = part.get_content_charset() or "utf-8"
            content = payload.decode(charset, errors="replace")
        if part.get_content_type() == "text/html":
            content = re.sub(r"(?is)<(script|style).*?</\1>", " ", content)
            content = re.sub(r"(?s)<[^>]+>", " ", content)
            content = re.sub(r"&nbsp;", " ", content)
        return re.sub(r"\s+", " ", content).strip()

    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue
            disposition = (part.get_content_disposition() or "").lower()
            ctype = part.get_content_type()
            filename = part.get_filename()
            if disposition == "attachment" or filename:
                attachments.append(
                    f"{filename or 'unnamed'} ({ctype})")
                continue
            if ctype in ("text/plain", "text/html"):
                text = payload_text(part)
                if text:
                    body_parts.append(text)
    else:
        if msg.get_content_type() in ("text/plain", "text/html"):
            text = payload_text(msg)
            if text:
                body_parts.append(text)

    return "\n\n".join(body_parts), attachments


def _format_email_text(msg, account: str, mailbox_name: str,
                       local_path: Path) -> tuple:
    headers = {
        "Subject": msg.get("subject", ""),
        "From": msg.get("from", ""),
        "To": msg.get("to", ""),
        "Cc": msg.get("cc", ""),
        "Bcc": msg.get("bcc", ""),
        "Date": msg.get("date", ""),
        "Message-ID": msg.get("message-id", ""),
        "In-Reply-To": msg.get("in-reply-to", ""),
        "References": msg.get("references", ""),
    }
    body, attachments = _message_body_text(msg)
    lines = [
        "Email message",
        f"Account: {account}",
        f"Mailbox: {mailbox_name}",
    ]
    for label, value in headers.items():
        clean = re.sub(r"\s+", " ", value or "").strip()
        if clean:
            lines.append(f"{label}: {clean}")
    if attachments:
        lines.append("Attachments: " + ", ".join(attachments))
    lines.extend(["", body])
    text = "\n".join(lines).strip()

    try:
        date_value = parsedate_to_datetime(headers["Date"]).isoformat()
    except Exception:
        date_value = ""
    return text, {
        "email_account": account,
        "email_mailbox": mailbox_name,
        "email_subject": headers["Subject"],
        "email_from": headers["From"],
        "email_to": headers["To"],
        "email_cc": headers["Cc"],
        "email_date": date_value or headers["Date"],
        "email_message_id": headers["Message-ID"],
        "email_local_path": str(local_path),
        "email_attachment_count": len(attachments),
    }


def _iter_maildir_messages(root: Path):
    account = _safe_source_part(root.name, "mail")
    for maildir_path in _discover_maildirs(root):
        mailbox_name = _maildir_name(root, maildir_path)
        for subdir in ("new", "cur"):
            folder = maildir_path / subdir
            for message_path in sorted(folder.iterdir()):
                if not message_path.is_file() or message_path.name.startswith("."):
                    continue
                try:
                    with message_path.open("rb") as fh:
                        msg = BytesParser(policy=policy.default).parse(fh)
                    text, meta = _format_email_text(
                        msg, account, mailbox_name, message_path)
                    stable_id = (msg.get("message-id") or "").strip()
                    if not stable_id:
                        stable_id = str(message_path.resolve().relative_to(root))
                    digest = hashlib.sha1(stable_id.encode("utf-8")).hexdigest()[:24]
                    source = (
                        f"{EMAIL_SOURCE_PREFIX}{account}/"
                        f"{_safe_source_part(mailbox_name, 'INBOX')}/{digest}.eml"
                    )
                    source_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                    yield source, source_hash, text, meta
                except Exception as e:
                    print(f"  [Warning] could not read email {message_path}: {e}")


def scan_mailboxes(roots: list = None, verbose: bool = True,
                   force: bool = False) -> dict:
    roots = [Path(r).expanduser().resolve() for r in (roots or email_maildir_roots())]
    indexed = get_indexed_sources(source_type="email")
    current = {}
    scanned_accounts = set()
    summary = {
        "new": [], "updated": [], "removed": [], "unchanged": [],
        "roots": [str(root) for root in roots],
        "maildirs": [],
    }

    if not roots:
        if verbose:
            print("  [Warning] no mail roots configured; set IMAP_MAILDIR_ROOTS")
        return summary

    for root in roots:
        if not root.exists():
            if verbose:
                print(f"  [Warning] mail root does not exist: {root}")
            continue
        scanned_accounts.add(_safe_source_part(root.name, "mail"))
        summary["maildirs"].extend(str(p) for p in _discover_maildirs(root))
        for source, source_hash, text, meta in _iter_maildir_messages(root):
            current[source] = source_hash
            if source not in indexed:
                if verbose:
                    print(f"  [+] Indexing new email: {source}")
                count = index_text_source(
                    source, text, source_hash,
                    project="email",
                    filename=meta.get("email_subject") or Path(source).name,
                    source_ext=".eml",
                    source_stem=Path(source).stem,
                    source_type="email",
                    extra_metadata=meta,
                )
                if verbose:
                    print(f"      Added {count} chunks")
                summary["new"].append(source)
            elif force or indexed[source] != source_hash:
                if verbose:
                    reason = "forced" if force else "changed"
                    print(f"  [~] Re-indexing {reason} email: {source}")
                remove_source(source)
                count = index_text_source(
                    source, text, source_hash,
                    project="email",
                    filename=meta.get("email_subject") or Path(source).name,
                    source_ext=".eml",
                    source_stem=Path(source).stem,
                    source_type="email",
                    extra_metadata=meta,
                )
                if verbose:
                    print(f"      Updated with {count} chunks")
                summary["updated"].append(source)
            else:
                summary["unchanged"].append(source)

    for source in indexed:
        account = Path(source).parts[1] if len(Path(source).parts) > 1 else ""
        if account in scanned_accounts and source not in current:
            if verbose:
                print(f"  [-] Removing deleted email: {source}")
            remove_source(source)
            summary["removed"].append(source)

    if summary["new"] or summary["updated"] or summary["removed"]:
        collection.ensure_indexes()
    return summary


def read_indexed_source_text(source: str) -> str:
    results = collection.get(where={"source": source},
                             include=["documents", "metadatas"])
    rows = sorted(
        zip(results.get("documents", []), results.get("metadatas", [])),
        key=lambda pair: int(pair[1].get("chunk_index", 0) or 0),
    )
    return "\n\n".join(doc for doc, _meta in rows if doc)

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


def _clean_metadata_value(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip(" \t\r\n:;,")


def _meaningful_header_lines(text: str, limit: int = 120) -> list:
    lines = []
    for line in text.splitlines():
        clean = _clean_metadata_value(line)
        if clean:
            lines.append(clean)
        if len(lines) >= limit:
            break
    return lines


def _labeled_block(lines: list, label: str, stop_labels: set = None) -> list:
    stop_labels = {s.lower() for s in (stop_labels or set())}
    label_lower = label.lower()
    out = []
    in_block = False
    for line in lines:
        normalized = line.lower().strip(":")
        if normalized == label_lower:
            in_block = True
            continue
        if in_block and normalized in stop_labels:
            break
        if in_block:
            out.append(line)
    return out


def _split_people(value: str) -> list:
    value = re.sub(r"\b(?:and|&)\b", ";", value or "", flags=re.I)
    value = re.sub(r"\s*\d+\s*(?=;|,|$)", "", value)
    parts = re.split(r";|\n", value)
    authors = []
    for part in parts:
        clean = _clean_metadata_value(part)
        if not clean:
            continue
        if re.search(r"\b(university|department|faculty|school|journal|vol\.|issue|publisher|email)\b", clean, re.I):
            continue
        if clean.lower() in {"authors", "author", "source", "article"}:
            continue
        if clean not in authors:
            authors.append(clean)
    return authors


def _looks_like_author_line(line: str) -> bool:
    if len(line) > 90:
        return False
    if re.search(r"\b(university|department|faculty|school|journal|abstract|email|received|accepted|doi)\b", line, re.I):
        return False
    tokens = [t for t in re.split(r"\s+", line) if t]
    if not 2 <= len(tokens) <= 6:
        return False
    return sum(bool(re.match(r"^[A-Z][A-Za-z'.-]+,?$", t)) for t in tokens) >= 2


def _document_authors(text: str) -> list:
    lines = _meaningful_header_lines(text)
    stop_labels = {
        "source", "publisher information", "publication year",
        "subject terms", "subject geographic", "description", "abstract",
    }
    labeled = _labeled_block(lines, "Authors", stop_labels)
    if labeled:
        return _split_people("\n".join(labeled))

    for i, line in enumerate(lines[:60]):
        if line.lower() != "article":
            continue
        title_seen = False
        for candidate in lines[i + 1:i + 8]:
            if candidate.lower() in {"abstract", "keywords"}:
                break
            if len(candidate) > 30 and not title_seen:
                title_seen = True
                continue
            if title_seen and _looks_like_author_line(candidate):
                return _split_people(candidate)

    for i, line in enumerate(lines[:50]):
        if line.lower().startswith(("abstract", "keywords", "introduction")):
            break
        if _looks_like_author_line(line):
            previous = " ".join(lines[max(0, i - 3):i]).lower()
            if any(marker in previous for marker in ["doi:", "article", "journal"]):
                return _split_people(line)
    return []


def _derive_document_label(source: str, text: str) -> str:
    lines = _meaningful_header_lines(text)
    stop_labels = {"authors", "author", "source", "abstract", "description"}
    title_block = _labeled_block(lines, "Title", stop_labels)
    if title_block:
        title = _clean_metadata_value(" ".join(title_block))
        if len(title) > 10:
            return title[:180]
    for line in lines[:100]:
        match = re.match(r"(?:Thesis\s+)?Title\s*:\s*(.+)", line.strip(), re.I)
        if match and len(match.group(1)) > 10:
            return match.group(1).strip()[:180]
    for i, line in enumerate(lines[:50]):
        if line.lower() == "article":
            title_lines = []
            for candidate in lines[i + 1:i + 5]:
                if _looks_like_author_line(candidate):
                    break
                title_lines.append(candidate)
            title = _clean_metadata_value(" ".join(title_lines))
            if len(title) > 20:
                return title[:180]
    for line in lines:
        clean = _clean_metadata_value(line)
        if len(clean) > 20:
            return clean[:180]
    return Path(source).stem.replace("_", " ").replace("-", " ").strip()


def _document_sections(text: str) -> dict:
    keywords = _criteria_terms("document_section")
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
    if any(term in lower[:4000] for term in
           _criteria_group_terms("genre_marker", "chat_export")):
        add("chat export")
        return genres

    research_guide_terms = _criteria_group_terms("genre_marker", "research_methods_guide")
    if any(term in lower[:4000] for term in research_guide_terms):
        add("research methods guide")
        book_chapter_terms = _criteria_group_terms("genre_marker", "book_chapter")
        if any(term in lower[:4000] for term in book_chapter_terms):
            add("book chapter")
        return genres

    dissertation_template_terms = _criteria_group_terms("genre_marker", "dissertation_template")
    if (any(term in filename for term in dissertation_template_terms)
            or any(term in lower[:4000] for term in dissertation_template_terms)):
        add("dissertation template")
        return genres

    dissertation_markers = _criteria_group_terms("genre_marker", "dissertation")
    if (
        any(term in filename for term in dissertation_markers)
        or any(term in lower[:10000] for term in dissertation_markers)
    ):
        add("dissertation")
        return genres

    research_book_filename_terms = _criteria_group_terms(
        "genre_marker", "research_book_filename")
    research_book_head_markers = _criteria_group_terms(
        "genre_marker", "research_book_head")
    if suffix == ".pdf" and (
        any(term in filename for term in research_book_filename_terms)
        or (
            sum(1 for term in research_book_head_markers if term in lower[:8000]) >= 2
            and any(term in lower[:8000] for term in
                    _criteria_group_terms("theme_marker", "research_methods"))
        )
    ):
        add("research methods guide")
        return genres

    coursework_terms = _criteria_group_terms("genre_marker", "coursework")
    if any(term in lower[:6000] or term in filename for term in coursework_terms):
        coursework_dissertation_terms = (
            _criteria_group_terms("genre_marker", "coursework_dissertation")
        )
        if any(term in lower[:6000] for term in coursework_dissertation_terms):
            add("dissertation draft")
        else:
            add("coursework")
        return genres

    presentation_terms = _criteria_group_terms("genre_marker", "presentation")
    if suffix == ".pptx" or any(term in lower for term in presentation_terms):
        add("presentation")
        return genres

    spreadsheet_terms = _criteria_group_terms("genre_marker", "spreadsheet")
    if suffix in {".xlsx", ".xls"} or any(lower.startswith(term) for term in spreadsheet_terms):
        add("spreadsheet")
        return genres

    academic_sections = _criteria_group_terms("genre_marker", "academic_section")
    head = lower[:8000]
    article_filename = bool(re.match(r"^\d+-\d+-\d+-", filename))
    article_filename_terms = (
        _criteria_group_terms("genre_marker", "academic_filename"))
    article_filename = article_filename or any(token in filename for token in article_filename_terms)
    scholarly_markers = _criteria_group_terms("genre_marker", "scholarly_marker")
    marker_hits = sum(1 for marker in scholarly_markers if marker in head)
    if suffix == ".pdf" and (
        (article_filename and marker_hits >= 1)
        or marker_hits >= 3
        or len(sections & academic_sections) >= 5
    ):
        add("academic article")

    legal_filing_terms = _criteria_group_terms("genre_marker", "legal_filing")
    if "academic article" not in genres and sum(1 for term in legal_filing_terms if term in head) >= 3:
        add("legal filing")

    agreement_head_terms = _criteria_group_terms("genre_marker", "contract_agreement")
    if "academic article" not in genres and any(term in head for term in agreement_head_terms):
        add("contract/agreement")

    if (
        any(term in lower[:8000] for term in (
            _criteria_group_terms("genre_marker", "interview_protocol")
        ))
        or (
            any(term in lower[:8000] for term in
                _criteria_group_terms("theme_marker", "research_methods"))
            and any(term in lower[:8000] for term in
                    _criteria_group_terms("genre_marker", "interview_protocol"))
        )
    ):
        add("interview protocol")

    literature_review_terms = (
        _criteria_group_terms("genre_marker", "literature_review")
    )
    if (
        "literature review" in sections
        or any(term in lower[:8000] for term in literature_review_terms)
    ):
        add("literature review")

    notes_terms = _criteria_group_terms("genre_marker", "notes")
    if suffix in {".md", ".txt"} or any(term in lower[:4000] for term in notes_terms):
        add("notes")

    return genres or ["document"]

def _document_themes(text: str) -> list:
    lower = text.lower()
    themes = []

    if any(term in lower for term in
           _criteria_group_terms("theme_marker", "ai_adoption")):
        themes.append("ai adoption")

    training_terms = (
        _criteria_group_terms("theme_marker", "training_usability")
    )
    technology_terms = (
        _criteria_group_terms("theme_marker", "technology_context")
    )
    if (
        any(term in lower for term in training_terms)
        and any(term in lower for term in technology_terms)
    ):
        themes.append("training and usability")

    if any(term in lower for term in
           _criteria_group_terms("theme_marker", "legal_privilege")):
        themes.append("legal privilege")

    if any(term in lower for term in
           _criteria_group_terms("theme_marker", "research_methods")):
        themes.append("research methods")

    if any(term in lower for term in
           _criteria_group_terms("theme_marker", "risk_governance")):
        themes.append("risk and governance")

    return themes


def _document_subject_terms(text: str) -> list:
    lines = _meaningful_header_lines(text)
    stop_labels = (
        _criteria_terms("subject_stop_label")
    )
    subjects = _labeled_block(lines, "Subject Terms", stop_labels)
    keywords = []
    for line in lines[:120]:
        match = re.match(r"keywords?\s*:?\s*(.+)", line, re.I)
        if match:
            keywords.extend(re.split(r";|,", match.group(1)))
    terms = []
    for term in subjects + keywords:
        clean = _clean_metadata_value(term)
        if clean and clean not in terms:
            terms.append(clean)
    return terms[:12]


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
    label = _derive_document_label(source, text)
    genres = _document_genres(source, text)
    themes = _document_themes(text)
    authors = _document_authors(text)
    subject_terms = _document_subject_terms(text)
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
        f"document_type: {', '.join(genres)}",
        f"title_or_label: {label}",
    ]
    if authors:
        parts.append("authors: " + "; ".join(authors))
    if subject_terms:
        parts.append("subject_terms: " + "; ".join(subject_terms))
    if themes:
        parts.append("themes: " + "; ".join(themes))
    if headings:
        parts.append("headings: " + " | ".join(headings))
    if lines:
        parts.append("opening: " + " | ".join(lines[:4]))
    if lead:
        parts.append("lead: " + lead)
    return "\n".join(parts)

def record_document_profile(source: str, text: str, project: str,
                            source_hash: str = None,
                            file_identity: dict = None):
    if not MEMORY_AVAILABLE or record_synopsis is None:
        return
    try:
        synopsis = _document_profile(source, text, project)
        file_identity = dict(file_identity or {})
        if record_document_upload_profile is not None:
            record_document_upload_profile(
                source,
                synopsis,
                word_count=len(text.split()),
                model=DOCUMENT_PROFILE_MODEL,
                source_hash=source_hash,
                chars=len(text),
                file_type=Path(source).suffix.lower().lstrip("."),
                project=project,
                label=_derive_document_label(source, text),
                sections_found=_document_sections(text),
                genres=_document_genres(source, text),
                themes=_document_themes(text),
                authors=_document_authors(text),
                subject_terms=_document_subject_terms(text),
                upload_state="project_file",
                file_id=file_identity.get("file_id"),
                file_version_id=file_identity.get("file_version_id"),
                storage_root_id=file_identity.get("storage_root_id"),
            )
        else:
            record_synopsis(
                source,
                synopsis,
                word_count=len(text.split()),
                model=DOCUMENT_PROFILE_MODEL,
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
    return not row or row.get("model") != DOCUMENT_PROFILE_MODEL


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

    project = _project_scope(project)
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
    project = _project_scope(project)
    if collection.count() == 0:
        return []
    results = _semantic_search(question, n_results=n_results, project=project)
    out = []
    for rank, (cid, doc, meta) in enumerate(
        zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
    ):
        out.append({
            "id": cid,
            "document": doc,
            "metadata": dict(meta),
            "score": 1.0 / (rank + 1),
            "signals": {"semantic"},
        })
    return out


def _fulltext_candidates(question: str, n_results: int, project: str = None):
    project = _project_scope(project)
    if collection.count() == 0 or not hasattr(collection, "text_search"):
        return []
    rows = collection.text_search(
        question,
        limit=max(n_results, 12),
        where={"project": project} if project else None,
    )
    out = []
    for rank, row in enumerate(rows):
        meta = collection._metadata_from_row(row)
        out.append({
            "id": row.get("id"),
            "document": row.get("document") or "",
            "metadata": meta,
            "score": 1.6 / (rank + 1),
            "signals": {"fulltext"},
        })
    return out


_SEARCH_CRITERIA_CACHE = {"loaded_at": 0.0, "rows": []}
_SEARCH_CRITERIA_TTL_SECONDS = 30


def _search_criteria_rows() -> list:
    if not MEMORY_AVAILABLE or get_search_criteria is None:
        return []
    now = time.time()
    if now - _SEARCH_CRITERIA_CACHE["loaded_at"] < _SEARCH_CRITERIA_TTL_SECONDS:
        return list(_SEARCH_CRITERIA_CACHE["rows"])
    try:
        rows = get_search_criteria(enabled_only=True)
        _SEARCH_CRITERIA_CACHE["loaded_at"] = now
        _SEARCH_CRITERIA_CACHE["rows"] = rows
        return list(rows)
    except Exception as e:
        print(f"  [Warning] search criteria unavailable from memory-db: {e}")
        return list(_SEARCH_CRITERIA_CACHE["rows"])


def _criteria_terms(criteria_type: str) -> set:
    return {
        (row.get("term") or "").lower()
        for row in _search_criteria_rows()
        if row.get("criteria_type") == criteria_type and row.get("term")
    }


def _criteria_group_terms(criteria_type: str, group_name: str) -> set:
    return {
        (row.get("term") or "").lower()
        for row in _search_criteria_rows()
        if (
            row.get("criteria_type") == criteria_type
            and (row.get("group_name") or "") == group_name
            and row.get("term")
        )
    }


def _domain_groups_from_criteria() -> dict:
    groups = {}
    for row in _search_criteria_rows():
        group = row.get("group_name") or ""
        term = (row.get("term") or "").lower()
        ctype = row.get("criteria_type")
        if not group or not term or ctype not in {"domain_trigger", "domain_term"}:
            continue
        groups.setdefault(group, {"triggers": set(), "terms": set()})
        if ctype == "domain_trigger":
            groups[group]["triggers"].add(term)
        elif ctype == "domain_term":
            groups[group]["terms"].add(term)
    return groups


def _terms(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def _compact(text: str) -> str:
    return "".join(_terms(text))


def _term_pattern(term: str):
    parts = _terms(term)
    if not parts:
        return None
    joined = r"\s+".join(re.escape(part) for part in parts)
    return re.compile(rf"(?<![a-z0-9]){joined}(?![a-z0-9])", re.I)


def _term_count(text: str, term: str) -> int:
    pattern = _term_pattern(term)
    return len(pattern.findall(text or "")) if pattern else 0


def _term_present(text: str, term: str) -> bool:
    return _term_count(text, term) > 0


def meaningful_words(text: str) -> list:
    """
    Words worth treating as exact-match search terms. Length alone is a bad
    filter here: short proper nouns (a first name, initials, a short case
    name) are exactly the kind of term someone searches for, and a bare
    length cutoff drops them silently. A stopword list is the right tool:
    keep every word that isn't a common English function word, regardless
    of length or case, so short proper nouns survive capitalized or not.

    Shared by search()'s filename/content matching and by writer.py's
    gather_evidence(), so both retrieval paths treat a question's meaningful
    terms identically rather than drifting apart over two copies of this.
    """
    words = []
    stopwords = _criteria_terms("stopword")
    low_signal = _criteria_terms("low_signal")
    for clean in _terms(text):
        if (clean and len(clean) >= 2 and clean not in stopwords
                and clean not in low_signal):
            words.append(clean)
    return words


def _word_count_score(text: str, words: list) -> int:
    return sum(_term_count(text, w) for w in words if len(w) >= 3)


def _meaningful_phrases(text: str) -> list:
    stopwords = _criteria_terms("stopword")
    tokens = [t for t in _terms(text) if t not in stopwords and len(t) >= 2]
    phrases = []
    for size in (3, 2):
        for i in range(0, max(0, len(tokens) - size + 1)):
            phrase = " ".join(tokens[i:i + size])
            if any(len(part) >= 4 for part in tokens[i:i + size]):
                phrases.append(phrase)
    return phrases


def _phrase_score(text: str, question: str) -> int:
    return sum(3 * _term_count(text, phrase)
               for phrase in _meaningful_phrases(question))


def _query_required_domain_groups(question: str) -> list:
    q_terms = set(_terms(question or ""))
    required = []
    for group in _domain_groups_from_criteria().values():
        if q_terms & group["triggers"]:
            required.append(group["terms"])
    return required


def _matches_required_domain_groups(text: str, required_groups: list) -> bool:
    if not required_groups:
        return True
    return all(any(_term_present(text, term) for term in group)
               for group in required_groups)


def _is_reference_like(text: str) -> bool:
    lower = (text or "").lower()
    noise_terms = _criteria_terms("section_noise")
    if any(_term_present(lower, term) for term in noise_terms):
        return True
    doi_hits = lower.count("doi:")
    bracket_refs = len(re.findall(r"(?<!\w)\[\d+\]", lower))
    citation_punctuation = len(re.findall(r"\bvol\.\s*\d+|\bpp\.\s*\d+|https?://", lower))
    return doi_hits >= 2 or bracket_refs >= 3 or citation_punctuation >= 3


def _query_allows_reference_chunks(question: str) -> bool:
    return any(_term_present(question, term)
               for term in (_criteria_terms("section_noise") or set()))


def _source_matches(source: str, words: list, question: str = "") -> int:
    source_lower = source.lower()
    filename_lower = Path(source).name.lower()
    stem_lower = Path(source).stem.lower()
    source_compact = _compact(source)
    query_compact = _compact(question)
    score = 0
    if query_compact and query_compact in source_compact:
        score += 14
    source_low_signal = _criteria_terms("source_low_signal")
    for w in words:
        if len(w) < 3:
            continue
        if w in source_low_signal:
            continue
        if _term_present(stem_lower, w) or w in _compact(stem_lower):
            score += 6
        elif _term_present(filename_lower, w) or w in _compact(filename_lower):
            score += 4
        elif _term_present(source_lower, w):
            score += 2
    return score


EMAIL_PERSON_RE = re.compile(
    r"\b(from|to|cc|sent by|sent from)\s+([A-Za-z0-9._%+-]+(?:\s+[A-Za-z0-9._%+-]+){0,3})",
    re.I,
)


def _email_intent(question: str) -> dict:
    lower = (question or "").lower()
    wants_mail = any(term in lower for term in ("email", "mail", "message"))
    fields = {}
    for field, value in EMAIL_PERSON_RE.findall(question or ""):
        normalized = "from" if field in {"from", "sent by", "sent from"} else field
        name = " ".join(value.split()).strip(" .,;:!?")
        if name:
            fields[normalized] = name
            wants_mail = True
    return {"wants_mail": wants_mail, "fields": fields}


def _email_match_score(meta: dict, question: str) -> float:
    if _metadata_source_type(meta) != "email":
        return 0.0
    intent = _email_intent(question)
    score = 0.0
    if intent["wants_mail"]:
        score += 1.5
    field_map = {
        "from": "email_from",
        "to": "email_to",
        "cc": "email_cc",
    }
    for field, value in intent["fields"].items():
        haystack = (meta.get(field_map.get(field, "")) or "").lower()
        terms = _terms(value)
        if terms and all(term in haystack for term in terms):
            score += 30.0
        elif terms and any(term in haystack for term in terms):
            score += 10.0
    try:
        dt = datetime.fromisoformat((meta.get("email_date") or "").replace("Z", "+00:00"))
        age_days = max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 86400)
        score += max(0.0, 6.0 - min(age_days, 365.0) / 60.0)
    except Exception:
        pass
    return score


def _metadata_source(meta: dict) -> str:
    return meta.get("source", "")


def _candidate_key(doc: str, meta: dict) -> str:
    return f"{_metadata_source(meta)}::{doc[:120]}"


def _add_candidate(candidates: dict, doc: str, meta: dict,
                   score: float, signal: str, cid: str = None):
    key = _candidate_key(doc, meta)
    if key not in candidates:
        candidates[key] = {
            "id": cid,
            "document": doc,
            "metadata": dict(meta),
            "score": 0.0,
            "signals": set(),
        }
    elif cid and not candidates[key].get("id"):
        candidates[key]["id"] = cid
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


def _file_excerpt(text: str, words: list, radius: int = 420) -> str:
    lower = text.lower()
    positions = [lower.find(w) for w in words if len(w) >= 3 and lower.find(w) >= 0]
    if not positions:
        return " ".join(text.split()[:120])
    pos = min(positions)
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    excerpt = text[start:end]
    return " ".join(excerpt.split())


def _document_row_text(row: dict) -> str:
    def as_list(value):
        if isinstance(value, list):
            return value
        if not value:
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    parts = [
        row.get("source") or "",
        row.get("label") or "",
        row.get("synopsis") or "",
        " ".join(as_list(row.get("genres"))),
        " ".join(as_list(row.get("themes"))),
        " ".join(as_list(row.get("authors"))),
        " ".join(as_list(row.get("subject_terms"))),
    ]
    return " ".join(parts)


def _main_body_text(text: str) -> str:
    """
    Keep source-level topical matching anchored in the article body.
    References and bibliographies are useful for citation lookups, but a topic
    appearing only there should not make the whole source count as being about
    that topic.
    """
    match = re.search(
        r"(?im)^\s*(references|bibliography|works cited|literature cited)\s*$",
        text or "",
    )
    return (text or "")[:match.start()] if match else (text or "")


def _json_list(value) -> list:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def mine_document_store(question: str, project: str = None,
                        limit: int = 8, max_chars_per_file: int = 120000) -> list:
    """
    Source-of-truth fallback over the actual document store.

    Chroma is a passage index. The document store is the file-server truth.
    When a result is thin or ambiguous, this scans the stored files directly
    and returns source-level matches plus extracted snippets.
    """
    project = _project_scope(project)
    words = meaningful_words(question)
    required_domain_groups = _query_required_domain_groups(question)
    if not words and not required_domain_groups:
        return []

    rows_by_source = {}
    if MEMORY_AVAILABLE and get_synopsis is not None:
        for source in get_indexed_sources():
            if project and projects.project_of(source) != project:
                continue
            try:
                row = get_synopsis(source) or {}
            except Exception:
                row = {}
            rows_by_source[source] = row

    matches = []
    root = Path(DOCUMENTS_FOLDER)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            source = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            source = path.name
        if project and projects.project_of(source) != project:
            continue
        if any(part in ignored_dirs() or part.startswith(".")
               for part in Path(source).parts[:-1]):
            continue

        row = rows_by_source.get(source) or {}
        metadata_text = _document_row_text(row)
        try:
            text = load_file(path)
        except Exception as e:
            print(f"  [Warning] source mining could not read {source}: {e}")
            continue
        if not text.strip():
            continue

        main_text = _main_body_text(text[:max_chars_per_file])
        identity_text = " ".join([
            source,
            row.get("label") or _derive_document_label(source, text),
        ])
        topic_searchable = f"{identity_text}\n{main_text}".lower()
        searchable = f"{metadata_text}\n{identity_text}\n{main_text}".lower()
        if not _matches_required_domain_groups(
                topic_searchable, required_domain_groups):
            continue
        distinct_term_hits = sum(
            1 for word in dict.fromkeys(words)
            if len(word) >= 3 and _term_present(searchable, word)
        )
        if len([w for w in words if len(w) >= 3]) >= 3 and distinct_term_hits < 2:
            continue
        score = _word_count_score(searchable, words)
        score += _source_matches(source, words, question=question)
        if score <= 0:
            continue
        matches.append({
            "source": source,
            "score": score,
            "label": row.get("label") or _derive_document_label(source, text),
            "genres": _json_list(row.get("genres")) or _document_genres(source, text),
            "authors": _json_list(row.get("authors")) or _document_authors(text),
            "subject_terms": (
                _json_list(row.get("subject_terms"))
                or _document_subject_terms(text)
            ),
            "snippet": _file_excerpt(main_text, words),
        })

    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[:limit]


def _registry_candidates(words: list, project: str = None) -> list:
    project = _project_scope(project)
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
    project = _project_scope(project)
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
    project = _project_scope(project)
    if collection.count() == 0:
        return []

    words = meaningful_words(question)
    required_domain_groups = _query_required_domain_groups(question)
    allow_reference_chunks = _query_allows_reference_chunks(question)
    candidates = {}
    all_data = collection.get(include=["metadatas", "documents"])

    for item in _citation_candidates(words, project=project):
        _add_candidate(
            candidates,
            item["document"],
            item["metadata"],
            item["score"],
            "citation",
            item.get("id"),
        )

    for item in _registry_candidates(words, project=project):
        _add_candidate(
            candidates,
            item["document"],
            item["metadata"],
            item["score"],
            "document_registry",
            item.get("id"),
        )

    for item in _fulltext_candidates(question, n_results * 3, project=project):
        _add_candidate(
            candidates,
            item["document"],
            item["metadata"],
            item["score"],
            "fulltext",
            item.get("id"),
        )

    for i, meta in enumerate(all_data["metadatas"]):
        if project and meta.get("project") != project:
            continue
        cid = all_data["ids"][i]
        doc = all_data["documents"][i]
        source = meta.get("source", "")
        candidate_text = " ".join([
            source,
            meta.get("filename") or "",
            meta.get("source_stem") or "",
            doc,
        ])
        if not _matches_required_domain_groups(
            candidate_text, required_domain_groups):
            continue
        source_score = _source_matches(source, words, question=question)
        content_score = _word_count_score(doc, words) + _phrase_score(doc, question)
        email_score = _email_match_score(meta, question)
        if _is_reference_like(doc) and not allow_reference_chunks:
            content_score = 0
        if email_score:
            _add_candidate(
                candidates, doc, meta, email_score, "email", cid)
        if source_score:
            _add_candidate(
                candidates, doc, meta, 2.0 + source_score / 4, "source", cid)
        if content_score:
            _add_candidate(candidates, doc, meta,
                           1.0 + math.log1p(content_score), "keyword", cid)

    semantic_limit = max(n_results * 3, 12)
    for item in _semantic_candidates(question, semantic_limit, project=project):
        _add_candidate(
            candidates,
            item["document"],
            item["metadata"],
            item["score"],
            "semantic",
            item.get("id"),
        )

    ranked = sorted(
        [
            c for c in candidates.values()
            if _matches_required_domain_groups(
                " ".join([
                    c["metadata"].get("source") or "",
                    c["metadata"].get("filename") or "",
                    c["metadata"].get("source_stem") or "",
                    c["document"] or "",
                ]),
                required_domain_groups,
            )
            and (
                allow_reference_chunks
                or not _is_reference_like(c["document"])
                or "source" in c["signals"]
                or "citation" in c["signals"]
            )
        ],
        key=lambda c: (
            c["score"] + 0.35 * max(0, len(c["signals"]) - 1),
            "citation" in c["signals"],
            "document_registry" in c["signals"],
            "fulltext" in c["signals"],
            "email" in c["signals"],
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
