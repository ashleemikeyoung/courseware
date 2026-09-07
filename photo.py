"""Photo helpers for Ask mode.

This module owns explicit /photo handling and photo-mode state. The first
version leans on the existing El Roi document scanner: photos placed under the
documents root can already be indexed for OCR, local vision descriptions, and
RAW metadata. The command layer makes that workflow deliberate.
"""

from __future__ import annotations

import re
from pathlib import Path

import projects

PHOTO_EXTENSIONS = {
    ".arw", ".bmp", ".cr2", ".cr3", ".dng", ".gif", ".jpeg", ".jpg",
    ".nef", ".orf", ".png", ".raf", ".rw2", ".tiff",
}

PHOTO_COMMAND_RE = re.compile(r"^\s*/photo\b\s*(.*)$", re.IGNORECASE | re.DOTALL)
PHOTO_EXIT_RE = re.compile(
    r"^\s*(?:/photo\s+(?:off|exit|stop|done)|/exit\s+photo|"
    r"exit\s+photo\s+mode|leave\s+photo\s+mode|stop\s+photo\s+mode)\s*$",
    re.IGNORECASE,
)


def _is_photo_command(question: str) -> bool:
    return bool(PHOTO_COMMAND_RE.match(question or ""))


def _photo_command_query(question: str) -> str:
    m = PHOTO_COMMAND_RE.match(question or "")
    return (m.group(1) if m else "").strip()


def _is_photo_mode_exit(question: str) -> bool:
    return bool(PHOTO_EXIT_RE.match(question or ""))


def _photo_mode_active(messages: list) -> bool:
    active = False
    for message in messages or []:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if _is_photo_mode_exit(content):
            active = False
        elif _is_photo_command(content):
            active = True
    return active


def _photo_mode_exit_response() -> dict:
    return {
        "text": "Photo mode is off. I’ll treat the next request normally.",
        "evidence": {},
        "grounded": False,
        "metrics": {"route": "photo_command", "photo_mode": False},
    }


def _photo_mode_question(question: str) -> str:
    return question if _is_photo_command(question) else f"/photo {question or ''}".strip()


def _documents_root() -> Path:
    from rag import DOCUMENTS_FOLDER
    return Path(DOCUMENTS_FOLDER).resolve()


def _resolve_ingest_target(raw: str, project: str = None) -> tuple[Path, str]:
    root = _documents_root()
    value = (raw or "").strip().strip("\"'")
    if not value:
        target = root / (project if project and project != projects.ALL else "Photos")
    else:
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            try:
                target = candidate.resolve().relative_to(root)
            except ValueError:
                return root, (
                    "For now, `/photo ingest` only scans folders under the El Roi "
                    f"documents root:\n`{root}`\n\n"
                    "Move or copy the photos into that folder, then run "
                    "`/photo ingest <folder-name>`."
                )
            target = root / target
        else:
            base = root if project in {None, projects.ALL} else root / project
            target = base / candidate
    return target.resolve(), ""


def _indexed_photos(project: str = None, limit: int = 12) -> list[dict]:
    from rag import collection
    data = collection.get(include=["metadatas", "documents"])
    rows, seen = [], set()
    active_project = None if project in {None, projects.ALL} else project
    for meta, doc in zip(data.get("metadatas") or [], data.get("documents") or []):
        source = meta.get("source") or ""
        if not source or source in seen:
            continue
        if active_project and meta.get("project") != active_project:
            continue
        if (meta.get("source_ext") or Path(source).suffix.lower()) not in PHOTO_EXTENSIONS:
            continue
        seen.add(source)
        rows.append({
            "source": source,
            "filename": meta.get("filename") or Path(source).name,
            "project": meta.get("project") or projects.UNFILED,
            "description": " ".join((doc or "").split())[:220],
            "text": doc or "",
            "chunks": meta.get("chunk_count") or 1,
        })
        if len(rows) >= limit:
            break
    return rows


def _list_indexed_photos(project: str = None) -> str:
    photos = _indexed_photos(project=project)
    scope = "all projects" if project == projects.ALL else (project or projects.UNFILED)
    if not photos:
        root = _documents_root()
        return (
            f"I don’t see indexed photos in `{scope}` yet.\n\n"
            f"Put photos under `{root}` and run `/photo ingest`, or use "
            "`/photo ingest <folder-name>` after creating a photo folder there."
        )
    lines = [f"Indexed photos in `{scope}`:"]
    for item in photos:
        detail = f" - {item['description']}" if item["description"] else ""
        lines.append(f"- `{item['source']}`{detail}")
    return "\n".join(lines)


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", text or "")
        if token.lower() not in {
            "photo", "image", "edit", "adjust", "analyze", "the", "and",
            "for", "with", "that", "this", "please",
        }
    }


def _matching_photos(query: str, project: str = None, limit: int = 3) -> list[dict]:
    photos = _indexed_photos(project=project, limit=200)
    terms = _tokens(query)
    if not terms:
        return photos[:limit]

    scored = []
    for item in photos:
        haystack = " ".join([
            item.get("source", ""),
            item.get("filename", ""),
            item.get("description", ""),
            item.get("text", ""),
        ]).lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["source"]))
    return [item for _, item in scored[:limit]]


def _infer_subject(text: str) -> str:
    lower = (text or "").lower()
    subjects = [
        ("person or portrait subject", ("person", "portrait", "face", "man", "woman", "child", "people")),
        ("animal subject", ("animal", "bird", "dog", "cat", "horse")),
        ("building or architectural subject", ("building", "architecture", "house", "church", "room", "interior")),
        ("landscape subject", ("landscape", "mountain", "sky", "sunset", "water", "tree", "forest", "beach")),
        ("product or object subject", ("product", "object", "bottle", "phone", "watch", "food", "flower")),
        ("document/text subject", ("text", "page", "document", "sign", "poster", "label")),
    ]
    for label, terms in subjects:
        if any(term in lower for term in terms):
            return label
    return "primary visual subject"


def _composition_notes(text: str) -> list[str]:
    lower = (text or "").lower()
    notes = [
        "Check the strongest subject point and crop so it lands near a third-line intersection.",
        "Preserve enough negative space in the direction the subject faces or moves.",
    ]
    if any(term in lower for term in ("horizon", "sky", "landscape", "water")):
        notes.append("Level the horizon and avoid splitting the frame exactly in half unless symmetry is intentional.")
    if any(term in lower for term in ("face", "portrait", "person", "people")):
        notes.append("Place the eyes near the upper third and remove excess headroom.")
    if any(term in lower for term in ("center", "centered", "symmetrical", "symmetry")):
        notes.append("Keep centered framing only if symmetry is the point; otherwise test a rule-of-thirds crop.")
    return notes


def _lighting_notes(text: str) -> list[str]:
    lower = (text or "").lower()
    notes = []
    if any(term in lower for term in ("dark", "shadow", "underexposed", "dim")):
        notes.append("Lift exposure and shadows gently while keeping blacks anchored.")
    else:
        notes.append("Balance exposure first, then use contrast rather than global brightness for shape.")
    if any(term in lower for term in ("bright", "highlight", "sun", "sky", "white", "overexposed")):
        notes.append("Pull highlights down before raising overall exposure.")
    notes.append("Use local dodging on the subject and subtle burning around frame edges.")
    return notes


def _separation_notes(text: str) -> list[str]:
    lower = (text or "").lower()
    notes = [
        "Create a subject mask, then add a small exposure/texture lift to the subject.",
        "Create an inverse background mask and slightly reduce clarity, saturation, and exposure.",
    ]
    if any(term in lower for term in ("busy", "crowd", "clutter", "background", "distracting")):
        notes.append("Reduce background distractions before increasing subject contrast.")
    if any(term in lower for term in ("face", "portrait", "person", "people")):
        notes.append("Protect skin tones; avoid sharpening pores or pushing orange saturation.")
    return notes


def _recipe_for_photo(item: dict, request: str = "") -> str:
    text = " ".join([item.get("description", ""), item.get("text", "")])
    subject = _infer_subject(text)
    lighting_notes = _lighting_notes(text)
    lines = [
        f"Photo: `{item.get('source')}`",
        f"Likely subject: {subject}",
        "",
        "Composition",
    ]
    lines.extend(f"- {note}" for note in _composition_notes(text))
    lines.extend(["", "Lighting"])
    lines.extend(f"- {note}" for note in lighting_notes)
    lines.extend(["", "Subject Separation"])
    lines.extend(f"- {note}" for note in _separation_notes(text))
    lines.extend([
        "",
        "Suggested Edit Recipe",
        "- Crop/straighten: test rule-of-thirds crop; keep the subject's important edge intact.",
        "- Global tone: exposure +0.10 to +0.35, highlights -10 to -35, shadows +10 to +30, contrast +5 to +15.",
        "- Subject mask: exposure +0.10 to +0.25, texture +5 to +12, clarity +3 to +8.",
        "- Background mask: exposure -0.10 to -0.30, saturation -5 to -15, clarity -5 to -15.",
        "- Finish: check edges at 100%, then compare before/after for natural separation.",
    ])
    if request:
        lines.extend(["", f"User goal: {request}"])
    return "\n".join(lines)


def _answer_photo_analyze(query: str, project: str = None) -> dict:
    matches = _matching_photos(query, project=project, limit=3)
    if not matches:
        return {
            "text": (
                "I don’t see indexed photos to analyze yet. Upload photos, then "
                "run `/photo ingest Uploaded Photos`; or put photos under the "
                "project documents folder and run `/photo ingest`."
            ),
            "evidence": {},
            "grounded": False,
            "metrics": {"route": "photo_command", "photo_mode": True, "action": "analyze"},
        }

    text = "Photo analysis:\n\n" + "\n\n---\n\n".join(
        _recipe_for_photo(item, request=query) for item in matches)
    return {
        "text": text,
        "evidence": {},
        "grounded": True,
        "metrics": {
            "route": "photo_command", "photo_mode": True,
            "action": "analyze", "matches": len(matches),
        },
    }


def _photo_help(project: str = None) -> str:
    root = _documents_root()
    default_folder = root / (project if project and project != projects.ALL else "Photos")
    return (
        "Photo mode is on. Send photo requests without typing `/photo` each time.\n\n"
        "Try:\n"
        "- `/photo ingest` to scan photos already under the documents root\n"
        "- `/photo ingest <folder-name>` to create or scan a specific folder there\n"
        "- `/photo list` to show indexed photos\n"
        "- `/photo analyze <filename or description>` to review subject, framing, lighting, and separation\n"
        "- `/photo edit <filename or description>: <instructions>` to draft an edit plan\n"
        "- `/photo off` or `/exit photo` to leave photo mode\n\n"
        f"Default photo folder: `{default_folder}`"
    )


def _answer_photo_edit(query: str, project: str = None) -> dict:
    matches = _matching_photos(query, project=project, limit=3)
    if matches:
        text = "Photo edit recipe:\n\n" + "\n\n---\n\n".join(
            _recipe_for_photo(item, request=query) for item in matches)
    else:
        text = (
            "I can draft the edit recipe, but I don’t see a matching indexed "
            "photo yet. Run `/photo list` to see what El Roi can see, or "
            "`/photo ingest` after adding photos.\n\n"
            f"Requested edit: {query.strip() or '(no edit instructions provided)'}"
        )
    return {
        "text": text,
        "evidence": {},
        "grounded": bool(matches),
        "metrics": {
            "route": "photo_command", "photo_mode": True,
            "action": "edit", "matches": len(matches),
        },
    }


def _answer_photo_command(question: str, project: str = None) -> dict:
    if not _is_photo_command(question):
        return None

    query = _photo_command_query(question)
    lower = query.lower()
    if not query or lower in {"on", "start", "help", "?"}:
        return {
            "text": _photo_help(project=project),
            "evidence": {},
            "grounded": False,
            "metrics": {"route": "photo_command", "photo_mode": True},
        }
    if _is_photo_mode_exit(question):
        return _photo_mode_exit_response()

    if lower in {"list", "photos", "status", "indexed", "library"}:
        return {
            "text": _list_indexed_photos(project=project),
            "evidence": {},
            "grounded": True,
            "metrics": {"route": "photo_command", "photo_mode": True, "action": "list"},
        }

    ingest = re.match(r"^(?:ingest|import|scan|rescan)\b\s*(.*)$", query, re.I | re.S)
    if ingest:
        target, error = _resolve_ingest_target(ingest.group(1), project=project)
        if error:
            return {
                "text": error,
                "evidence": {},
                "grounded": False,
                "metrics": {
                    "route": "photo_command", "photo_mode": True,
                    "action": "ingest", "blocked": True,
                },
            }
        target.mkdir(parents=True, exist_ok=True)
        from rag import scan_documents
        summary = scan_documents(verbose=False)
        text = (
            f"Photo ingest is complete. Folder ready: `{target}`\n\n"
            f"Indexed {len(summary['new'])} new, {len(summary['updated'])} updated, "
            f"{len(summary['removed'])} removed, {len(summary['unchanged'])} unchanged."
        )
        return {
            "text": text,
            "evidence": {},
            "grounded": True,
            "metrics": {"route": "photo_command", "photo_mode": True, "action": "ingest"},
        }

    if lower.startswith(("analyze", "review", "critique", "compose", "composition")):
        cleaned = re.sub(
            r"^(?:analyze|review|critique|compose|composition)\b\s*:?",
            "",
            query,
            flags=re.I,
        )
        return _answer_photo_analyze(cleaned.strip(), project=project)

    if lower.startswith(("edit", "adjust", "retouch", "grade", "crop", "develop")):
        cleaned = re.sub(
            r"^(?:edit|adjust|retouch|grade|crop|develop)\b\s*:?",
            "",
            query,
            flags=re.I,
        )
        return _answer_photo_edit(cleaned.strip(), project=project)

    return _answer_photo_edit(query, project=project)


answer_photo_command = _answer_photo_command
is_photo_command = _is_photo_command
is_photo_mode_exit = _is_photo_mode_exit
photo_mode_active = _photo_mode_active
photo_mode_exit_response = _photo_mode_exit_response
photo_mode_question = _photo_mode_question

__all__ = [
    "answer_photo_command",
    "is_photo_command",
    "is_photo_mode_exit",
    "photo_mode_active",
    "photo_mode_exit_response",
    "photo_mode_question",
]
