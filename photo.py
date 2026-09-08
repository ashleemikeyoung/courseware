"""Photo helpers for Ask mode.

This module owns explicit /photo handling and photo-mode state. The first
version leans on the existing El Roi document scanner: photos placed under the
documents root can already be indexed for OCR, local vision descriptions, and
RAW metadata. The command layer makes that workflow deliberate.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from urllib.parse import quote_plus

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
RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".raf", ".rw2"}


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


def _last_photo_source(messages: list) -> str:
    for message in reversed(messages or []):
        if message.get("role") != "assistant":
            continue
        text = message.get("content") or ""
        match = re.search(r"^Photo:\s+`([^`]+)`", text, re.MULTILINE)
        if match:
            return match.group(1)
    return ""


def _photo_mode_question(question: str, messages: list = None) -> str:
    if _is_photo_command(question):
        return question
    q = (question or "").strip()
    lower = q.lower()
    if lower.startswith(("list", "photos", "status", "indexed", "library", "ingest", "import", "scan", "rescan")):
        return f"/photo {q}".strip()
    source = _last_photo_source(messages or [])
    if source:
        return f"/photo edit {source}: {q}".strip()
    return f"/photo {q}".strip()


def _documents_root() -> Path:
    from rag import DOCUMENTS_FOLDER
    return Path(DOCUMENTS_FOLDER).resolve()


def _source_path(source: str) -> Path:
    return (_documents_root() / (source or "")).resolve()


def _project_path(rel_path: str) -> Path:
    return (projects.PROJECTS_ROOT / (rel_path or "")).resolve()


def _preview_root(project: str = None) -> Path:
    name = projects.safe(project or projects.UNFILED)
    return projects.PROJECTS_ROOT / name / "photo-previews"


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
    strategy = _composition_strategy(" ".join([request, text]))
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
        lines.extend([
            "",
            "El Roi Verb Plan",
            f"- Framing choice: {strategy}.",
        ])
        lines.extend(f"- {line}" for line in _verb_plan_lines(request))
        lines.append(f"User goal: {request}")
    return "\n".join(lines)


def _open_photo_preview(path: Path):
    from PIL import Image

    suffix = path.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        return Image.fromarray(rgb).convert("RGB")
    return Image.open(path).convert("RGB")


def _fit_preview(image, max_edge: int = 1800):
    image = image.copy()
    image.thumbnail((max_edge, max_edge))
    return image


def _warm_image(image, amount: float = 1.04):
    from PIL import Image

    r, g, b = image.split()
    r = r.point(lambda value: min(255, int(value * amount)))
    b = b.point(lambda value: max(0, int(value / amount)))
    return Image.merge("RGB", (r, g, b))


def _crop_preview(image, instructions: str):
    lower = (instructions or "").lower()
    if not any(term in lower for term in (
        "crop", "frame", "framing", "tighter", "4:5", "5:4", "square",
        "optimize frame", "optimal frame", "autoframe", "rule of thirds",
        "thirds", "leading lines",
    )):
        return image
    w, h = image.size
    if "square" in lower:
        target_ratio = 1.0
    elif "4:5" in lower:
        target_ratio = 4 / 5
    elif "5:4" in lower:
        target_ratio = 5 / 4
    else:
        target_ratio = w / h
    if any(term in lower for term in ("tighter", "optimize frame", "optimal frame", "autoframe", "leading lines", "rule of thirds", "thirds")):
        w2, h2 = int(w * 0.80), int(h * 0.80)
    elif abs((w / h) - target_ratio) < 0.03:
        return image
    else:
        w2, h2 = w, h
    if w2 / h2 > target_ratio:
        w2 = int(h2 * target_ratio)
    else:
        h2 = int(w2 / target_ratio)
    left = max(0, (w - w2) // 2)
    top = max(0, (h - h2) // 2)
    return image.crop((left, top, left + w2, top + h2))


def _add_subject_separation(image, instructions: str):
    lower = (instructions or "").lower()
    if not any(term in lower for term in ("separation", "subject", "background", "pop", "portrait")):
        return image

    from PIL import ImageEnhance, ImageFilter, Image

    base = image.convert("RGB")
    background = ImageEnhance.Color(base).enhance(0.72)
    background = ImageEnhance.Contrast(background).enhance(0.88)
    background = ImageEnhance.Brightness(background).enhance(0.78)
    background = background.filter(ImageFilter.GaussianBlur(radius=2.4))

    subject = ImageEnhance.Sharpness(base).enhance(1.28)
    subject = ImageEnhance.Brightness(subject).enhance(1.12)
    subject = ImageEnhance.Contrast(subject).enhance(1.12)

    w, h = base.size
    mask = Image.new("L", (w, h), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    pad_x, pad_y = int(w * 0.22), int(h * 0.12)
    draw.ellipse((pad_x, pad_y, w - pad_x, h - pad_y), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(12, min(w, h) // 14)))
    return Image.composite(subject, background, mask)


def _add_foreground_blur(image, amount: float = 2.6):
    from PIL import Image, ImageDraw, ImageFilter

    base = image.convert("RGB")
    blurred = base.filter(ImageFilter.GaussianBlur(radius=amount))
    w, h = base.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((0, int(h * 0.58), w, h), fill=210)
    draw.rectangle((0, int(h * 0.72), w, h), fill=255)
    draw.rectangle((0, 0, int(w * 0.10), h), fill=65)
    draw.rectangle((int(w * 0.90), 0, w, h), fill=65)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(18, min(w, h) // 18)))
    return Image.composite(blurred, base, mask)


def _center_subject_mask(size: tuple[int, int]):
    from PIL import Image, ImageDraw, ImageFilter

    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    pad_x, pad_y = int(w * 0.22), int(h * 0.12)
    draw.ellipse((pad_x, pad_y, w - pad_x, h - pad_y), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=max(12, min(w, h) // 14)))


def _apply_background_adjustment(image, darken: float = 0.0, blur: float = 0.0):
    from PIL import Image, ImageEnhance, ImageFilter

    base = image.convert("RGB")
    background = base
    if darken:
        background = ImageEnhance.Brightness(background).enhance(
            max(0.35, 1 - max(0, darken) / 100))
        background = ImageEnhance.Color(background).enhance(
            max(0.35, 1 - max(0, darken) / 160))
    if blur:
        background = background.filter(ImageFilter.GaussianBlur(radius=max(0, blur) / 2))
    subject = ImageEnhance.Sharpness(base).enhance(1.08)
    mask = _center_subject_mask(base.size)
    return Image.composite(subject, background, mask)


def _preview_action_lines(instructions: str) -> list[str]:
    lower = (instructions or "").lower()
    actions = [
        "Applied a visible baseline enhancement: stronger contrast, color, brightness, and sharpening.",
    ]
    if any(term in lower for term in ("crop", "frame", "framing", "tighter", "4:5", "5:4", "square", "optimize frame", "autoframe", "rule of thirds", "thirds", "leading lines")):
        actions.append(f"Optimized the frame using {_composition_strategy(instructions)}.")
    if "blur foreground" in lower or "foreground blur" in lower:
        actions.append("Softened the foreground while keeping the central subject area clearer.")
    if "blur background" in lower or "background blur" in lower:
        actions.append("Softened and darkened the background to increase subject separation.")
    if any(term in lower for term in ("bright", "brighter", "lift", "exposure", "light")):
        actions.append("Raised preview brightness.")
    if any(term in lower for term in ("dark", "moody", "deeper")):
        actions.append("Darkened the preview.")
    if any(term in lower for term in ("warm", "warmer", "golden")):
        actions.append("Warmed the color balance.")
    if any(term in lower for term in ("vibrant", "color", "saturation")):
        actions.append("Increased color saturation.")
    if any(term in lower for term in ("separation", "subject", "background", "pop", "portrait")):
        actions.append(
            "Applied a center-weighted subject separation preview: brighter/sharper center, darker/softer background."
        )
    elif not any(term in lower for term in ("flat", "neutral", "minimal", "subtle", "only crop", "crop only")):
        actions.append(
            "Applied a default center-weighted subject pop so the modified preview is visibly different."
        )
    return actions


def _composition_strategy(instructions: str) -> str:
    lower = (instructions or "").lower()
    if "leading line" in lower or "lines" in lower:
        return "leading lines with a gentle crop"
    if "rule of thirds" in lower or "thirds" in lower:
        return "rule-of-thirds placement"
    if any(term in lower for term in ("table", "food", "product", "box", "object", "chocolate")):
        return "a product-style crop that balances repeated shapes and keeps the strongest box near a third line"
    return "the strongest composition between rule of thirds, leading lines, and subject balance"


def _verb_plan_lines(instructions: str) -> list[str]:
    lower = (instructions or "").lower()
    lines = []
    if "blur foreground" in lower or "foreground blur" in lower:
        lines.append("Foreground: soften near-camera distractions without flattening the subject.")
    if "blur background" in lower or "background blur" in lower:
        lines.append("Background: reduce detail and brightness behind the subject.")
    if any(term in lower for term in ("optimize frame", "optimal frame", "autoframe", "frame", "composition", "crop")):
        lines.append("Frame: choose the crop by composition rather than applying a fixed crop amount.")
    if any(term in lower for term in ("subject", "separation", "pop")):
        lines.append("Subject: keep the main subject brighter and crisper than surrounding areas.")
    if not lines:
        lines.append("Tone: preserve the automatic El Roi look and make only goal-directed changes.")
    return lines


def _apply_preview_edits(image, instructions: str):
    from PIL import ImageEnhance, ImageFilter, ImageOps

    lower = (instructions or "").lower()
    edited = _crop_preview(image, instructions)
    edited = ImageOps.autocontrast(edited, cutoff=1.5)
    edited = ImageEnhance.Contrast(edited).enhance(1.22)
    edited = ImageEnhance.Color(edited).enhance(1.14)
    edited = ImageEnhance.Brightness(edited).enhance(1.06)
    edited = ImageEnhance.Sharpness(edited).enhance(1.18)
    if any(term in lower for term in ("bright", "brighter", "lift", "exposure", "light")):
        edited = ImageEnhance.Brightness(edited).enhance(1.18)
    if any(term in lower for term in ("dark", "moody", "deeper")):
        edited = ImageEnhance.Brightness(edited).enhance(0.85)
    if any(term in lower for term in ("contrast", "pop", "separation", "subject")):
        edited = ImageEnhance.Contrast(edited).enhance(1.16)
    if any(term in lower for term in ("warm", "warmer", "golden")):
        edited = _warm_image(edited, amount=1.12)
    if any(term in lower for term in ("vibrant", "color", "saturation")):
        edited = ImageEnhance.Color(edited).enhance(1.22)
    if any(term in lower for term in ("flat", "neutral", "minimal", "subtle", "only crop", "crop only")):
        return edited.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=3))
    edited = _add_subject_separation(edited, instructions + " subject background")
    if "blur foreground" in lower or "foreground blur" in lower:
        edited = _add_foreground_blur(edited)
    return edited


def _apply_slider_adjustments(image, adjustments: dict = None):
    from PIL import ImageEnhance, ImageFilter

    adjustments = adjustments or {}

    def number(key: str, default: float = 0.0) -> float:
        try:
            return float(adjustments.get(key, default))
        except (TypeError, ValueError):
            return default

    brightness = number("brightness")
    contrast = number("contrast")
    warmth = number("warmth")
    saturation = number("saturation")
    background = number("background")
    blur = max(0.0, number("blur"))
    crop = max(0.0, min(30.0, number("crop")))

    edited = image.convert("RGB")
    if crop:
        w, h = edited.size
        inset_x = int(w * crop / 200)
        inset_y = int(h * crop / 200)
        edited = edited.crop((inset_x, inset_y, w - inset_x, h - inset_y))
    edited = ImageEnhance.Brightness(edited).enhance(1 + brightness / 75)
    edited = ImageEnhance.Contrast(edited).enhance(1 + contrast / 70)
    edited = ImageEnhance.Color(edited).enhance(1 + saturation / 70)
    if warmth:
        edited = _warm_image(edited, amount=1 + warmth / 120)
    if background or blur:
        edited = _apply_background_adjustment(edited, darken=background, blur=blur)
    return edited


def _preview_delta(original, edited) -> float:
    from PIL import ImageChops, ImageStat

    diff = ImageChops.difference(original.resize(edited.size), edited)
    return max(ImageStat.Stat(diff).mean)


def generate_adjusted_preview(source: str, project: str = None,
                              adjustments: dict = None,
                              base_preview: str = "") -> tuple[dict, float, str]:
    source = source or ""
    path = _source_path(source)
    root = _documents_root()
    if not path.exists() or not path.is_relative_to(root):
        return {}, 0.0, f"Source file is not available: {source}"

    out_dir = _preview_root(project)
    out_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    edited = out_dir / f"modified-preview-{token}.jpg"

    image = _fit_preview(_open_photo_preview(path))
    base_image = None
    base_path = _project_path(base_preview) if base_preview else None
    if base_path and base_path.exists() and base_path.is_relative_to(projects.PROJECTS_ROOT):
        base_image = _open_photo_preview(base_path)
    if base_image is None:
        base_image = _apply_preview_edits(image, "subject background pop warm vibrant color contrast")
    edited_image = _apply_slider_adjustments(base_image, adjustments)
    delta = _preview_delta(image, edited_image)
    edited_image.save(edited, "JPEG", quality=92)

    try:
        edited_rel = str(edited.relative_to(projects.PROJECTS_ROOT))
    except ValueError:
        return {}, 0.0, "Preview output path is outside the project workspace."
    return {
        "filename": "modified-preview.jpg",
        "content_type": "image/jpeg",
        "size": edited.stat().st_size,
        "image": True,
        "url": f"/api/photo/file?kind=preview&path={quote_plus(edited_rel)}",
        "photo_preview": True,
        "role": "modified",
        "source": source,
        "project": projects.safe(project or projects.UNFILED),
        "preview_path": edited_rel,
    }, delta, ""


def _preview_pair(item: dict, instructions: str, project: str = None) -> tuple[list, str, float]:
    source = item.get("source") or ""
    path = _source_path(source)
    root = _documents_root()
    if not path.exists() or not path.is_relative_to(root):
        return [], f"Source file is not available: {source}", 0.0

    out_dir = _preview_root(project or item.get("project"))
    out_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    original = out_dir / f"original-preview-{token}.jpg"
    edited = out_dir / f"modified-preview-{token}.jpg"

    image = _fit_preview(_open_photo_preview(path))
    edited_image = _apply_preview_edits(image, instructions)
    delta = _preview_delta(image, edited_image)
    image.save(original, "JPEG", quality=92)
    edited_image.save(edited, "JPEG", quality=92)

    try:
        original_rel = str(original.relative_to(projects.PROJECTS_ROOT))
        edited_rel = str(edited.relative_to(projects.PROJECTS_ROOT))
    except ValueError:
        return [], "Preview output path is outside the project workspace.", 0.0
    return [
        {
            "filename": "original-preview.jpg",
            "content_type": "image/jpeg",
            "size": original.stat().st_size,
            "image": True,
            "url": f"/api/photo/file?kind=preview&path={quote_plus(original_rel)}",
            "photo_preview": True,
            "role": "original",
            "source": source,
            "project": projects.safe(project or item.get("project") or projects.UNFILED),
            "preview_path": original_rel,
        },
        {
            "filename": "modified-preview.jpg",
            "content_type": "image/jpeg",
            "size": edited.stat().st_size,
            "image": True,
            "url": f"/api/photo/file?kind=preview&path={quote_plus(edited_rel)}",
            "photo_preview": True,
            "role": "modified",
            "source": source,
            "project": projects.safe(project or item.get("project") or projects.UNFILED),
            "preview_path": edited_rel,
        },
    ], "", delta


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
        "- Verb examples: `blur background`, `blur foreground`, `optimize frame`, `use leading lines`, `use rule of thirds`, `improve subject separation`\n"
        "- `/photo off` or `/exit photo` to leave photo mode\n\n"
        f"Default photo folder: `{default_folder}`"
    )


def _answer_photo_edit(query: str, project: str = None) -> dict:
    matches = _matching_photos(query, project=project, limit=3)
    attachments, preview_errors = [], []
    if matches:
        text = "Photo edit recipe:\n\n" + "\n\n---\n\n".join(
            _recipe_for_photo(item, request=query) for item in matches)
        preview_delta = 0.0
        for item in matches[:1]:
            pair, error, delta = _preview_pair(item, query, project=project)
            attachments.extend(pair)
            preview_delta = max(preview_delta, delta)
            if error:
                preview_errors.append(error)
        if attachments:
            text += (
                "\n\nApplied Preview Changes\n"
                + "\n".join(f"- {line}" for line in _preview_action_lines(query))
                + f"\n- Preview change strength: {preview_delta:.1f}/255 average channel shift."
                + "\n\nI generated a before/after preview pair below."
            )
        elif preview_errors:
            text += "\n\nPreview note: " + "; ".join(preview_errors)
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
        "attachments": attachments,
        "showAttachments": bool(attachments),
        "metrics": {
            "route": "photo_command", "photo_mode": True,
            "action": "edit", "matches": len(matches),
            "previews": len(attachments),
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

    if lower.startswith((
        "edit", "adjust", "retouch", "grade", "crop", "develop",
        "blur", "soften", "sharpen", "optimize", "frame", "reframe",
        "use leading", "leading", "rule of thirds", "thirds", "improve",
    )):
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
