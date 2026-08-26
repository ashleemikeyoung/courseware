"""
coder.py -- explicit local code-writing action for Ask.

The normal Ask flow answers questions. This module is only for requests that
clearly ask the app to edit code, and it keeps that power bounded:

- the request must name at least one target file
- target files must stay inside this app's repo
- runtime/data/document folders are blocked
- the model must return a unified diff
- the diff is checked before it is applied
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import requests

from config import BASE_DIR, CODER_MODEL, OLLAMA_URL


ALLOWED_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sql",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
BLOCKED_PREFIXES = {
    "backup/",
    "backups/",
    "chroma_db/",
    "documents/",
    "lancedb/",
    "memory/data/",
    "memory/keys/",
    "output/",
    "plans/",
    "projects/",
    "__pycache__/",
}
BLOCKED_FILES = {".env", ".env-bak", "app-restart.log", "benchmark.jsonl"}
PATH_RE = re.compile(
    r"(?<![\w./-])([A-Za-z0-9_./-]+\.(?:py|html|css|js|json|md|sql|toml|txt|ya?ml))"
)


class CoderError(RuntimeError):
    pass


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(BASE_DIR).as_posix()


def _is_blocked(rel: str) -> bool:
    return (
        rel in BLOCKED_FILES
        or rel.endswith(".pyc")
        or ".bak-" in rel
        or any(rel.startswith(prefix) for prefix in BLOCKED_PREFIXES)
    )


def _safe_target(rel: str) -> Path:
    candidate = (BASE_DIR / rel).resolve()
    try:
        safe_rel = candidate.relative_to(BASE_DIR).as_posix()
    except ValueError as exc:
        raise CoderError(f"Refusing to edit outside the app workspace: {rel}") from exc
    if _is_blocked(safe_rel):
        raise CoderError(f"Refusing to edit runtime or private data: {safe_rel}")
    if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        raise CoderError(f"Unsupported code-edit file type: {safe_rel}")
    return candidate


def _git_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--cached"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    files = []
    for line in result.stdout.splitlines():
        rel = line.strip()
        if not rel or _is_blocked(rel):
            continue
        if Path(rel).suffix.lower() in ALLOWED_SUFFIXES:
            files.append(rel)
    return sorted(dict.fromkeys(files))


def _mentioned_targets(request: str) -> list[Path]:
    known_files = _git_files()
    lowered = (request or "").lower()
    matches: list[str] = []

    for rel in known_files:
        name = Path(rel).name.lower()
        if rel.lower() in lowered or name in lowered:
            matches.append(rel)

    for raw in PATH_RE.findall(request or ""):
        raw = raw.strip("./")
        if raw:
            matches.append(raw)

    targets = []
    seen = set()
    for rel in matches:
        try:
            target = _safe_target(rel)
        except CoderError:
            raise
        safe_rel = _repo_rel(target)
        if safe_rel not in seen:
            seen.add(safe_rel)
            targets.append(target)
    return targets


def _read_context(targets: list[Path], max_chars_per_file: int = 24000) -> str:
    parts = []
    for target in targets:
        rel = _repo_rel(target)
        if target.exists():
            text = target.read_text(encoding="utf-8", errors="replace")
            if len(text) > max_chars_per_file:
                text = (
                    text[: max_chars_per_file // 2]
                    + "\n\n[... middle omitted for prompt size ...]\n\n"
                    + text[-max_chars_per_file // 2 :]
                )
            parts.append(f"--- {rel} ---\n{text}")
        else:
            parts.append(f"--- {rel} ---\n[NEW FILE]")
    return "\n\n".join(parts)


def _ask_coder_model(request: str, targets: list[Path]) -> tuple[str, dict]:
    target_list = "\n".join(f"- {_repo_rel(path)}" for path in targets)
    prompt = (
        "User request:\n"
        f"{request}\n\n"
        "Editable target files:\n"
        f"{target_list}\n\n"
        "Current file contents:\n"
        f"{_read_context(targets)}\n\n"
        "Return only a unified git diff. Do not include markdown fences, "
        "commentary, shell commands, or prose. The diff may modify only the "
        "editable target files listed above."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful coding agent. Produce minimal, correct "
                "unified git diffs. Preserve the existing style. Do not touch "
                "files the user did not name."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    start = time.time()
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": CODER_MODEL, "messages": messages, "stream": False},
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    metrics = {"model": CODER_MODEL, "elapsed_s": round(time.time() - start, 3)}
    for key in ("prompt_eval_count", "eval_count"):
        if key in data:
            metrics[key] = data[key]
    return data["message"]["content"], metrics


def _extract_diff(text: str) -> str:
    text = (text or "").strip()
    fence = re.search(r"```(?:diff|patch)?\s*(.*?)```", text, re.S | re.I)
    if fence:
        text = fence.group(1).strip()
    start = text.find("diff --git ")
    if start >= 0:
        return text[start:].strip() + "\n"
    if text.startswith("--- ") and "\n+++ " in text:
        return text + "\n"
    raise CoderError("The coder model did not return a unified diff.")


def _diff_paths(diff: str) -> set[str]:
    paths = set()
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            for item in parts[2:4]:
                if item.startswith("a/") or item.startswith("b/"):
                    paths.add(item[2:])
        elif line.startswith(("--- ", "+++ ")):
            item = line[4:].strip()
            if item != "/dev/null":
                if item.startswith("a/") or item.startswith("b/"):
                    item = item[2:]
                paths.add(item)
    return paths


def _validate_diff(diff: str, targets: list[Path]) -> list[str]:
    allowed = {_repo_rel(path) for path in targets}
    touched = _diff_paths(diff)
    if not touched:
        raise CoderError("The diff did not identify any files to change.")
    extra = sorted(path for path in touched if path not in allowed)
    if extra:
        raise CoderError(
            "The coder model tried to edit files that were not named: "
            + ", ".join(extra)
        )
    for rel in touched:
        _safe_target(rel)
    return sorted(touched)


def _apply_diff(diff: str) -> None:
    check = subprocess.run(
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        cwd=BASE_DIR,
        input=diff,
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode != 0:
        raise CoderError((check.stderr or check.stdout or "Patch check failed").strip())
    applied = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=BASE_DIR,
        input=diff,
        capture_output=True,
        text=True,
        check=False,
    )
    if applied.returncode != 0:
        raise CoderError((applied.stderr or applied.stdout or "Patch apply failed").strip())


def write_code(request: str) -> dict:
    targets = _mentioned_targets(request)
    if not targets:
        return {
            "applied": False,
            "needs_target_files": True,
            "message": (
                "I can write code for that, but I need the target file name(s) "
                "in the request first. Example: code: update ask.py to ..."
            ),
        }

    raw, metrics = _ask_coder_model(request, targets)
    diff = _extract_diff(raw)
    touched = _validate_diff(diff, targets)
    _apply_diff(diff)
    return {
        "applied": True,
        "files": touched,
        "diff": diff,
        "metrics": metrics,
    }
