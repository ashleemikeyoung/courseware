"""
redactor.py -- create redacted copies of indexed local documents.

This module is deliberately file-oriented. pii.py redacts strings for display
or generated output; this patches a real DOCX package and writes a separate
redacted copy under output/redactions so the original document library stays
unchanged.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from config import BASE_DIR
import pii
import summarize


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
DEFAULT_MASK_CHAR = "X"
SOURCE_TERM_STOPWORDS = {
    "affidavit",
    "case",
    "casesummary",
    "claim",
    "complaint",
    "coverpage",
    "delivery",
    "docx",
    "draft",
    "exhibit",
    "file",
    "final",
    "pdf",
    "replevin",
    "summary",
}


class RedactionError(RuntimeError):
    pass


class UnsupportedRedactionError(RedactionError):
    pass


@dataclass(frozen=True)
class RedactionRule:
    pattern: re.Pattern[str]
    label: str


def _unzip_docx(docx_path: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx_path, "r") as z:
        z.extractall(out_dir)


def _zip_docx(in_dir: Path, out_docx_path: Path) -> None:
    out_docx_path.parent.mkdir(parents=True, exist_ok=True)
    if out_docx_path.exists():
        out_docx_path.unlink()
    with zipfile.ZipFile(out_docx_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(in_dir):
            for filename in files:
                abs_path = Path(root) / filename
                rel_path = abs_path.relative_to(in_dir)
                z.write(abs_path, rel_path.as_posix())


def _iter_word_parts(unzipped: Path, include_comments: bool) -> list[Path]:
    word = unzipped / "word"
    parts: list[Path] = []
    for name in ["document.xml", "footnotes.xml", "endnotes.xml"]:
        part = word / name
        if part.exists():
            parts.append(part)
    parts.extend(sorted(word.glob("header*.xml")))
    parts.extend(sorted(word.glob("footer*.xml")))
    if include_comments:
        comments = word / "comments.xml"
        if comments.exists():
            parts.append(comments)
    return parts


def _text_nodes(paragraph: etree._Element) -> list[etree._Element]:
    return list(paragraph.xpath(".//w:t", namespaces=NS))


def _materialize_spaces(node: etree._Element) -> None:
    text = node.text or ""
    if text.startswith(" ") or text.endswith(" "):
        node.set(XML_SPACE, "preserve")


def _merge_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    spans = [(s, e, label) for s, e, label in spans if e > s]
    spans.sort(key=lambda item: (item[0], item[1]))
    merged: list[tuple[int, int, str]] = []
    for start, end, label in spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end, label))
            continue
        prev_start, prev_end, prev_label = merged[-1]
        merged[-1] = (prev_start, max(prev_end, end), prev_label or label)
    return merged


def _mask_text(text: str, spans: list[tuple[int, int, str]],
               mask_char: str = DEFAULT_MASK_CHAR) -> str:
    out = list(text)
    for start, end, _label in reversed(_merge_spans(spans)):
        out[start:end] = list(mask_char * (end - start))
    return "".join(out)


def _pii_spans(text: str, score_threshold: float) -> list[tuple[int, int, str]]:
    try:
        findings = pii.analyze_text(text, score_threshold=score_threshold)
    except Exception:
        return []
    return [(f["start"], f["end"], f["entity_type"]) for f in findings]


def _regex_rules(source: str | None, extra_terms: list[str] | None) -> list[RedactionRule]:
    rules = [
        RedactionRule(
            re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
            "email",
        ),
        RedactionRule(
            re.compile(
                r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?)\d{3}[\s.-]?\d{4}",
                re.I,
            ),
            "phone",
        ),
        RedactionRule(re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "ssn"),
        RedactionRule(re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "card_like_number"),
        RedactionRule(
            re.compile(
                r"\b\d{1,6}\s+(?:[A-Z][A-Za-z0-9'.-]*\s+){1,6}"
                r"(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|"
                r"Drive|Dr\.?|Lane|Ln\.?|Court|Ct\.?|Circle|Cir\.?|Way|"
                r"Highway|Hwy\.?|Trail|Trl\.?|Place|Pl\.?)\b"
            ),
            "street_address",
        ),
        RedactionRule(
            re.compile(
                r"\b[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,3},\s+"
                r"[A-Z][A-Za-z'.-]+(?:\s+\d{5}(?:-\d{4})?)?\b"
            ),
            "city_state_zip",
        ),
        RedactionRule(re.compile(r"\b\d{5}(?:-\d{4})?\b"), "zip_code"),
    ]

    tokens: set[str] = set()
    for value in [source or "", *(extra_terms or [])]:
        for token in re.split(r"[^A-Za-z]+", value):
            if len(token) >= 4 and token.lower() not in SOURCE_TERM_STOPWORDS:
                tokens.add(token)
    for token in sorted(tokens, key=len, reverse=True):
        rules.append(
            RedactionRule(
                re.compile(rf"\b{re.escape(token)}\b", re.I),
                "source_term",
            )
        )
    return rules


def _regex_spans(text: str, rules: list[RedactionRule]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for rule in rules:
        for match in rule.pattern.finditer(text):
            spans.append((match.start(), match.end(), rule.label))
    return spans


def _apply_redactions_to_paragraph(
    paragraph: etree._Element,
    rules: list[RedactionRule],
    score_threshold: float,
    mask_char: str,
) -> int:
    nodes = _text_nodes(paragraph)
    if not nodes:
        return 0
    original_segments = [node.text or "" for node in nodes]
    full_text = "".join(original_segments)
    if not full_text:
        return 0

    spans = _pii_spans(full_text, score_threshold=score_threshold)
    spans.extend(_regex_spans(full_text, rules))
    chosen = _merge_spans(spans)
    if not chosen:
        return 0

    redacted = _mask_text(full_text, chosen, mask_char=mask_char)
    index = 0
    for node, segment in zip(nodes, original_segments, strict=True):
        take = len(segment)
        node.text = redacted[index:index + take]
        _materialize_spaces(node)
        index += take
    return len(chosen)


def _redacted_output_path(source: str) -> Path:
    source_path = Path(source)
    safe_parts = [part for part in source_path.parts if part not in {"", ".", ".."}]
    rel = Path(*safe_parts) if safe_parts else Path(source_path.name)
    return BASE_DIR / "output" / "redactions" / rel.with_name(
        f"{rel.stem}.redacted{rel.suffix}"
    )


def redact_docx_file(
    input_docx: Path,
    output_docx: Path,
    source: str | None = None,
    extra_terms: list[str] | None = None,
    include_comments: bool = True,
    mask_char: str = DEFAULT_MASK_CHAR,
    score_threshold: float = 0.45,
) -> dict:
    if input_docx.suffix.lower() != ".docx":
        raise UnsupportedRedactionError("Only .docx redaction is currently wired.")
    if not input_docx.exists():
        raise FileNotFoundError(input_docx)

    rules = _regex_rules(source=source, extra_terms=extra_terms)
    with tempfile.TemporaryDirectory(prefix="rag_docx_redact_") as td:
        tmp = Path(td)
        _unzip_docx(input_docx, tmp)
        stats = {
            "source": source,
            "input_path": str(input_docx),
            "output_path": str(output_docx),
            "parts_processed": 0,
            "paragraphs_touched": 0,
            "matches_redacted": 0,
        }
        for part in _iter_word_parts(tmp, include_comments=include_comments):
            parser = etree.XMLParser(remove_blank_text=False)
            tree = etree.parse(str(part), parser)
            root = tree.getroot()
            part_touched = 0
            part_matches = 0
            for paragraph in root.xpath(".//w:p", namespaces=NS):
                matches = _apply_redactions_to_paragraph(
                    paragraph,
                    rules=rules,
                    score_threshold=score_threshold,
                    mask_char=mask_char,
                )
                if matches:
                    part_touched += 1
                    part_matches += matches
            if part_matches:
                stats["paragraphs_touched"] += part_touched
                stats["matches_redacted"] += part_matches
            stats["parts_processed"] += 1
            tree.write(str(part), xml_declaration=True, encoding="UTF-8", standalone="yes")
        _zip_docx(tmp, output_docx)
        return stats


def redact_source(source: str, project: str = None, extra_terms: list[str] = None) -> dict:
    input_path = summarize.resolve_path(source)
    suffix = input_path.suffix.lower()
    if suffix != ".docx":
        raise UnsupportedRedactionError(
            f"{suffix or 'this file type'} redaction is not wired yet; use a .docx source."
        )
    output_path = _redacted_output_path(source)
    return redact_docx_file(
        input_docx=input_path,
        output_docx=output_path,
        source=source,
        extra_terms=extra_terms,
    )
