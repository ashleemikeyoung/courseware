"""
bench.py — controlled comparison of drafting models on your own documents.

The point of a benchmark is to vary one thing. So this holds everything constant
except the model:

  same outline          loaded from plan.json, not regenerated per model
  same retrieval        evidence gathered ONCE, before any model runs, and the
                        identical passages with identical [C] markers handed to
                        every candidate
  same prompts          same system prompt, same targets, same temperature
  same running notes    compressed by the SAME notes model for all candidates,
                        so a weak notes model cannot flatter its own drafter

That last one is subtle and matters. If each candidate compressed its own notes,
a model with poor summarization would poison its own later sections and you would
be measuring two things at once.

Usage:
  python bench.py                                   # interactive
  python bench.py --models qwen3:32b,gemma4:e4b     # named sweep
  python bench.py --sections 2                      # first 2 sections only, fast
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import writer
import quality
import projects
from writer import (
    CitationRegistry, gather_evidence, evidence_block, draft_section,
    compress_for_notes, preflight, available_models, _in, _confirm,
    NOTES_MODEL,
)

def bench_dir(project: str = None) -> Path:
    """History is per project. Comparing a model on a thesis against the same
    model on client contracts averages two different retrieval landscapes."""
    return projects.ensure(project or writer.CURRENT_PROJECT)["bench"]


# ---------------------------------------------------------------------------
# Shared evidence, gathered once
# ---------------------------------------------------------------------------

def prepare_evidence(outline: dict) -> tuple:
    """
    Retrieve for every section up front, into one shared registry.

    Doing this once rather than per model is what makes the comparison valid,
    and it also means a six-model sweep does one pass of embedding work instead
    of six.
    """
    registry = CitationRegistry()
    per_section = []
    for s in outline["sections"]:
        queries = s.get("queries") or [s.get("heading", "")]
        ev = gather_evidence(queries, registry, per_query=8, window=1)
        per_section.append(ev)
        print(f"  {s.get('heading', '')[:48]:<50} {len(ev)} passages")
    return registry, per_section


# ---------------------------------------------------------------------------
# One model, one full pass
# ---------------------------------------------------------------------------

def run_model(model: str, outline: dict, per_section: list,
              registry: CitationRegistry) -> dict:
    print(f"\n{'#' * 70}\n#  {model}\n{'#' * 70}")

    sections, notes, timings = [], "", []
    started = time.time()

    for i, spec in enumerate(outline["sections"]):
        heading = spec.get("heading", f"Section {i+1}")
        print(f"  [{i+1}/{len(outline['sections'])}] {heading} ... ",
              end="", flush=True)
        body, m = draft_section(outline, i, spec, notes, registry,
                                evidence=per_section[i], model=model,
                                echo=False)
        sections.append(body)
        timings.append(m)
        print(f"{m.get('elapsed_s', '?')}s  "
              f"{m.get('tokens_per_second', '?')} tok/s  "
              f"{len(body.split())}w")
        # Notes always come from the fixed notes model, never the candidate.
        notes += compress_for_notes(heading, body)

    elapsed = time.time() - started

    offered = {e.marker for ev in per_section for e in ev}
    evidence_text = "\n".join(e.text for ev in per_section for e in ev)
    target = sum(int(s.get("target_words", 700)) for s in outline["sections"])

    report = quality.score_document(
        sections, evidence_text,
        valid_markers={e.marker for e in registry.items},
        target_words=target, offered=offered,
    )

    tps = [t["tokens_per_second"] for t in timings if t.get("tokens_per_second")]
    report["speed"] = {
        "total_min": round(elapsed / 60, 2),
        "mean_tok_per_s": round(sum(tps) / len(tps), 1) if tps else None,
        "words_per_min": round(report["length"]["words"] / (elapsed / 60), 1),
    }
    report["model"] = model
    return {"report": report, "sections": sections}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def comparison_table(results: list) -> str:
    rows = sorted(results, key=lambda r: -r["report"]["score"])
    w = [24, 7, 5, 8, 8, 8, 7, 8, 7]
    head = ["model", "score", "fab", "grnd", "dist3", "xsect", "len", "tok/s", "min"]
    lines = [
        "  " + "".join(h.ljust(c) for h, c in zip(head, w)),
        "  " + "-" * sum(w),
    ]
    for r in rows:
        q = r["report"]
        tps = q["speed"]["mean_tok_per_s"]
        cells = [
            q["model"][:23],
            f"{q['score']:.1f}",
            str(q["citations"]["fabricated_count"]),
            f"{q['grounding']:.3f}",
            f"{q['distinct_3']:.3f}",
            f"{q['cross_section']['mean']:.3f}",
            f"{q['length']['ratio']:.2f}",
            f"{tps:.1f}" if tps else "-",
            f"{q['speed']['total_min']:.1f}",
        ]
        lines.append("  " + "".join(c.ljust(n) for c, n in zip(cells, w)))

    lines += [
        "",
        "  fab    fabricated citations, MUST be 0",
        "  grnd   5-gram overlap with the evidence, higher means less invention",
        "  dist3  distinct trigrams, below 0.75 means visible repetition",
        "  xsect  mean cross-section overlap, above 0.05 means it repeats itself",
        "  len    delivered vs target words, 1.0 is on the nose",
    ]

    for r in rows:
        q = r["report"]
        notes = []
        if q["citations"]["fabricated"]:
            notes.append(f"invented {', '.join(q['citations']['fabricated'][:5])}")
        if q["flags"]:
            notes.append(", ".join(q["flags"]))
        if q["cross_section"]["worst_pair"] and q["cross_section"]["max"] > 0.12:
            a, b = q["cross_section"]["worst_pair"]
            notes.append(f"sections {a} and {b} overlap heavily")
        if notes:
            lines.append(f"\n  {q['model']}: " + "; ".join(notes))

    return "\n".join(lines)


def save_run(results: list, outline: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = bench_dir() / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        slug = re.sub(r"[^a-z0-9]+", "-", r["report"]["model"].lower()).strip("-")
        (run_dir / f"{slug}.md").write_text(
            f"# {outline.get('title', '')}\n\n"
            f"*drafted by {r['report']['model']}*\n\n"
            + "\n\n".join(r["sections"])
        )

    (run_dir / "results.json").write_text(json.dumps({
        "timestamp": stamp,
        "title": outline.get("title"),
        "notes_model": NOTES_MODEL,
        "sections": len(outline["sections"]),
        "results": [r["report"] for r in results],
    }, indent=2))

    # Append to a rolling history so you can watch models across many topics,
    # not just within one sweep. One topic proves very little.
    history = bench_dir() / "history.jsonl"
    with history.open("a") as f:
        for r in results:
            f.write(json.dumps({"timestamp": stamp,
                                "title": outline.get("title"),
                                "project": writer.CURRENT_PROJECT,
                                **r["report"]}) + "\n")

    return run_dir


def show_history(last: int = 40):
    path = bench_dir() / "history.jsonl"
    if not path.exists():
        print("  No benchmark history yet.")
        return
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    agg = {}
    for r in rows:
        a = agg.setdefault(r["model"], {"n": 0, "score": 0.0, "fab": 0})
        a["n"] += 1
        a["score"] += r["score"]
        a["fab"] += r["citations"]["fabricated_count"]

    print(f"\n  Across {len(rows)} runs on "
          f"{len({r['title'] for r in rows})} topic(s):\n")
    print(f"  {'model':<24}{'runs':<7}{'avg score':<12}{'total fabrications'}")
    print("  " + "-" * 60)
    for model, a in sorted(agg.items(), key=lambda kv: -kv[1]["score"] / kv[1]["n"]):
        print(f"  {model[:23]:<24}{a['n']:<7}{a['score'] / a['n']:<12.1f}{a['fab']}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def choose_models() -> list:
    have = sorted(available_models())
    if not have:
        print("  Cannot reach Ollama.")
        return []
    print("\n  Local models:\n")
    for i, m in enumerate(have, 1):
        tag = "  <- cloud, skips your privacy boundary" if "cloud" in m else ""
        print(f"    {i:>2}. {m}{tag}")
    print("\n  Numbers to compare, comma separated. Example: 2,5,7")
    picked = _in("  > ")
    out = []
    for p in picked.split(","):
        p = p.strip()
        if p.isdigit() and 1 <= int(p) <= len(have):
            out.append(have[int(p) - 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma separated, else you get a picker")
    ap.add_argument("--plan", default="plan.json")
    ap.add_argument("--sections", type=int,
                    help="only draft the first N sections, for a fast sweep")
    ap.add_argument("--project", help="project folder under documents/")
    ap.add_argument("--history", action="store_true",
                    help="show aggregate results across past runs and exit")
    args = ap.parse_args()

    if args.project:
        writer.set_project(args.project)

    if args.history:
        show_history()
        return

    plan = Path(args.plan) if args.plan != "plan.json" else writer.plan_path()
    if not plan.exists():
        print(f"  No {plan}. Run writer.py first and save an outline.")
        return
    outline = json.loads(plan.read_text())

    if args.sections:
        outline = dict(outline)
        outline["sections"] = outline["sections"][:args.sections]

    models = ([m.strip() for m in args.models.split(",")]
              if args.models else choose_models())
    if not models:
        return

    missing = set(models) - available_models()
    if missing:
        print(f"\n  Not installed: {', '.join(sorted(missing))}")
        return

    n_sec = len(outline["sections"])
    est = len(models) * n_sec * 4
    print(f"\n  {len(models)} model(s) x {n_sec} section(s). "
          f"Roughly {est} minutes.")
    print(f"  Notes model held constant at {NOTES_MODEL}.")
    if not _confirm("  Go?", default=True):
        return

    print("\n  Gathering shared evidence...")
    registry, per_section = prepare_evidence(outline)
    print(f"  {len(registry.items)} unique passages, identical for every model.")

    results = []
    for m in models:
        try:
            results.append(run_model(m, outline, per_section, registry))
        except Exception as e:
            print(f"\n  {m} failed: {e}")

    if not results:
        return

    print(f"\n\n{'=' * 70}\n  RESULTS\n{'=' * 70}\n")
    print(comparison_table(results))
    run_dir = save_run(results, outline)
    print(f"\n  Drafts saved to {run_dir}")
    print("  Read them. The numbers rank candidates, they do not judge prose.")


if __name__ == "__main__":
    main()
