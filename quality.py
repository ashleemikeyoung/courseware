"""
quality.py — computable quality metrics for locally generated long-form drafts.

The honest framing first: none of this measures whether the prose is any good.
No local metric does. What these measure is whether a model FAILED in one of the
specific, boring, mechanical ways that long-form RAG generation fails, and those
failures are common enough that catching them automatically saves you reading
six full drafts to notice one of them is quietly fabricating citations.

The six failure modes, and how each is detected:

  fabricated citations   model cites [C47] when only C1-C12 exist
  ungrounded prose       fluent text written from model priors, not your documents
  self-repetition        the same paragraph rewritten three times inside one section
  cross-section bleed    section 5 re-explains what section 2 already established
  length drift           400 words delivered against an 800 word target
  format disobedience    preamble, meta-commentary, missing heading

Every metric is direction-annotated so a comparison table can be read at a glance.
Scores are heuristics with visible weights, meant for ranking candidates against
each other, not as absolute grades. Read the dimensions, not the composite.
"""

import re
from collections import Counter

MARKER_RE = re.compile(r"\[(C\d+)\]")
WORD_RE = re.compile(r"[a-z0-9']+")

# Openers the drafting system prompt explicitly forbids. Catching these is the
# cheapest possible proxy for "did the model follow instructions at all".
PREAMBLE_RE = re.compile(
    r"\b(in this section,?\s+(we|i|this)|this section will|"
    r"we will (explore|examine|discuss|look at)|"
    r"let('s| us) (begin|start|explore)|"
    r"as an ai|i cannot|i'm sorry|here is the section|"
    r"certainly[!,.]|sure[!,.]\s+here)\b",
    re.IGNORECASE,
)


def tokens(text: str) -> list:
    return WORD_RE.findall(text.lower())


def ngrams(toks: list, n: int) -> Counter:
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def distinct_n(text: str, n: int = 3) -> float:
    """
    Unique n-grams over total n-grams. Standard degeneration metric.

    A model looping or padding drops here fast. Healthy expository prose sits
    around 0.85-0.95 for trigrams. Below ~0.75 means visible repetition.
    Higher is better.
    """
    grams = ngrams(tokens(text), n)
    total = sum(grams.values())
    return round(len(grams) / total, 4) if total else 0.0


def grounding(text: str, evidence_text: str, n: int = 5) -> float:
    """
    Fraction of the draft's 5-grams that also appear in the evidence it was given.

    This is the metric that matters most for a RAG system and the one nobody
    measures. A model can write beautifully fluent prose entirely from its own
    parametric knowledge while ignoring your documents completely, and it will
    look great until you notice none of it is actually about your material.

    Absolute values are low by nature, since good writing paraphrases rather than
    copies. Typical healthy range is 0.04-0.15. Treat it as strictly comparative:
    of two models given identical evidence, the higher one is leaning on your
    documents more and inventing less. Very high (>0.4) is its own problem, that
    is extraction rather than writing. Higher is better, within reason.
    """
    draft = ngrams(tokens(text), n)
    if not draft:
        return 0.0
    source = set(ngrams(tokens(evidence_text), n))
    hits = sum(c for g, c in draft.items() if g in source)
    return round(hits / sum(draft.values()), 4)


def cross_section_overlap(sections: list, n: int = 5) -> dict:
    """
    Pairwise Jaccard similarity on 5-grams between every pair of sections.

    This is the failure mode unique to sectioned generation and the reason the
    running-notes mechanism exists. When compression fails, or when a model
    ignores the do-not-repeat instruction, section six starts relitigating
    section two. Mean overlap above ~0.05, or any single pair above ~0.12,
    means the document reads as repetitive. Lower is better.
    """
    if len(sections) < 2:
        return {"mean": 0.0, "max": 0.0, "worst_pair": None}

    sets = [set(ngrams(tokens(s), n)) for s in sections]
    scores = []
    worst, worst_pair = 0.0, None
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if not union:
                continue
            jac = len(sets[i] & sets[j]) / len(union)
            scores.append(jac)
            if jac > worst:
                worst, worst_pair = jac, (i + 1, j + 1)

    return {
        "mean": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "max": round(worst, 4),
        "worst_pair": worst_pair,
    }


def citation_stats(text: str, valid_markers: set, offered: set = None) -> dict:
    """
    Citation fidelity. The single most important number in this whole file.

    fabricated  markers cited that were never in the evidence. Should be zero.
                Anything above zero means the model invents sources under
                pressure, which is disqualifying for research writing no matter
                how nice the prose reads.
    uptake      of the passages actually offered to it, how many did it use.
                Low uptake means the model ignored most of your retrieval, so
                you paid for context it threw away.
    density     citations per 100 words. Very low means unsupported assertion,
                very high means the model is stringing quotes instead of arguing.
    """
    used = MARKER_RE.findall(text)
    used_set = set(used)
    real = used_set & valid_markers
    fabricated = used_set - valid_markers
    n_words = len(tokens(text)) or 1

    stats = {
        "cited_distinct": len(real),
        "fabricated": sorted(fabricated),
        "fabricated_count": len(fabricated),
        "density_per_100w": round(100 * len(used) / n_words, 2),
    }
    if offered:
        stats["uptake"] = round(len(real & offered) / len(offered), 3)
    return stats


def length_adherence(text: str, target: int) -> dict:
    """Ratio of delivered to requested length. 1.0 is perfect, <1 undershoots."""
    actual = len(tokens(text))
    return {
        "words": actual,
        "target": target,
        "ratio": round(actual / target, 3) if target else 0.0,
    }


def format_flags(text: str) -> list:
    """Cheap instruction-following checks against the drafting system prompt."""
    flags = []
    stripped = text.strip()
    if not stripped.startswith("##"):
        flags.append("no_h2_heading")
    if PREAMBLE_RE.search(stripped[:600]):
        flags.append("preamble_or_meta")
    if "```" in stripped:
        flags.append("stray_code_fence")
    if "<think" in stripped or "</think>" in stripped:
        flags.append("leaked_thinking")
    if re.search(r"\[citation needed\]|\[source\]|\[ref\]", stripped, re.I):
        flags.append("placeholder_citation")
    return flags


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

# Transparent and editable. Fabrication dominates on purpose: a model that
# invents sources is not a candidate regardless of how it scores elsewhere.
WEIGHTS = {
    "no_fabrication": 40,
    "grounding": 20,
    "low_repetition": 15,
    "low_cross_overlap": 10,
    "length_adherence": 10,
    "clean_format": 5,
}


def composite(report: dict) -> float:
    """
    Heuristic 0-100 ranking score. Useful for sorting a comparison table.
    Not a grade. Always look at the dimensions underneath it.
    """
    c = report["citations"]
    s = 0.0

    s += WEIGHTS["no_fabrication"] if c["fabricated_count"] == 0 else max(
        0, WEIGHTS["no_fabrication"] - 10 * c["fabricated_count"])

    # Grounding is scored as a band, not a slope. Rising to 0.12 earns full
    # marks, but past ~0.30 the model has stopped paraphrasing and started
    # transcribing, which also shows up as a repetition problem. Without this
    # ceiling a model that loops over copied evidence scores well on grounding
    # for exactly the wrong reason.
    g = report["grounding"]
    s += WEIGHTS["grounding"] * (
        min(1.0, g / 0.12) if g <= 0.30 else max(0.0, 1 - (g - 0.30) / 0.30)
    )

    # distinct-3 of 0.90 or better is full marks, 0.70 is zero.
    s += WEIGHTS["low_repetition"] * max(0.0, min(
        1.0, (report["distinct_3"] - 0.70) / 0.20))

    # mean cross-section overlap of 0 is full marks, 0.10 is zero.
    s += WEIGHTS["low_cross_overlap"] * max(0.0, min(
        1.0, 1 - report["cross_section"]["mean"] / 0.10))

    # Symmetric penalty for over- and undershooting the target.
    s += WEIGHTS["length_adherence"] * max(
        0.0, 1 - abs(1 - report["length"]["ratio"]))

    s += WEIGHTS["clean_format"] * max(0.0, 1 - 0.34 * len(report["flags"]))

    return round(s, 1)


def score_document(sections: list, evidence_text: str,
                   valid_markers: set, target_words: int,
                   offered: set = None) -> dict:
    """Score a whole drafted document. sections is a list of section bodies."""
    full = "\n\n".join(sections)
    report = {
        "citations": citation_stats(full, valid_markers, offered),
        "grounding": grounding(full, evidence_text),
        "distinct_3": distinct_n(full, 3),
        "distinct_1": distinct_n(full, 1),
        "cross_section": cross_section_overlap(sections),
        "length": length_adherence(full, target_words),
        "flags": sorted({f for s in sections for f in format_flags(s)}),
    }
    report["score"] = composite(report)
    return report
