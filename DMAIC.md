# DMAIC for El Roi's modules

Every module that generates or transforms text for the user gets its own
Define/Measure/Analyze/Improve/Control pass, recorded the same way, in the
same table (`memory/query_quality_events`), so `/review` in orchestrator.py
(or any future dashboard) is one queue, not one per module.

This file is the template. When a new module ships, add a section below
before considering it done — "it works on my test question" is not the bar;
"it has a Define, a Measure, and a way to show up in /review when it fails"
is.

## The five phases, as used here

- **Define** — the explicit rule(s) the output must satisfy, in the same
  words the user gave them. Stored in `define_json` per turn. Vague rules
  ("write it well") can't be measured; a rule belongs here only once it's
  phrased as a check you could fail.
- **Measure** — the objective signals computed from the actual output:
  regex/structural checks, `quality.py`'s metrics, retrieval counts.
  Never a vibe, always a number or a boolean.
- **Analyze** — what the measurements mean: which specific rule broke, and
  a `bug_types` list naming the failure mode(s).
- **Improve** — what was already tried automatically to fix it this turn
  (an LLM repair pass, a retry, a mechanical patch) and whether it worked.
  If a mechanical patch (regex-based) fires, that's itself worth watching —
  it means the smarter fix (asking the model again) didn't land, so patch
  usage is a leading indicator, not a solved problem.
- **Control** — `needs_review: true/false` plus `bug_types`, written to
  `control_json` on every turn. This is what `/review` reads. A module with
  a Define and a Measure but nothing feeding `bug_types` into `control` has
  not actually implemented Control — it just has documentation.

## Where this lives in code today

- `ask.py::_apa7_issues` — Define, as regex checks, for APA 7 formatting:
  no citation opening/closing a paragraph, 3+ sentences per paragraph,
  author-date in-text citations (not `[C1]` markers or raw URLs), a
  References section, dated reference entries, the required count of
  distinct 2024+ peer-reviewed in-text citations.
- `ask.py::_detect_response_defects` — the Measure/Analyze step run on
  every chat turn, feature-agnostic: folds in `_apa7_issues`, a filtered
  subset of `quality.format_flags` (leaked `<think>` tags, stray code
  fences, placeholder citations, meta-preamble — the ones that are wrong in
  chat, unlike `no_h2_heading` which only makes sense for `writer.py`'s
  sectioned drafts), and whether the degenerate-output retry failed.
- `ask.py::_quality_finish` — Control: turns `bug_types` (plus a
  grounded-but-empty result) into `control.needs_review`, and always calls
  `record_query_quality` so the row lands in `query_quality_events`
  regardless of whether anything was wrong.
- `quality.py` — the six failure modes for `writer.py`'s long-form drafts:
  fabricated citations, ungrounded prose, self-repetition, cross-section
  bleed, length drift, format disobedience. This is the Measure layer for
  the draft path.
- `writer.py::write_document` — Control for drafts: any fabricated marker,
  any `quality.py` format flag, or length ratio off by more than 30%,
  writes `needs_review: true` into the same `query_quality_events` table.
- `orchestrator.py::cmd_review` (`/review`) — reads
  `memory_client.list_needs_review()`, newest first, scoped to the active
  project. This is the actual "which threads have a bug and need fixing"
  view. `/flag` still exists separately for whatever a human notices that
  no automated check catches yet — a `/flag`'d pattern that recurs is a
  signal to add a new Define/Measure check here, not to keep flagging it
  by hand forever.

## Adding a new module's DMAIC

1. Write the Define as a list of falsifiable rules (regex or otherwise),
   not prose intentions.
2. Write a Measure function that returns those rule violations as a list
   of strings (see `_apa7_issues`) or booleans.
3. Fold the result into a `bug_types` list and pass it through as the
   `analysis` argument wherever this module's version of `_quality_finish`
   /`record_query_quality` runs, so it lands in `control.needs_review`.
4. Note in this file which mechanical "Improve" patches exist, since those
   are the parts most likely to themselves need fixing later (see the
   2026-08-30 APA formatting incident below).

## Known incidents (Analyze/Improve history)

- **2026-08-30 — APA formatting, garbled + duplicated citations.** The
  repair loop in `ask.py` accepted a partial fix (fewer issues, not zero)
  and then ran two mechanical patches unconditionally:
  `_ensure_apa_in_text_citations` could insert a seed citation into a
  sentence that already had one, nesting citations
  (`(Source Material, n.d (Soltani, 2024).)`); `_tidy_apa_output` closed
  every still-bad paragraph with one identical hardcoded sentence, which is
  why it showed up twice verbatim in one answer. Fixed by (a) skipping a
  paragraph entirely when every sentence already carries a citation instead
  of falling back into one, (b) rotating through several distinct closer
  sentences, (c) raising the repair loop from 2 to 3 attempts. This
  incident is also the reason `_detect_response_defects`/`/review` exist:
  the bug was invisible until a person happened to read that one answer
  closely — nothing was watching for it across every other conversation.
