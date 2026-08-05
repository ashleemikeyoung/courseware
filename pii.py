"""
pii.py -- local PII detection and redaction via Microsoft Presidio.

Requires (install yourself, not run automatically by anything here, into
the "rag" conda environment -- activate it first if your prompt doesn't
already show "(rag)"):
    conda activate rag
    pip install presidio-analyzer presidio-anonymizer spacy
    python -m spacy download en_core_web_lg

Everything here runs fully local -- Presidio's analyzer combines pattern
matching (SSNs, emails, phone numbers, credit cards, etc.) with a spaCy NER
model for free-text entities (names, locations, organizations). No network
calls, no data leaves the machine.

IMPORTANT, read before trusting this for anything: automated PII detection
has real false-negative and false-positive rates. It is a strong aid, not a
guarantee. Presidio's own docs are explicit that recall depends heavily on
document formatting, entity type, and language. Treat a "0 findings" result
as "nothing detected," not as "nothing present" -- especially for anything
you'd actually be liable for if it leaked. When it matters, a human should
still look.

Nothing here stores actual PII text anywhere. Callers that want a
persistent record should use memory_client.record_pii_scan(), which only
takes entity *types* and a *count* -- never the matched text itself. This
module's functions operate on live text in memory and return live text;
what a caller does with the result (display it, discard it, etc.) is up
to them.
"""

from functools import lru_cache

# Deliberately imported lazily inside functions, not at module load time --
# importing spaCy's model is slow (a few seconds), and every script that
# imports this module (orchestrator.py, writer.py, a future ingest hook)
# would pay that cost on startup even if PII redaction is never actually
# used in that run. Paying it once, on first real use, is a better trade.
_analyzer = None
_anonymizer = None


def _get_engines():
    global _analyzer, _anonymizer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        _analyzer = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


# The entity types Presidio checks by default include some that are noisy
# for a legal/academic document corpus (e.g. URL, IP_ADDRESS flagging every
# citation link and perma.cc archive URL in a law review article). Scoped
# down to the categories that are actually personally identifying.
DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "LOCATION",
    "DATE_TIME",  # off by default in practice -- see analyze_text's entities= param
]


def analyze_text(text: str, entities: list = None, score_threshold: float = 0.5):
    """
    Returns a list of findings: [{"entity_type": str, "start": int, "end": int,
    "score": float}, ...]. Never includes the matched text itself in the
    return value -- callers that need the actual matched substring should
    slice it from `text` themselves using start/end, not rely on this
    function to hand it back, which makes it harder to accidentally end up
    persisting PII text somewhere by just logging what this function returned.
    """
    analyzer, _ = _get_engines()
    entities = entities or [e for e in DEFAULT_ENTITIES if e != "DATE_TIME"]
    results = analyzer.analyze(text=text, entities=entities, language="en",
                               score_threshold=score_threshold)
    return [
        {"entity_type": r.entity_type, "start": r.start, "end": r.end,
         "score": round(r.score, 3)}
        for r in results
    ]


def redact_text(text: str, entities: list = None, score_threshold: float = 0.5) -> str:
    """
    Returns text with detected PII replaced by placeholders like
    "<PERSON>", "<EMAIL_ADDRESS>". Computed fresh every call -- nothing
    cached or persisted. This is the only function that should be called
    at the point text is actually about to be shown or saved.
    """
    analyzer, anonymizer = _get_engines()
    entities = entities or [e for e in DEFAULT_ENTITIES if e != "DATE_TIME"]
    results = analyzer.analyze(text=text, entities=entities, language="en",
                               score_threshold=score_threshold)
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text


def summarize_entity_types(text: str, score_threshold: float = 0.5) -> dict:
    """
    For ingest-time scanning: returns {"entity_types": [...], "finding_count": N}
    -- the summary shape memory_client.record_pii_scan() expects. No PII text
    included, only which categories were found and how many total.
    """
    findings = analyze_text(text, score_threshold=score_threshold)
    types = sorted({f["entity_type"] for f in findings})
    return {"entity_types": types, "finding_count": len(findings)}
