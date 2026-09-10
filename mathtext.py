"""
mathtext.py — make model-written LaTeX safe to render, and readable where it
cannot be rendered.

Two jobs, one shared rule about what counts as maths.

normalize() rewrites the delimiters a local model actually produces into the
two the browser renderer is configured for. render_unicode() turns the same
maths into Unicode for the terminal, which has no renderer at all.

Why the $ delimiter cannot simply be handed to the renderer:

    KaTeX's auto-render will happily accept "$" as an inline delimiter, and
    for most documents that is fine. It is not fine here. This tool teaches
    decision theory, where the source material is full of money: the Allais
    paradox is stated in dollars, 14.03's worked example is a 50/50 draw
    between $2 and $0, and a sentence like "the price was $5 and the cost was
    $3" pairs its two dollar signs and renders as "5andthecostwas3".

    Verified in a browser before this module existed, which is how the bug was
    found rather than shipped.

    So the disambiguation happens here, in Python, where a real rule can be
    written: a $-pair becomes maths only when what sits between the signs
    actually looks like TeX -- a control sequence, a subscript, a superscript.
    "$5 and the cost was $3" contains none of those and is left exactly as the
    author typed it. "$\\sum p_n = 1$" contains \\sum and becomes \\( \\sum p_n = 1
    \\). The browser is then configured for \\( \\) and \\[ \\] only, so a stray
    dollar sign can never again swallow a sentence.

    The cost of this rule is a false negative: "$x$" on its own, with no
    control sequence and no script, stays literal. That is the right way to be
    wrong. A missed italic x is a blemish; a mangled price is a lie about the
    material.
"""

import re

# A $-pair is maths only if what it encloses carries one of these. Anything
# else is money, or prose, or a typo.
TEX_MARKUP_RE = re.compile(r"\\[A-Za-z]+|[_^]|\\left|\\right")
EQUATION_MARKUP_RE = re.compile(r"[=<>+\-−*/×⋅·]|\([^)\n]+\)|\[[^\]\n]+\]")

# Deliberately not DOTALL: real inline maths does not span a paragraph, and
# allowing it to means one unmatched dollar sign eats the rest of the answer.
INLINE_DOLLAR_RE = re.compile(r"(?<![\\$])\$(?!\$)((?:\\\$|[^\n$]){1,200}?)(?<!\\)\$(?!\$)")
DISPLAY_DOLLAR_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
WRAPPED_MONEY_RE = re.compile(
    r"(?<!\\)\$\\+\$(\d+(?:\.\d+)?(?:\s+(?:million|billion))?)\$(?!\$)",
    re.IGNORECASE,
)
PROSE_MONEY_RE = re.compile(
    r"(?<!\\)\$(\d+(?:\.\d+)?)\$(\s+(?:million|billion|trillion)\b)",
    re.IGNORECASE,
)
PROSE_STANDALONE_MONEY_RE = re.compile(
    r"(?<!\\)\$(\d+(?:\.\d+)?)\$(?=\s*(?:[.,;:]|\)|$))"
)
PROSE_PERCENT_RE = re.compile(r"(?<!\\)\$(\d+(?:\.\d+)?)\\%\$(?!\$)")
IDENT_RE = r"[A-Za-z][A-Za-z0-9_]*"
TEX_EQUATION_RE = re.compile(
    r"(?<![\\\w$])"
    r"((?:\\mathbb\{[A-Za-z]\}|[A-Za-z])[A-Za-z0-9_]*(?:\[[^\]\n]+\]|\([^\)\n]+\))?"
    r"\s*=\s*[^,.;:\n]*\\[A-Za-z]+[^,.;:\n]*)")
BARE_EQUATION_RE = re.compile(
    r"(?<![\\\w$])"
    r"(" + IDENT_RE + r"(?:\(" + IDENT_RE + r"\))?\s*=\s*"
    r"[-+]?\d+(?:\.\d+)?(?:\s*[+*/-]\s*"
    r"(?:\d+(?:\.\d+)?" + IDENT_RE + r"(?:\(" + IDENT_RE + r"\))?|"
    r"\d+(?:\.\d+)?|" + IDENT_RE + r"(?:\(" + IDENT_RE + r"\))?))*"
    r")"
    r"(?![\w$])")
BARE_INEQUALITY_RE = re.compile(
    r"(?<![\\\w$])"
    r"((?:\d+(?:\.\d+)?" + IDENT_RE + r"|" + IDENT_RE + r"|\d+(?:\.\d+)?)"
    r"(?:\s*[+*/-]\s*(?:\d+(?:\.\d+)?" + IDENT_RE + r"|" + IDENT_RE + r"|\d+(?:\.\d+)?))+"
    r"\s*(?:<=|>=|≤|≥|<|>)\s*\d+(?:\.\d+)?)"
    r"(?![\w$])")
BAD_SET_LABEL_RE = re.compile(r"\(?\bB(feasible|preferences|budget)\b\)?")
MATH_SPAN_RE = re.compile(r"(\\\(.+?\\\)|\\\[.+?\\\])", re.DOTALL)

MAX_INLINE = 200
MATH_ATOM_RE = re.compile(
    r"^\s*(?:[A-Za-z]|[()\[\]{}=+\-−<>×⋅·*/|]|\\?[A-Za-z]+|"
    r"\d+(?:\.\d+)?|…|\.{2,})\s*$")
MATHY_PREFIX_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*(?:\[[^\]\n]+\]|\([^\)\n]+\))?"
    r"(?:\s*[=+\-−<>×⋅·*/]\s*[^,.;:\n]+)?)")


# Words that only appear between two dollar signs when those dollar signs are
# money and the text between them is a sentence. Kept small and boring on
# purpose -- this list decides what gets typeset, so anything ambiguous stays
# out of it.
PROSE_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "cost", "costs",
    "each", "for", "from", "gets", "he", "in", "is", "it", "million",
    "billion", "of", "or", "pay", "pays", "per", "price", "she", "so", "than",
    "that", "the", "then", "they", "to", "was", "were", "when", "with", "you",
}

# Beyond this, a run between two dollar signs is a sentence, not a symbol.
MAX_BARE = 40

WORD_RE = re.compile(r"[A-Za-z]+")


def _compact_math(lines: list) -> str:
    out = "".join((line or "").strip() for line in lines)
    out = out.replace("−", "-")
    return out


def repair_line_broken_math(text: str) -> str:
    """
    Collapse copied/model-written math stacks into one LaTeX-renderable span.

    A local model sometimes emits both the line-broken visual form and the
    plain source form, e.g. "E" / "[" / "X" / "]" / "E[X]". KaTeX can render
    the source form, but the visual stack is just stray text. This pass removes
    the stack when the next line starts with the same compact expression.
    """
    if not text or "\n" not in text:
        return text or ""

    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        if not MATH_ATOM_RE.match(lines[i] or ""):
            out.append(lines[i])
            i += 1
            continue

        j = i
        atoms = []
        while j < len(lines) and MATH_ATOM_RE.match(lines[j] or ""):
            atoms.append(lines[j])
            j += 1

        compact = _compact_math(atoms)
        if (len(atoms) >= 2 and len(compact) >= 2 and j < len(lines)
                and MATHY_PREFIX_RE.match(lines[j] or "")):
            match = MATHY_PREFIX_RE.match(lines[j])
            formula = (match.group(1) or "").strip().replace("−", "-")
            squashed = re.sub(r"\s+", "", formula)
            if squashed.startswith(compact) or compact.startswith(squashed):
                rest = lines[j][match.end():]
                out.append(f"\\({formula}\\){rest}")
                i = j + 1
                continue

        out.extend(lines[i:j])
        i = j

    return "\n".join(out)


def wrap_bare_equations(text: str) -> str:
    """
    Wrap short assignment-like formulas the model leaves in prose.

    This is deliberately narrower than a real parser. It catches common
    teaching fragments such as b=3 and v(c)=2+3u(c), skips existing LaTeX
    spans, and does not touch dollar amounts because there is no dollar sign
    in the pattern at all.
    """
    if not text:
        return text or ""
    parts = MATH_SPAN_RE.split(text)
    for i, part in enumerate(parts):
        if not part or MATH_SPAN_RE.fullmatch(part):
            continue
        part = BAD_SET_LABEL_RE.sub(
            lambda m: r"\(\mathcal{B}_{\text{" + m.group(1) + r"}}\)",
            part)
        part = TEX_EQUATION_RE.sub(lambda m: f"\\({m.group(1).strip()}\\)", part)
        part = BARE_INEQUALITY_RE.sub(lambda m: f"\\({m.group(1)}\\)", part)
        parts[i] = BARE_EQUATION_RE.sub(lambda m: f"\\({m.group(1)}\\)", part)
    return "".join(parts)


def looks_like_math(body: str) -> bool:
    """
    Does the text between two dollar signs actually contain maths?

    Two ways to qualify.

    The certain one is TeX markup: a control sequence, a subscript, a
    superscript. "$\\sum p_n = 1$" is maths and nothing else.

    The second exists because economics prose is full of bare symbols --
    "$E(X)$", "$V(X)$", "$u$", "$F$" -- which carry no markup at all. An
    earlier version of this rule refused all of them, and the very first real
    lesson printed "($E(X)$)" at the reader with the dollar signs showing.
    That is the same class of failure as the currency bug, just in the other
    direction.

    So a short run also qualifies when it cannot be money: money written in
    English starts with a digit right after the sign ("$5", "$1 million",
    "$0.50"), so a leading digit disqualifies outright. What remains has to
    contain a letter, stay under MAX_BARE characters, and contain no ordinary
    English word -- because the thing between the dollar signs in "the price
    was $5 and the cost was $3" is a clause, and clauses have words like
    "and" in them.

    The residual failure is a sentence of pure symbols between two prices,
    which does not occur in this material.
    """
    body = (body or "").strip()
    if not body or len(body) > MAX_INLINE:
        return False
    if TEX_MARKUP_RE.search(body):
        return True

    if len(body) > MAX_BARE:
        return False
    words = WORD_RE.findall(body)
    if re.fullmatch(r"[A-Za-z]", body):
        return True
    if body[0].isdigit() or body[0] in ".,":
        return bool(EQUATION_MARKUP_RE.search(body)
                    and not any(w.lower() in PROSE_WORDS for w in words))

    if not words:
        return False
    if EQUATION_MARKUP_RE.search(body) and all(len(w) <= 2 for w in words):
        return True
    return not any(w.lower() in PROSE_WORDS for w in words)


def normalize(text: str) -> str:
    """
    Rewrite $-delimited maths into \\( \\) and \\[ \\], leaving currency alone.

    Display maths first: $$ is unambiguous, has no currency reading, and doing
    it first stops the inline pass from tearing a $$ pair in half.
    """
    text = repair_line_broken_math(text or "")
    if "$" not in text:
        return wrap_bare_equations(text)

    text = WRAPPED_MONEY_RE.sub(lambda m: f"${m.group(1)}", text)
    text = PROSE_MONEY_RE.sub(lambda m: f"${m.group(1)}{m.group(2)}", text)
    text = PROSE_STANDALONE_MONEY_RE.sub(lambda m: f"${m.group(1)}", text)
    text = PROSE_PERCENT_RE.sub(lambda m: f"{m.group(1)}%", text)
    text = DISPLAY_DOLLAR_RE.sub(lambda m: f"\\[{m.group(1)}\\]", text)
    text = INLINE_DOLLAR_RE.sub(
        lambda m: f"\\({m.group(1)}\\)" if looks_like_math(m.group(1))
        else m.group(0),
        text)
    return wrap_bare_equations(text)


# ---------------------------------------------------------------------------
# Unicode, for the terminal
#
# Not a TeX engine and not trying to be. It covers what decision theory
# actually writes -- sums, integrals, expectations, greek, primes, sub- and
# superscripts, a few operators -- and leaves anything else as its own source
# text. A partially converted formula is still readable; a wrong one is not,
# so nothing here guesses.
# ---------------------------------------------------------------------------

SYMBOLS = {
    r"\sum": "∑", r"\prod": "∏", r"\int": "∫", r"\infty": "∞",
    r"\partial": "∂", r"\nabla": "∇", r"\forall": "∀", r"\exists": "∃",
    r"\in": "∈", r"\notin": "∉", r"\subset": "⊂", r"\subseteq": "⊆",
    r"\cup": "∪", r"\cap": "∩", r"\emptyset": "∅",
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥",
    r"\neq": "≠", r"\ne": "≠", r"\approx": "≈", r"\equiv": "≡",
    r"\sim": "∼", r"\propto": "∝", r"\pm": "±", r"\mp": "∓",
    r"\times": "×", r"\cdot": "·", r"\div": "÷",
    r"\to": "→", r"\rightarrow": "→", r"\leftarrow": "←",
    r"\Rightarrow": "⇒", r"\Leftarrow": "⇐", r"\iff": "⟺",
    r"\Leftrightarrow": "⇔", r"\mapsto": "↦",
    r"\succeq": "⪰", r"\succ": "≻", r"\preceq": "⪯", r"\prec": "≺",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\iota": "ι", r"\kappa": "κ", r"\lambda": "λ",
    r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ", r"\pi": "π", r"\rho": "ρ",
    r"\sigma": "σ", r"\tau": "τ", r"\upsilon": "υ", r"\phi": "φ",
    r"\varphi": "φ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Xi": "Ξ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Phi": "Φ",
    r"\Psi": "Ψ", r"\Omega": "Ω",
    r"\ldots": "…", r"\cdots": "⋯", r"\dots": "…",
    r"\prime": "′", r"\circ": "∘", r"\quad": "  ", r"\qquad": "    ",
    r"\,": " ", r"\;": " ", r"\!": "", r"\ ": " ",
}

SUPERSCRIPT = str.maketrans("0123456789+-=()nia", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵃ")
SUBSCRIPT = str.maketrans("0123456789+-=()aeioruvxjhklmnpst",
                          "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢₒᵣᵤᵥₓⱼₕₖₗₘₙₚₛₜ")

BLACKBOARD = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "E": "𝔼", "P": "ℙ"}


def _script(body: str, table: dict, ok: str) -> str:
    """Convert a script run, or give up on it whole. Half-converted is worse."""
    if body and all(ch in ok for ch in body):
        return body.translate(table)
    return None


def _expression(tex: str) -> str:
    out = tex

    out = re.sub(r"\\mathbb\{([A-Z])\}",
                 lambda m: BLACKBOARD.get(m.group(1), m.group(1)), out)
    out = re.sub(r"\\(?:mathrm|mathit|text|mathbf|operatorname)\{([^{}]*)\}",
                 r"\1", out)
    out = re.sub(r"\\(?:frac|dfrac|tfrac)\{([^{}]+)\}\{([^{}]+)\}",
                 r"(\1)/(\2)", out)
    out = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", out)
    out = re.sub(r"\\left|\\right", "", out)

    for name in sorted(SYMBOLS, key=len, reverse=True):
        out = out.replace(name, SYMBOLS[name])

    def sup(m):
        body = m.group(1) or m.group(2)
        got = _script(body, SUPERSCRIPT, "0123456789+-=()nia")
        return got if got is not None else f"^{body}"

    def sub(m):
        body = m.group(1) or m.group(2)
        got = _script(body, SUBSCRIPT, "0123456789+-=()aeioruvxjhklmnpst")
        return got if got is not None else f"_{body}"

    out = re.sub(r"\^\{([^{}]*)\}|\^(\w)", sup, out)
    out = re.sub(r"_\{([^{}]*)\}|_(\w)", sub, out)

    out = out.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", out).strip()


INLINE_TEX_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
DISPLAY_TEX_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)


def render_unicode(text: str) -> str:
    """
    LaTeX to Unicode, for a surface with no renderer.

    normalize() runs first so $-maths is picked up too. Display equations are
    put on their own indented line, because in a terminal that indent is the
    only thing distinguishing a formula from the sentence around it.
    """
    text = normalize(text or "")
    text = DISPLAY_TEX_RE.sub(lambda m: "\n\n    " + _expression(m.group(1)) + "\n",
                              text)
    return INLINE_TEX_RE.sub(lambda m: _expression(m.group(1)), text)
