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

# Deliberately not DOTALL: real inline maths does not span a paragraph, and
# allowing it to means one unmatched dollar sign eats the rest of the answer.
INLINE_DOLLAR_RE = re.compile(r"(?<!\$)\$(?!\$)([^\n$]{1,200}?)\$(?!\$)")
DISPLAY_DOLLAR_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)

MAX_INLINE = 200


def looks_like_math(body: str) -> bool:
    """
    Does the text between two dollar signs actually contain maths?

    Length is checked as well as markup: a very long run between two dollar
    signs is far more likely to be two prices in one sentence than a single
    inline expression, even if a stray backslash happens to fall between them.
    """
    body = body or ""
    if not body.strip() or len(body) > MAX_INLINE:
        return False
    return bool(TEX_MARKUP_RE.search(body))


def normalize(text: str) -> str:
    """
    Rewrite $-delimited maths into \\( \\) and \\[ \\], leaving currency alone.

    Display maths first: $$ is unambiguous, has no currency reading, and doing
    it first stops the inline pass from tearing a $$ pair in half.
    """
    if not text or "$" not in text:
        return text or ""

    text = DISPLAY_DOLLAR_RE.sub(lambda m: f"\\[{m.group(1)}\\]", text)
    return INLINE_DOLLAR_RE.sub(
        lambda m: f"\\({m.group(1)}\\)" if looks_like_math(m.group(1))
        else m.group(0),
        text)


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
