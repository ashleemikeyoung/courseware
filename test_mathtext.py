"""
test_mathtext.py — the currency-versus-maths rule, and the terminal fallback.

    python test_mathtext.py

No network, no models, no state. Every case here is a real string from the
material: 14.03's 50/50 draw between $2 and $0, the Allais payoffs in millions,
and the formulas a local model produced when asked to synthesize expected
utility from OCW notes.

The bug this exists to prevent was found in a browser, not in review. With "$"
enabled as an inline delimiter, "the price was $5 and the cost was $3" pairs
its two dollar signs and renders as "5andthecostwas3". Anything that loosens
looks_like_math() needs to keep every case in the first section passing.
"""

import sys

import mathtext


_failures = 0


def chk(label, got, want):
    global _failures
    ok = got == want
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"         got:  {got!r}")
        print(f"         want: {want!r}")
        _failures += 1


def main():
    print("Currency is never maths")
    chk("two prices in one sentence",
        mathtext.normalize("the price was $5 and the cost was $3"),
        "the price was $5 and the cost was $3")
    chk("Allais payoffs",
        mathtext.normalize("$1 million with certainty, or $5 million at 0.10"),
        "$1 million with certainty, or $5 million at 0.10")
    chk("14.03's worked draw",
        mathtext.normalize("a 50/50 gamble between $2 and $0"),
        "a 50/50 gamble between $2 and $0")
    chk("one unmatched dollar sign",
        mathtext.normalize("costs $5 today"), "costs $5 today")
    chk("a certainty equivalent quoted in dollars",
        mathtext.normalize("CE = $0.50, so the premium is $0.50"),
        "CE = $0.50, so the premium is $0.50")

    print("Bare symbols are maths, because economics prose is full of them")
    chk("expectation operator", mathtext.normalize("Expected Value ($E(X)$):"),
        "Expected Value (\\(E(X)\\)):")
    chk("variance operator", mathtext.normalize("Variance ($V(X)$)"),
        "Variance (\\(V(X)\\))")
    chk("a single letter", mathtext.normalize("the utility $u$ is concave"),
        "the utility \\(u\\) is concave")
    chk("a CDF", mathtext.normalize("lottery $F$ dominates"),
        "lottery \\(F\\) dominates")
    chk("uppercase commodity label",
        mathtext.normalize("apples ($A$) and bananas (B)."),
        "apples (\\(A\\)) and bananas (B).")
    chk("single-letter equation variable",
        mathtext.normalize("Here, $a=2$ and b=3."),
        "Here, \\(a=2\\) and \\(b=3\\).")
    chk("single-letter action variable",
        mathtext.normalize("the action $a$ maximizes utility"),
        "the action \\(a\\) maximizes utility")
    chk("bare function assignment",
        mathtext.normalize("The new utility function is v(c)=2+3u(c)."),
        "The new utility function is \\(v(c)=2+3u(c)\\).")
    chk("bare numbered variable assignment",
        mathtext.normalize("buy x1=3 apples"),
        "buy \\(x1=3\\) apples")
    chk("malformed bundle labels are repaired",
        mathtext.normalize("Feasible Bundles (Bfeasible), Preferences (Bpreferences), Budget Constraint (Bbudget)"),
        r"Feasible Bundles \(\mathcal{B}_{\text{feasible}}\), Preferences \(\mathcal{B}_{\text{preferences}}\), Budget Constraint \(\mathcal{B}_{\text{budget}}\)")
    chk("a clause between two prices is still not maths",
        mathtext.normalize("worth $5 and worth $3"), "worth $5 and worth $3")
    chk("a long symbol run is not maths",
        mathtext.normalize("$" + "abc " * 12 + "$"), "$" + "abc " * 12 + "$")

    print("Real maths is converted")
    chk("a control sequence makes it maths",
        mathtext.normalize(r"probabilities ($\sum p_n = 1$)."),
        r"probabilities (\(\sum p_n = 1\)).")
    chk("subscripts make it maths",
        mathtext.normalize(r"$E[U(X)] = \sum p_i u(x_i)$"),
        r"\(E[U(X)] = \sum p_i u(x_i)\)")
    chk("escaped currency can appear inside a formula",
        mathtext.normalize(r"the cost is $1.00 \times 3 = \$3.00$."),
        r"the cost is \(1.00 \times 3 = \$3.00\).")
    chk("budget arithmetic with escaped currency renders as one formula",
        mathtext.normalize(r"remaining is $5.00 - 3.00 = \$2.00$."),
        r"remaining is \(5.00 - 3.00 = \$2.00\).")
    chk("model-wrapped prose money becomes plain money",
        mathtext.normalize(r"has $\$10$, apples cost $\$1$ each"),
        "has $10, apples cost $1 each")
    chk("model-wrapped million payoff becomes prose money",
        mathtext.normalize(r"Win $1$ million for sure."),
        "Win $1 million for sure.")
    chk("model-wrapped percent becomes prose percent",
        mathtext.normalize(r"probability $10\%$, nothing with $1\%$."),
        "probability 10%, nothing with 1%.")
    chk("model-wrapped plain payoff becomes prose money",
        mathtext.normalize(r"If the ball is red, you win $100$."),
        "If the ball is red, you win $100.")
    chk("bare budget inequality is wrapped",
        mathtext.normalize("The budget constraint is 1A+2B≤10."),
        r"The budget constraint is \(1A+2B≤10\).")
    chk("bare TeX equation is wrapped",
        mathtext.normalize(r"E[U(W)] = \sum_i p_i u(W + x_i), where p_i is probability."),
        r"\(E[U(W)] = \sum_i p_i u(W + x_i)\), where p_i is probability.")
    chk("display dollars become bracket display",
        mathtext.normalize(r"$$P_W(\pi) = \omega(\pi) u(x)$$"),
        r"\[P_W(\pi) = \omega(\pi) u(x)\]")
    chk("delimiters already correct are untouched",
        mathtext.normalize(r"\(r_A(x)\) and \[y = 2\]"),
        r"\(r_A(x)\) and \[y = 2\]")
    chk("prose with no dollars is untouched",
        mathtext.normalize("plain prose about risk aversion"),
        "plain prose about risk aversion")
    chk("a long run between dollars is not swallowed",
        mathtext.normalize("$5 " + "x" * 250 + r" \alpha $"),
        "$5 " + "x" * 250 + r" \alpha $")

    print("Line-broken model maths is repaired")
    chk("duplicate expectation stack",
        mathtext.normalize("Expected Value (\nE\n[\nX\n]\nE[X]):"),
        "Expected Value (\n\\(E[X]\\)):")
    chk("duplicate utility stack",
        mathtext.normalize("utility function \nU\n(\nW\n)\nU(W), but"),
        "utility function \n\\(U(W)\\), but")
    chk("ordinary prose lines are untouched",
        mathtext.normalize("Heads: win $125.\nTails: lose $100."),
        "Heads: win $125.\nTails: lose $100.")

    print("Unicode, for a surface with no renderer")
    chk("sum with subscripts",
        mathtext.render_unicode(r"\(E[U(X)] = \sum p_i u(x_i)\)"),
        "E[U(X)] = ∑ pᵢ u(xᵢ)")
    chk("Arrow-Pratt as a readable ratio",
        mathtext.render_unicode(r"\(r_A(x) = -\frac{u''(x)}{u'(x)}\)"),
        "r_A(x) = -(u''(x))/(u'(x))")
    chk("blackboard expectation",
        mathtext.render_unicode(r"\(\mathbb{E}_F[x]\)"), "𝔼_F[x]")
    chk("integral", mathtext.render_unicode(r"\(\int u(x) dF(x)\)"),
        "∫ u(x) dF(x)")
    chk("preference relation",
        mathtext.render_unicode(r"\(p \succeq q\)"), "p ⪰ q")
    chk("inverse superscript",
        mathtext.render_unicode(r"\(u^{-1}\)"), "u⁻¹")
    chk("a script it cannot render keeps its source form",
        mathtext.render_unicode(r"\(x^{\alpha\beta}\)"), "x^αβ")
    chk("display maths gets its own indented line",
        "\n    x = y" in mathtext.render_unicode(r"before \[x = y\] after"), True)
    chk("currency survives the unicode pass too",
        mathtext.render_unicode("the price was $5 and the cost was $3"),
        "the price was $5 and the cost was $3")
    chk("$-maths is picked up by the unicode pass as well",
        mathtext.render_unicode(r"($\sum p_n = 1$)"), "(∑ pₙ = 1)")

    print()
    if _failures:
        print(f"FAILURES: {_failures}")
        return 1
    print("All mathtext checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
