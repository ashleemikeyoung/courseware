"""
logo.py — generate theme-matched variants of the wordmark.

The source logo is two colours: a navy Q and a lighter blue for the rest. Navy
scores 1.51 contrast against the dark theme's background, which is invisible in
practice, so shipping one file and hoping is not an option.

Rather than filter in CSS (unreliable on two-tone art) or hand-trace an SVG
(lossy on those thin serifs and that long Q tail), this recolours the pixels
directly. Each pixel's position on the navy-to-blue gradient is measured, then
remapped onto the theme's own pair. Antialiased edges land at the same fraction
between the new colours, so the letterforms stay exactly as drawn.

Runs once at startup, caches to static/, and regenerates only if the source is
newer than the output.
"""

from pathlib import Path

SRC_NAVY = (0x12, 0x31, 0x56)
SRC_BLUE = (0x65, 0x99, 0xce)

# Per theme: (dark tone for the Q, lighter tone for the wordmark).
# The dark tone tracks each theme's --ink and the light tone its --blue, so the
# mark reads as part of the interface rather than pasted on top of it.
THEME_COLORS = {
    "light": ((0x12, 0x31, 0x56), (0x65, 0x99, 0xce)),   # unchanged, it already sits well
    "dark":  ((0xc7, 0xd8, 0xef), (0x7e, 0x9f, 0xd4)),   # inverted weighting: the Q must carry
    "pink":  ((0x3a, 0x2b, 0x52), (0x8b, 0x7a, 0xb8)),   # violet-leaning, to sit with the rose
    "grass": ((0xc8, 0xff, 0xc8), (0x68, 0xd3, 0x72)),   # Mac Terminal-ish green phosphor
}


def _mix(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _project(rgb):
    """
    Where this pixel sits between the source navy and blue, 0 to 1.

    Projecting onto the line between the two source colours (rather than
    thresholding) is what preserves antialiasing: an edge pixel that is 40%
    of the way between navy and blue comes out 40% of the way between the
    theme's two colours.
    """
    ax, ay, az = SRC_NAVY
    bx, by, bz = SRC_BLUE
    dx, dy, dz = bx - ax, by - ay, bz - az
    denom = dx * dx + dy * dy + dz * dz
    px, py, pz = rgb[0] - ax, rgb[1] - ay, rgb[2] - az
    t = (px * dx + py * dy + pz * dz) / denom
    return max(0.0, min(1.0, t))


def build(source: Path, out_dir: Path, force: bool = False) -> dict:
    """Write one PNG per theme. Returns {theme: path}."""
    try:
        from PIL import Image
    except ImportError:
        print("  Pillow not installed, skipping logo variants "
              "(pip install pillow). The light logo will be used everywhere.")
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    made = {}

    src_mtime = source.stat().st_mtime
    im = None

    for theme, (dark, light) in THEME_COLORS.items():
        dest = out_dir / f"logo-{theme}.png"
        made[theme] = dest
        if dest.exists() and not force and dest.stat().st_mtime >= src_mtime:
            continue

        if im is None:
            im = Image.open(source).convert("RGBA")

        w, h = im.size
        out = Image.new("RGBA", (w, h))
        src_px, out_px = im.load(), out.load()

        # Cache by source colour. The art has two colours plus their blend, so
        # a full-resolution recolour resolves to a few hundred lookups.
        cache = {}
        for y in range(h):
            for x in range(w):
                r, g, b, a = src_px[x, y]
                if a == 0:
                    continue
                key = (r, g, b)
                if key not in cache:
                    cache[key] = _mix(dark, light, _project(key))
                nr, ng, nb = cache[key]
                out_px[x, y] = (nr, ng, nb, a)

        out.save(dest, optimize=True)
        print(f"  logo: wrote {dest.name}")

    return made


if __name__ == "__main__":
    here = Path(__file__).parent
    build(here / "static" / "logo.png", here / "static", force=True)
