#!/usr/bin/env python3
"""Write a throughput graph from a benchmark result file:

    scripts/performance.py bench/results/neon.md docs/performance-aarch64.svg "caption"

The input is the table printed by `utf32-bench` (GB/s of UTF-8 input). The
colours are those of the winnow-args figures."""
import re, sys

FONT = "ui-sans-serif,system-ui,Segoe UI,Helvetica,Arial,sans-serif"
TEAL, GREY = "#5fafaf", "#8a8a8a"
# (label, row name in the table, colour). The first row that exists is used.
SERIES = [
    ("simdutf (C++)", "simdutf C++ (strict)", "#d78787"),
    ("std + chars", "std from_utf8 + chars", "#87afaf"),
    ("scalar", "ours scalar", GREY),
    ("sse4.1", "ours sse4.1", "#87af87"),
    ("avx2", "ours avx2", "#5faf87"),
    ("avx512", "ours avx512", "#5faf5f"),
    ("neon", "ours neon", "#5faf5f"),
]
TITLES = {
    "ascii": "ASCII",
    "latin": "Latin (2-byte)",
    "cjk": "CJK (3-byte)",
    "box/mixed": "box drawing and ASCII",
    "emoji": "emoji (4-byte)",
    "dirty": "dirty: lossy, errors every ~12 B",
}


def parse(path):
    cols, rows = None, {}
    for line in open(path):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cols is None:
            cols = cells[1:]
            continue
        rows[cells[0]] = [None if c == "-" else float(c) for c in cells[1:]]
    return cols, rows


def value(rows, name, ci, dirty):
    # Clean columns use the strict slice function; the dirty one only has lossy.
    for suffix in (["lossy_into (C1)"] if dirty else ["slice fn"]) if name.startswith("ours") else [""]:
        key = f"{name} {suffix}".strip()
        for k, v in rows.items():
            if k == key or (not suffix and k.startswith(name)):
                if v[ci] is not None:
                    return v[ci]
    return None


def nice_top(m):
    for step in (0.5, 1, 2, 2.5, 5, 10, 20):
        if m / step <= 5:
            return step * (int(m / step) + (0 if m % step == 0 else 1)), step
    return m, m


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


def main(src, dst, caption):
    cols, rows = parse(src)
    out = []

    def text(x, y, s, anchor="start", fill=GREY, weight=None, size=12):
        w = f' font-weight="{weight}"' if weight else ""
        out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
                   f'fill="{fill}" text-anchor="{anchor}"{w}>{esc(s)}</text>')

    PW, PX, LABEL, BARW, ROW, BAR = 350, 8, 94, 198, 19, 13
    maxrows = max(sum(value(rows, n, ci, c == 'dirty') is not None for _, n, _ in SERIES) for ci, c in enumerate(cols))
    PITCH = 26 + ROW * maxrows + 32
    for pi, col in enumerate(cols):
        dirty = col == "dirty"
        data = []
        for label, name, colour in SERIES:
            v = value(rows, name, pi, dirty)
            if v is not None:
                data.append((label, v, colour))
        if not data:
            continue
        gx = PX + (pi % 2) * (PW + 10)
        top_y = 6 + (pi // 2) * PITCH
        text(gx, top_y + 14, TITLES.get(col, col), fill=TEAL, weight=600, size=13)
        top, step = nice_top(max(v for _, v, _ in data))
        x0, y0 = gx + LABEL, top_y + 26
        y1 = y0 + ROW * len(data)
        t = 0.0
        while t <= top + 1e-9:
            x = x0 + BARW * t / top
            out.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{GREY}" stroke-opacity="0.35" stroke-width="1"/>')
            text(x, y1 + 14, f"{t:g}", "middle")
            t += step
        for n, (label, v, colour) in enumerate(data):
            ry = y0 + n * ROW + (ROW - BAR) // 2
            w = BARW * v / top
            text(x0 - 6, ry + BAR - 2, label, "end")
            out.append(f'<rect x="{x0}" y="{ry}" width="{w:.1f}" height="{BAR}" rx="2" fill="{colour}"/>')
            text(x0 + w + 5, ry + BAR - 2, f"{v:.2f}")
    nrows = (len(cols) + 1) // 2
    H = 6 + nrows * PITCH + 2
    text(360, H, "GB/s of UTF-8 input, higher is better. " + caption, "middle", size=11)
    H += 10
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="{round(H * 1024 / 720)}" '
           f'viewBox="0 0 720 {H}" role="img" aria-label="UTF-8 to UTF-32 throughput in GB/s, by input type and implementation.">\n'
           f'<rect width="720" height="{H}" fill="none"/>\n' + "\n".join(out) + "\n</svg>\n")
    open(dst, "w").write(svg)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
