#!/usr/bin/env python3
"""Write a cross-architecture speedup summary from the benchmark result files:

    scripts/performance_summary.py

For each input class the graph shows the best SIMD level's speedup over scalar
on the same machine. Absolute GB/s from different machines are not comparable,
so the graph only compares ratios. It also prints the table as markdown.

The colours are those of the winnow-args figures."""
from performance import GREY, TEAL, esc, nice_top, parse, value

FONT = "ui-sans-serif,system-ui,Segoe UI,Helvetica,Arial,sans-serif"
COLS = ["ascii", "latin", "cjk", "box/mixed", "emoji", "dirty"]
TITLES = {
    "ascii": "ASCII",
    "latin": "Latin (2-byte)",
    "cjk": "CJK (3-byte)",
    "box/mixed": "box drawing and ASCII",
    "emoji": "emoji (4-byte)",
    "dirty": "dirty: lossy, errors every ~12 B",
}
ARCHES = [
    ("x86 / AVX-512", "bench/results/default.md", "ours avx512", "#5faf5f"),
    ("aarch64 / NEON", "bench/results/neon.md", "ours neon", "#8a7faf"),
    ("POWER9 / VSX", "bench/results/ppc64.md", "ours vsx", "#3f8f6f"),
]


def ratios():
    data = {}
    for label, path, simd, colour in ARCHES:
        _, rows = parse(path)
        cols, _ = parse(path)
        data[label] = (colour, {})
        for col in COLS:
            if col not in cols:
                continue
            ci = cols.index(col)
            dirty = col == "dirty"
            simd_v = value(rows, simd, ci, dirty)
            scalar_v = value(rows, "ours scalar", ci, dirty)
            if simd_v is not None and scalar_v:
                data[label][1][col] = simd_v / scalar_v
    return data


def main():
    data = ratios()
    print("| Speedup over scalar | " + " | ".join(TITLES.get(c, c) for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for label, (_, values) in data.items():
        cells = [f"{values[c]:.2f}x" if c in values else "–" for c in COLS]
        print(f"| {label} | " + " | ".join(cells) + " |")

    out = []

    def text(x, y, s, anchor="start", fill=GREY, weight=None, size=12):
        w = f' font-weight="{weight}"' if weight else ""
        out.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}"{w}>{esc(s)}</text>'
        )

    PW, PX, LABEL, BARW, ROW, BAR = 350, 8, 118, 170, 20, 13
    PITCH = 26 + ROW * len(ARCHES) + 32
    for pi, col in enumerate(COLS):
        gx = PX + (pi % 2) * (PW + 10)
        top_y = 6 + (pi // 2) * PITCH
        text(gx, top_y + 14, TITLES.get(col, col), fill=TEAL, weight=600, size=13)
        bars = [(label, colour, values[col]) for label, (colour, values) in data.items() if col in values]
        top, step = nice_top(max(v for _, _, v in bars))
        x0, y0 = gx + LABEL, top_y + 26
        y1 = y0 + ROW * len(bars)
        t = 0.0
        while t <= top + 1e-9:
            x = x0 + BARW * t / top
            out.append(
                f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" '
                f'stroke="{GREY}" stroke-opacity="0.35" stroke-width="1"/>'
            )
            text(x, y1 + 14, f"{t:g}", "middle")
            t += step
        for n, (label, colour, v) in enumerate(bars):
            ry = y0 + n * ROW + (ROW - BAR) // 2
            w = BARW * v / top
            text(x0 - 6, ry + BAR - 2, label, "end", size=10)
            out.append(f'<rect x="{x0}" y="{ry}" width="{w:.1f}" height="{BAR}" rx="2" fill="{colour}"/>')
            text(x0 + w + 5, ry + BAR - 2, f"{v:.2f}x", size=10)
    H = 6 + ((len(COLS) + 1) // 2) * PITCH + 2
    text(
        360, H,
        "SIMD speedup over scalar on the same machine. One machine per architecture; ratios only. median of 7.",
        "middle", size=11,
    )
    H += 10
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="{round(H * 1024 / 720)}" '
        f'viewBox="0 0 720 {H}" role="img" aria-label="SIMD speedup over scalar by input type and architecture.">\n'
        f'<rect width="720" height="{H}" fill="none"/>\n' + "\n".join(out) + "\n</svg>\n"
    )
    with open("docs/performance-summary.svg", "w") as f:
        f.write(svg)


if __name__ == "__main__":
    main()
