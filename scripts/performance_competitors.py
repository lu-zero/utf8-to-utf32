#!/usr/bin/env python3
"""Write a cross-architecture comparison against the other implementations:

    scripts/performance_competitors.py

For each input class the graph shows our best SIMD level's throughput divided
by the fastest other implementation on the same machine. Absolute GB/s from
different machines are not comparable, so the graph only compares ratios. It
also prints the table as markdown.

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
    ("x86 / best SIMD", "bench/results/default.md", "#5faf5f"),
    ("aarch64 / NEON", "bench/results/neon.md", "#8a7faf"),
    ("POWER9 / VSX", "bench/results/ppc64.md", "#3f8f6f"),
    ("Spacemit X60 / RVV", "bench/results/riscv.md", "#8f7f5f"),
]
CLEAN_COMPETITORS = [
    ("simdutf", "simdutf C++ (strict)"),
    ("simdutf8", "simdutf8 + chars"),
    ("std", "std from_utf8 + chars"),
]
DIRTY_COMPETITOR = ("utf8_chunks", "utf8_chunks + chars (C1)")


def ours_value(rows, ci, dirty):
    suffix = "lossy_into (C1)" if dirty else "slice fn"
    vals = [
        v[ci]
        for k, v in rows.items()
        if k.startswith("ours ")
        and not k.startswith("ours scalar")
        and k.endswith(" " + suffix)
        and v[ci] is not None
    ]
    return max(vals) if vals else None


def best_competitor(rows, ci, dirty):
    candidates = [DIRTY_COMPETITOR] if dirty else CLEAN_COMPETITORS
    vals = [(short, value(rows, name, ci, dirty)) for short, name in candidates]
    vals = [(short, v) for short, v in vals if v is not None]
    return max(vals, key=lambda t: t[1]) if vals else None


def ratios():
    data = {}
    for label, path, colour in ARCHES:
        cols, rows = parse(path)
        data[label] = (colour, {})
        for col in COLS:
            if col not in cols:
                continue
            ci = cols.index(col)
            dirty = col == "dirty"
            ours = ours_value(rows, ci, dirty)
            other = best_competitor(rows, ci, dirty)
            if ours is not None and other:
                ref, other_v = other
                data[label][1][col] = (ours / other_v, ref)
    return data


def main():
    data = ratios()
    print("| Our SIMD vs best other | " + " | ".join(TITLES.get(c, c) for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for label, (_, values) in data.items():
        cells = [f"{values[c][0]:.2f}x {values[c][1]}" if c in values else "–" for c in COLS]
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
        bars = [(label, colour, *values[col]) for label, (colour, values) in data.items() if col in values]
        top, step = nice_top(max(v for _, _, v, _ in bars))
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
        for n, (label, colour, v, ref) in enumerate(bars):
            ry = y0 + n * ROW + (ROW - BAR) // 2
            w = BARW * v / top
            text(x0 - 6, ry + BAR - 2, label, "end", size=10)
            out.append(f'<rect x="{x0}" y="{ry}" width="{w:.1f}" height="{BAR}" rx="2" fill="{colour}"/>')
            text(x0 + w + 5, ry + BAR - 2, f"{v:.2f}x {ref}", size=9)
    H = 6 + ((len(COLS) + 1) // 2) * PITCH + 2
    text(
        360, H,
        "Our best SIMD level over the fastest other implementation on the same machine; "
        "the reference is named after each ratio (utf8_chunks only for dirty input). "
        "One machine per architecture; ratios only. median of 7.",
        "middle", size=10,
    )
    H += 10
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="{round(H * 1024 / 720)}" '
        f'viewBox="0 0 720 {H}" role="img" aria-label="Our SIMD throughput over the fastest other implementation, '
        f'by input type and architecture.">\n'
        f'<rect width="720" height="{H}" fill="none"/>\n' + "\n".join(out) + "\n</svg>\n"
    )
    with open("docs/performance-competitors.svg", "w") as f:
        f.write(svg)


if __name__ == "__main__":
    main()
