| GB/s (median of 7)           |     ascii |     latin |       cjk | box/mixed |     emoji |     dirty |
| simdutf C++ (strict)         |      1.28 |      0.28 |      0.37 |      0.30 |      0.27 |         - |
| simdutf8 + chars             |      0.15 |      0.09 |      0.11 |      0.09 |      0.11 |         - |
| std from_utf8 + chars        |      0.15 |      0.09 |      0.11 |      0.09 |      0.11 |         - |
| utf8_chunks + chars (C1)     |      0.09 |      0.09 |      0.10 |      0.09 |      0.11 |      0.07 |
| ours scalar slice fn         |      0.40 |      0.09 |      0.11 |      0.08 |      0.09 |         - |
| ours scalar lossy_into (C1)  |      0.40 |      0.09 |      0.11 |      0.08 |      0.09 |      0.08 |
| ours scalar Decoder pieces   |      0.67 |      0.09 |      0.11 |      0.08 |      0.09 |      0.05 |
| ours rvv slice fn            |      0.99 |      0.20 |      0.20 |      0.19 |      0.13 |         - |
| ours rvv lossy_into (C1)     |      1.14 |      0.20 |      0.20 |      0.19 |      0.13 |      0.08 |
| ours rvv Decoder pieces      |      2.59 |      0.22 |      0.20 |      0.20 |      0.13 |      0.03 |
