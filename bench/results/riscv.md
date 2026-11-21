| GB/s (median of 7)           |     ascii |     latin |       cjk | box/mixed |     emoji |     dirty |
| simdutf C++ (strict)         |      1.29 |      0.28 |      0.37 |      0.30 |      0.27 |         - |
| simdutf8 + chars             |      0.14 |      0.09 |      0.11 |      0.09 |      0.10 |         - |
| std from_utf8 + chars        |      0.14 |      0.09 |      0.11 |      0.09 |      0.10 |         - |
| utf8_chunks + chars (C1)     |      0.08 |      0.09 |      0.10 |      0.09 |      0.10 |      0.07 |
| ours scalar slice fn         |      0.39 |      0.09 |      0.11 |      0.08 |      0.09 |         - |
| ours scalar lossy_into (C1)  |      0.39 |      0.09 |      0.11 |      0.08 |      0.09 |      0.08 |
| ours scalar Decoder pieces   |      0.70 |      0.09 |      0.11 |      0.08 |      0.09 |      0.05 |
| ours rvv slice fn            |      1.30 |      0.20 |      0.20 |      0.19 |      0.19 |         - |
| ours rvv lossy_into (C1)     |      1.29 |      0.20 |      0.20 |      0.19 |      0.21 |      0.08 |
| ours rvv Decoder pieces      |      2.59 |      0.21 |      0.20 |      0.20 |      0.21 |      0.03 |
