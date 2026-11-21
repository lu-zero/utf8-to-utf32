| GB/s (median of 7)           |     ascii |     latin |       cjk | box/mixed |     emoji |     dirty |
| simdutf C++ (strict)         |      9.70 |      1.36 |      2.01 |      1.11 |      1.19 |         - |
| simdutf8 + chars             |      0.83 |      0.82 |      1.10 |      0.85 |      0.95 |         - |
| std from_utf8 + chars        |      0.82 |      0.46 |      0.55 |      0.47 |      0.54 |         - |
| utf8_chunks + chars (C1)     |      0.42 |      0.45 |      0.58 |      0.48 |      0.59 |      0.46 |
| ours scalar slice fn         |      4.52 |      1.06 |      0.78 |      0.82 |      0.91 |         - |
| ours scalar lossy_into (C1)  |      4.52 |      1.06 |      0.78 |      0.82 |      0.91 |      0.88 |
| ours scalar Decoder pieces   |     13.57 |      1.06 |      0.78 |      0.82 |      0.91 |      0.33 |
| ours neon slice fn           |      8.02 |      2.19 |      1.92 |      1.87 |      1.31 |         - |
| ours neon lossy_into (C1)    |      8.03 |      2.19 |      1.92 |      1.87 |      1.32 |      0.82 |
| ours neon Decoder pieces     |     18.81 |      2.18 |      1.90 |      1.78 |      1.31 |      0.21 |
