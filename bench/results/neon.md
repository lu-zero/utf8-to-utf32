| GB/s (median of 7)           |     ascii |     latin |       cjk | box/mixed |     emoji |     dirty |
| simdutf C++ (strict)         |     10.46 |      1.36 |      2.01 |      1.17 |      1.18 |         - |
| simdutf8 + chars             |      0.83 |      0.82 |      1.10 |      0.85 |      0.95 |         - |
| std from_utf8 + chars        |      0.82 |      0.47 |      0.55 |      0.47 |      0.54 |         - |
| utf8_chunks + chars (C1)     |      0.42 |      0.32 |      0.58 |      0.48 |      0.59 |      0.46 |
| ours scalar slice fn         |      4.53 |      1.06 |      0.79 |      0.82 |      0.91 |         - |
| ours scalar lossy_into (C1)  |      4.53 |      1.06 |      0.78 |      0.82 |      0.91 |      0.88 |
| ours scalar Decoder pieces   |     13.59 |      1.06 |      0.78 |      0.82 |      0.91 |      0.33 |
| ours neon slice fn           |      8.05 |      2.16 |      1.96 |      1.85 |      1.32 |         - |
| ours neon lossy_into (C1)    |      8.05 |      2.16 |      1.96 |      1.85 |      1.32 |      0.82 |
| ours neon Decoder pieces     |     18.83 |      2.11 |      1.93 |      1.83 |      1.31 |      0.21 |
