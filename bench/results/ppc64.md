| GB/s (median of 7)           |     ascii |     latin |       cjk | box/mixed |     emoji |     dirty |
| simdutf C++ (strict)         |      4.01 |      0.45 |      0.71 |      0.39 |      0.36 |         - |
| simdutf8 + chars             |      0.74 |      0.31 |      0.37 |      0.35 |      0.39 |         - |
| std from_utf8 + chars        |      0.73 |      0.31 |      0.40 |      0.35 |      0.41 |         - |
| utf8_chunks + chars (C1)     |      0.14 |      0.17 |      0.29 |      0.18 |      0.22 |      0.20 |
| ours scalar slice fn         |      2.01 |      0.63 |      0.54 |      0.54 |      0.48 |         - |
| ours scalar lossy_into (C1)  |      2.01 |      0.63 |      0.54 |      0.54 |      0.48 |      0.32 |
| ours scalar Decoder pieces   |     10.45 |      0.59 |      0.54 |      0.53 |      0.48 |      0.18 |
| ours vsx slice fn            |      3.33 |      0.68 |      0.74 |      0.63 |      0.48 |         - |
| ours vsx lossy_into (C1)     |      3.33 |      0.68 |      0.74 |      0.63 |      0.48 |      0.40 |
| ours vsx Decoder pieces      |     10.96 |      0.67 |      0.74 |      0.63 |      0.48 |      0.09 |
