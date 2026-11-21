| GB/s (median of 7)           |     ascii |     latin |       cjk | box/mixed |     emoji |     dirty |
| simdutf C++ (strict)         |      4.03 |      0.45 |      0.71 |      0.38 |      0.36 |         - |
| simdutf8 + chars             |      0.75 |      0.30 |      0.37 |      0.33 |      0.41 |         - |
| std from_utf8 + chars        |      0.76 |      0.31 |      0.40 |      0.34 |      0.42 |         - |
| utf8_chunks + chars (C1)     |      0.14 |      0.17 |      0.29 |      0.18 |      0.22 |      0.20 |
| ours scalar slice fn         |      2.01 |      0.65 |      0.54 |      0.34 |      0.38 |         - |
| ours scalar lossy_into (C1)  |      2.01 |      0.63 |      0.54 |      0.34 |      0.39 |      0.33 |
| ours scalar Decoder pieces   |     10.45 |      0.60 |      0.54 |      0.34 |      0.39 |      0.18 |
| ours vsx slice fn            |      5.56 |      0.78 |      0.79 |      0.70 |      0.61 |         - |
| ours vsx lossy_into (C1)     |      5.58 |      0.78 |      0.78 |      0.71 |      0.59 |      0.37 |
| ours vsx Decoder pieces      |     13.44 |      0.77 |      0.77 |      0.70 |      0.59 |      0.10 |
