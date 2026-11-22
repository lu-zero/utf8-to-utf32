active level (dispatch): Avx512
| GB/s (median of 7)           |     ascii |     latin |       cjk | box/mixed |     emoji |     dirty |
| simdutf C++ (strict)         |      5.02 |      3.80 |      4.03 |      3.78 |      3.79 |         - |
| simdutf8 + chars             |      1.50 |      1.48 |      2.53 |      1.76 |      2.07 |         - |
| std from_utf8 + chars        |      1.49 |      0.79 |      1.18 |      0.88 |      1.01 |         - |
| utf8_chunks + chars (C1)     |      0.58 |      0.76 |      0.96 |      0.78 |      0.96 |      1.01 |
| ours scalar slice fn         |      5.17 |      1.79 |      1.21 |      1.44 |      1.62 |         - |
| ours scalar lossy_into (C1)  |      5.11 |      1.76 |      1.20 |      1.40 |      1.63 |      0.96 |
| ours scalar Decoder pieces   |     31.46 |      1.95 |      1.25 |      1.41 |      1.43 |      0.78 |
| ours sse4.1 slice fn         |      6.82 |      2.11 |      1.60 |      1.54 |      1.18 |         - |
| ours sse4.1 lossy_into (C1)  |      6.93 |      2.10 |      1.50 |      1.54 |      1.22 |      1.04 |
| ours sse4.1 Decoder pieces   |     41.18 |      2.13 |      1.25 |      1.54 |      1.22 |      0.79 |
| ours avx2 slice fn           |      6.72 |      3.49 |      2.21 |      2.89 |      1.70 |         - |
| ours avx2 lossy_into (C1)    |      6.77 |      3.51 |      2.28 |      2.86 |      1.66 |      1.19 |
| ours avx2 Decoder pieces     |     75.30 |      3.45 |      2.88 |      2.86 |      1.71 |      0.74 |
| ours avx512 slice fn         |      6.80 |      5.74 |      5.88 |      5.66 |      3.67 |         - |
| ours avx512 lossy_into (C1)  |      6.82 |      6.05 |      5.80 |      5.62 |      3.67 |      1.08 |
| ours avx512 Decoder pieces   |     83.87 |      6.19 |      5.87 |      5.62 |      3.64 |      0.77 |
