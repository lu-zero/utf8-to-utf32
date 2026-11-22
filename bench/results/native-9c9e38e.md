active level (dispatch): Avx512
| GB/s (median of 9)           |     ascii |     latin |       cjk | box/mixed |     emoji |     dirty |
| simdutf C++ (strict)         |      5.32 |      3.79 |      4.02 |      3.78 |      3.78 |         - |
| simdutf8 + chars             |      1.49 |      1.70 |      2.55 |      1.86 |      2.23 |         - |
| std from_utf8 + chars        |      1.48 |      0.94 |      1.16 |      0.96 |      1.06 |         - |
| utf8_chunks + chars (C1)     |      0.74 |      0.62 |      1.07 |      0.73 |      0.86 |      0.79 |
| ours scalar slice fn         |      5.82 |      1.80 |      1.43 |      1.45 |      1.76 |         - |
| ours scalar lossy_into (C1)  |      5.79 |      1.79 |      1.42 |      1.44 |      1.73 |      0.98 |
| ours scalar Decoder pieces   |     30.08 |      1.88 |      1.42 |      1.47 |      1.74 |      0.60 |
| ours sse4.1 slice fn         |      6.48 |      2.19 |      1.57 |      1.48 |      1.15 |         - |
| ours sse4.1 lossy_into (C1)  |      6.51 |      2.20 |      1.57 |      1.49 |      1.15 |      1.10 |
| ours sse4.1 Decoder pieces   |     40.09 |      2.17 |      1.56 |      1.49 |      1.15 |      0.40 |
| ours avx2 slice fn           |      6.34 |      3.44 |      3.24 |      3.24 |      1.97 |         - |
| ours avx2 lossy_into (C1)    |      6.40 |      3.44 |      3.23 |      3.24 |      1.96 |      1.09 |
| ours avx2 Decoder pieces     |     72.76 |      3.41 |      3.22 |      3.22 |      1.95 |      0.55 |
| ours avx512 slice fn         |      6.37 |      5.95 |      5.79 |      5.49 |      3.69 |         - |
| ours avx512 lossy_into (C1)  |      6.44 |      5.89 |      5.80 |      5.51 |      3.71 |      1.11 |
| ours avx512 Decoder pieces   |     81.99 |      5.97 |      5.73 |      5.42 |      3.73 |      0.59 |
