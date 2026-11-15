active level (dispatch): Avx512
| GB/s (median of 7)           |     ascii |     latin |       cjk | box/mixed |     dirty |
| simdutf C++ (strict)         |      5.24 |      3.80 |      3.99 |      3.80 |         - |
| simdutf8 + chars             |      1.45 |      1.59 |      2.48 |      1.74 |         - |
| std from_utf8 + chars        |      1.50 |      0.93 |      0.97 |      0.94 |         - |
| utf8_chunks + chars (C1)     |      0.52 |      0.59 |      0.98 |      0.65 |      1.04 |
| ours scalar slice fn         |      5.22 |      1.34 |      1.08 |      1.07 |         - |
| ours scalar lossy_into (C1)  |      5.07 |      1.27 |      1.05 |      1.04 |      0.96 |
| ours scalar Decoder pieces   |     15.60 |      1.33 |      1.06 |      1.02 |      0.60 |
| ours sse4.1 slice fn         |      6.54 |      1.55 |      1.59 |      1.29 |         - |
| ours sse4.1 lossy_into (C1)  |      6.50 |      1.54 |      1.58 |      1.29 |      1.06 |
| ours sse4.1 Decoder pieces   |     41.20 |      1.52 |      1.59 |      1.30 |      0.55 |
| ours avx2 slice fn           |      6.52 |      2.77 |      2.81 |      2.70 |         - |
| ours avx2 lossy_into (C1)    |      6.52 |      2.76 |      2.82 |      2.69 |      1.09 |
| ours avx2 Decoder pieces     |     75.33 |      3.00 |      2.80 |      2.70 |      0.55 |
| ours avx512 slice fn         |      6.76 |      5.23 |      4.81 |      4.63 |         - |
| ours avx512 lossy_into (C1)  |      6.56 |      5.09 |      4.79 |      4.63 |      1.04 |
| ours avx512 Decoder pieces   |     84.01 |      4.97 |      4.73 |      4.70 |      0.59 |
