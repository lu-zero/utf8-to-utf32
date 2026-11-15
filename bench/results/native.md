active level (dispatch): Avx512
| GB/s (median of 7)           |     ascii |     latin |       cjk | box/mixed |     dirty |
| simdutf C++ (strict)         |      5.41 |      3.81 |      4.07 |      3.83 |         - |
| simdutf8 + chars             |      1.49 |      1.71 |      2.58 |      1.89 |         - |
| std from_utf8 + chars        |      1.49 |      0.96 |      0.84 |      0.91 |         - |
| utf8_chunks + chars (C1)     |      0.55 |      0.67 |      0.93 |      0.83 |      0.90 |
| ours scalar slice fn         |      5.72 |      1.81 |      1.44 |      1.58 |         - |
| ours scalar lossy_into (C1)  |      5.69 |      1.81 |      1.45 |      1.54 |      0.99 |
| ours scalar Decoder pieces   |     31.86 |      1.92 |      1.46 |      1.61 |      0.62 |
| ours sse4.1 slice fn         |      5.77 |      1.58 |      1.74 |      1.32 |         - |
| ours sse4.1 lossy_into (C1)  |      5.75 |      1.57 |      1.74 |      1.31 |      1.18 |
| ours sse4.1 Decoder pieces   |     41.21 |      1.54 |      1.74 |      1.33 |      0.56 |
| ours avx2 slice fn           |      6.17 |      2.95 |      3.01 |      2.85 |         - |
| ours avx2 lossy_into (C1)    |      6.18 |      2.95 |      3.00 |      2.83 |      0.97 |
| ours avx2 Decoder pieces     |     63.85 |      3.28 |      3.00 |      2.81 |      0.57 |
| ours avx512 slice fn         |      6.38 |      5.33 |      4.74 |      4.35 |         - |
| ours avx512 lossy_into (C1)  |      6.18 |      5.26 |      4.77 |      4.44 |      0.91 |
| ours avx512 Decoder pieces   |     84.28 |      5.14 |      4.81 |      4.64 |      0.61 |
