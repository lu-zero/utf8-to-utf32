active level (dispatch): Avx512
| GB/s (median of 9)           |     ascii |     latin |       cjk | box/mixed |     emoji |     dirty |
| simdutf C++ (strict)         |      5.57 |      3.78 |      4.01 |      3.79 |      3.78 |         - |
| simdutf8 + chars             |      1.50 |      1.70 |      2.57 |      1.86 |      2.24 |         - |
| std from_utf8 + chars        |      1.49 |      0.82 |      1.18 |      0.91 |      0.95 |         - |
| utf8_chunks + chars (C1)     |      0.91 |      0.72 |      1.08 |      0.84 |      0.99 |      0.84 |
| ours scalar slice fn         |      5.97 |      1.80 |      1.19 |      1.52 |      1.71 |         - |
| ours scalar lossy_into (C1)  |      5.96 |      1.81 |      1.21 |      1.54 |      1.74 |      0.99 |
| ours scalar Decoder pieces   |     31.66 |      1.84 |      1.10 |      1.42 |      1.70 |      0.73 |
| ours sse4.1 slice fn         |      6.65 |      2.20 |      1.57 |      1.49 |      1.14 |         - |
| ours sse4.1 lossy_into (C1)  |      6.64 |      2.20 |      1.57 |      1.49 |      1.15 |      1.00 |
| ours sse4.1 Decoder pieces   |     31.95 |      2.17 |      1.57 |      1.49 |      1.14 |      0.75 |
| ours avx2 slice fn           |      6.46 |      3.46 |      3.24 |      3.23 |      1.96 |         - |
| ours avx2 lossy_into (C1)    |      6.27 |      3.45 |      3.24 |      3.21 |      1.96 |      0.95 |
| ours avx2 Decoder pieces     |     74.81 |      3.43 |      3.23 |      3.21 |      1.96 |      0.75 |
| ours avx512 slice fn         |      6.47 |      5.95 |      5.81 |      5.45 |      3.69 |         - |
| ours avx512 lossy_into (C1)  |      6.46 |      5.96 |      5.81 |      5.52 |      3.69 |      1.14 |
| ours avx512 Decoder pieces   |     83.11 |      6.04 |      5.74 |      5.36 |      3.69 |      0.74 |
