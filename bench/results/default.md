active level (dispatch): Avx512
| GB/s (median of 7)           |     ascii |     latin |       cjk | box/mixed |     emoji |     dirty |
| simdutf C++ (strict)         |      5.57 |      3.80 |      4.02 |      3.80 |      3.55 |         - |
| simdutf8 + chars             |      1.51 |      1.52 |      2.51 |      1.75 |      2.00 |         - |
| std from_utf8 + chars        |      1.51 |      0.87 |      1.17 |      0.96 |      0.96 |         - |
| utf8_chunks + chars (C1)     |      0.52 |      0.68 |      0.95 |      0.76 |      0.85 |      0.90 |
| ours scalar slice fn         |      5.18 |      1.62 |      1.40 |      1.41 |      1.69 |         - |
| ours scalar lossy_into (C1)  |      5.22 |      1.68 |      1.36 |      1.41 |      1.71 |      0.97 |
| ours scalar Decoder pieces   |     30.12 |      1.75 |      1.38 |      1.43 |      1.69 |      0.60 |
| ours sse4.1 slice fn         |      6.88 |      2.13 |      1.58 |      1.54 |      1.23 |         - |
| ours sse4.1 lossy_into (C1)  |      6.79 |      2.04 |      1.58 |      1.51 |      1.23 |      1.19 |
| ours sse4.1 Decoder pieces   |     32.93 |      2.13 |      1.58 |      1.47 |      1.23 |      0.42 |
| ours avx2 slice fn           |      6.58 |      3.52 |      2.82 |      2.83 |      1.72 |         - |
| ours avx2 lossy_into (C1)    |      6.60 |      3.51 |      2.82 |      2.86 |      1.72 |      1.21 |
| ours avx2 Decoder pieces     |     71.17 |      3.45 |      2.86 |      2.88 |      1.73 |      0.52 |
| ours avx512 slice fn         |      6.61 |      6.14 |      5.96 |      5.64 |      3.70 |         - |
| ours avx512 lossy_into (C1)  |      6.64 |      6.10 |      5.87 |      5.66 |      3.69 |      1.10 |
| ours avx512 Decoder pieces   |     80.00 |      5.88 |      5.83 |      5.53 |      3.71 |      0.58 |
