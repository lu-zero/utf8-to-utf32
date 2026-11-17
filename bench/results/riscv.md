| GB/s (median of 7)           |     ascii |     latin |       cjk | box/mixed |     emoji |     dirty |
| simdutf C++ (strict)         |      0.40 |      0.04 |      0.07 |      0.04 |      0.05 |         - |
| simdutf8 + chars             |      0.14 |      0.09 |      0.11 |      0.09 |      0.10 |         - |
| std from_utf8 + chars        |      0.14 |      0.09 |      0.11 |      0.09 |      0.10 |         - |
| utf8_chunks + chars (C1)     |      0.08 |      0.09 |      0.10 |      0.09 |      0.10 |      0.07 |
| ours scalar slice fn         |      0.40 |      0.09 |      0.11 |      0.08 |      0.09 |         - |
| ours scalar lossy_into (C1)  |      0.40 |      0.09 |      0.11 |      0.08 |      0.09 |      0.08 |
| ours scalar Decoder pieces   |      0.69 |      0.09 |      0.11 |      0.08 |      0.09 |      0.05 |
| ours rvv slice fn            |      0.51 |      0.05 |      0.05 |      0.05 |      0.05 |         - |
| ours rvv lossy_into (C1)     |      0.72 |      0.05 |      0.05 |      0.05 |      0.05 |      0.08 |
| ours rvv Decoder pieces      |      2.19 |      0.05 |      0.05 |      0.05 |      0.05 |      0.02 |
