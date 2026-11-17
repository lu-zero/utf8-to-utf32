# riscv64: the RVV kernel

`src/riscv64.rs` holds one kernel, `rvv_impl`. It is gated behind the `rvv`
Cargo feature and needs a nightly toolchain (`riscv_target_feature` and, with
`std`, `stdarch_riscv_feature_detection`). Default stable builds do not compile
it.

Rust's `core::arch` has no RVV intrinsics yet, so the vector work is inline
`asm!`. Vector registers can only be clobbers in Rust inline assembly, so each
helper sets its own `vsetvli`; the decode loop and its pointer updates live
inside one asm block. Upstream tracks the missing intrinsics in
`rust-lang/stdarch#1817`, blocked on scalable vector types
(`rust-lang/rust#145052`, RFC #3838). SVE has them on nightly, RVV does not.

The kernel is **vector-length agnostic**. `vlmax` is read once at entry, and
each iteration takes `n = min(remaining - 3, vlmax)` bytes. `Level::is_supported`
requires the V extension (with `std`) and `vlmax >= 32`; that is a conservative
minimum for the tested hardware, not a fixed block width.

## Overview

There is no fixed block size. Each iteration sets `vl` from `n` bytes, where the
three bytes not counted are lookahead for the shifted loads. `adv` is `n`,
shortened by up to three when a character starts near the end of the region;
`i` advances by `adv`, so every block starts on a character boundary and no
character is decoded before its last byte has been validated. The rest goes to
the scalar tail.

1. A cheap SEW=8 `vmsgeu` + `vfirst.m` test detects an all-ASCII region.
2. ASCII is widened to u32 at SEW=32, looping by the runtime `vl`.
3. Otherwise validation runs at SEW=8 and produces the first invalid position.
4. The decode loop runs at SEW=32: `vzext.vf4` widens the source bytes into u32
   lanes, code-point arithmetic computes a candidate for every position, a lead
   mask selects the real ones, and `vcompress.vm` compacts them.
5. A failing block decodes the characters before the cut with SIMD, then scalar
   resolves one character and covers a growing stretch before SIMD retries.

## Validation

The check is the one the [x86 kernels](x86.md) use, in the vector domain. For
each byte position `j` it classifies `b0` and compares it with the bytes one,
two and three positions back (obtained with `vslideup.vi`):

- `cont` is a continuation byte (`0x80..=0xBF`) that is not a lead.
- `claimed` is set when a preceding lead claims this position: a lead one byte
  back, a 3-byte lead two back, or a 4-byte lead three back.
- `cont ^ claimed` is bad: a continuation nobody claimed, or a byte a lead
  claimed that is not a continuation.
- C0, C1 and F5+ leads are invalid outright.
- After E0, ED, F0 and F4 the next byte must be in range (overlong, surrogate
  and above-U+10FFFF checks).

The first bad lane marks where a problem becomes *visible*, which can be one
byte after the offending lead. The kernel never uses it as an error position:
it finds the last lead strictly before it and lets `scalar::run` and
`scalar::step` produce the exact `valid_up_to` and `error_len`.

A lead inside the block whose continuation lies past the shortened advance is
also bad. Without this term a lead just before the cut could be decoded with a
continuation that belongs to the next block, and the invalid sequence would be
missed.

`tests/conformance.rs` and `tests/exhaustive_neon.rs` run every supported level,
so the RVV validator is checked against `core::str::from_utf8` and
`utf8_chunks` on random, structured and exhaustive inputs, including block
boundaries. On a riscv64 machine run `cargo +nightly test --release --features
rvv` after a change to the validator or a decode path.

## Decoding

At SEW=32 one loop iteration covers `vl` byte positions, where `vl` is the
number of u32 lanes the hardware has. Four `vle8.v` loads are widened with
`vzext.vf4`, so every candidate code point has its four source bytes in u32
lanes. The code point is selected by the length class of `b0`:

| Length | Code point |
|---|---|
| 1 | `b0` |
| 2 | `((b0 & 0x1F) << 6) \| (b1 & 0x3F)` |
| 3 | `((b0 & 0x0F) << 12) \| ((b1 & 0x3F) << 6) \| (b2 & 0x3F)` |
| 4 | `((b0 & 7) << 18) \| ((b1 & 0x3F) << 12) \| ((b2 & 0x3F) << 6) \| (b3 & 0x3F)` |

The length class comes from three unsigned compares against C0, E0 and F0;
`vmerge.vvm` picks the matching result. The lead mask is the inverse of
"continuation" (`b0 >= 0x80 && b0 < 0xC0`), ANDed with `idx < rem` and built
from `vid.v`. `vpopc.m` returns the number of output characters and
`vcompress.vm` packs the selected lanes into the first positions of the
destination vector. The whole vector is stored; lanes past the count are
overwritten by the next store, which is why `dst` must be at least `src.len()`
long. The loop then advances the source by `vl`, the destination by four times
the character count, and `rem` by `vl`.

An LMUL=4 variant would cover the whole region in one SEW=32 iteration and
remove the loop, but `vzext.vf4` with `e32, m4` raises SIGILL on the Spacemit
X60, so the m1 loop is kept.

## Error-dense input

`ERR_STRETCH` scalar bytes are decoded after a failing block, doubled for each
consecutive failure up to `ERR_BACKOFF_MAX`. This follows the other kernels:
validation, prefix decode and a restart on every error would otherwise dominate
on input with an error every few bytes.

## Measured behaviour

Measured on `f3_4`, a Spacemit X60 riscv64 machine with the V extension, with
Rust 1.102 nightly, an LTO release build and `--features rvv`. Each figure is
GB/s of UTF-8 input, the median of 7 runs of about 1 MB. The full table,
including the other implementations, is in `bench/results/riscv.md`; it came
from `cargo +nightly run --release -p utf32-bench --features utf8-to-utf32/rvv`.

| GB/s | ascii | latin | cjk | box/mixed | emoji | dirty |
|---|---|---|---|---|---|---|
| scalar slice fn | 0.40 | 0.09 | 0.11 | 0.08 | 0.09 | – |
| RVV slice fn | 0.51 | 0.05 | 0.05 | 0.05 | 0.05 | – |
| RVV lossy_into (C1) | 0.72 | 0.05 | 0.05 | 0.05 | 0.05 | 0.08 |

RVV wins on ASCII (1.28x for the slice function, 1.8x for `lossy_into`) and the
`Decoder`'s ASCII path is 2.19 against scalar's 0.69. Non-ASCII input is about
half the scalar speed: validation at SEW=8 plus the SEW=32 decode loop costs
far more instructions than the scalar decoder uses for the same bytes. Dirty
input is about even in lossy mode; the `Decoder` is slower there (0.02 against
0.05), the same restart cost described in
[the main architecture page](../architecture.md).

The ASCII path needed a cheap `vfirst.m` non-ASCII test before the full
validator: running the validator on every block had made ASCII 0.15 GB/s, well
below scalar. Going VLA kept the same throughput on this machine, where a full
register is already 32 bytes; it should use the wider registers of other
implementations without a rewrite. A full-width LMUL=4 form would remove the
SEW=32 loop but traps on this CPU; see above.
