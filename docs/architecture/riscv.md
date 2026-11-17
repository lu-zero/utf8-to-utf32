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

The check is the Keiser–Lemire lookup scheme, as in the
[NEON](aarch64.md) and [VSX](ppc64.md) kernels. Three 16-byte tables are padded
to 32 bytes so one full-width `vle8.v` loads each. The previous byte (and the
two before it) come from `vslideup.vi` shifts of the loaded block; nibble
indices feed three `vrgather.vv` lookups whose AND is the error-class vector.

All boolean combining happens in **mask registers** with `vmand.mm`, `vmor.mm`,
`vmxor.mm` and `vmnot.m`. Doing it in data registers after `vmv.v.v` is wrong:
mask registers hold one bit per lane and the upper bits of a copied lane are
undefined, so AND/OR on those bytes produces spurious errors. The check:

- `must` is set where a 3-byte lead two back or a 4-byte lead three back needs
  a continuation here. XOR with the lookup's `TWO_CONTS` bit catches both a
  missing continuation and an unexpected one.
- A 4-byte lead one back is invalid outright (C0, C1 and F5+ too).

Overlong, surrogate and above-U+10FFFF sequences need no explicit range check:
the table entries already distinguish the second-byte classes after E0, ED, F0
and F4. An earlier version carried those compares over from the x86 bitmask
scheme; removing them kept conformance and the exhaustive sweep green.

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

Decoding is two phases. At SEW=8, one pass over the whole block computes the
three bytes of every candidate code point. The length class comes from three
unsigned compares against C0, E0 and F0; `vmerge.vvm` selects among the 1-, 2-,
3- and 4-byte candidates for each byte:

| Byte | 1-byte | 2-byte | 3-byte | 4-byte |
|---|---|---|---|---|
| low | `b0` | `(b0 << 6) \| (b1 & 0x3F)` | `(b1 << 6) \| (b2 & 0x3F)` | `(b2 << 6) \| (b3 & 0x3F)` |
| high | 0 | `(b0 >> 2) & 7` | `((b0 & 0x0F) << 4) \| ((b1 & 0x3C) >> 2)` | `((b1 & 0x0F) << 4) \| ((b2 & 0x3C) >> 2)` |
| top | 0 | 0 | 0 | `((b0 & 7) << 2) \| ((b1 & 0x30) >> 4)` |

`vsll`/`vsrl` at SEW=8 keep only the low byte of each shift, which is exactly
the byte needed. The three byte vectors and the original `b0` are stored to
four 32-byte stack buffers.

At SEW=32 with **LMUL=2** the second phase runs a short loop: `vl` bytes from
each buffer are widened with `vzext.vf4`, combined as `lo | hi << 8 | top << 16`,
and compacted with `vcompress.vm` under a lead mask rebuilt from the widened
`b0`. `vpopc.m` returns the character count. LMUL=2 gives 16 u32 lanes at
VLEN=256, halving the loop to two iterations per block; the simpler byte-domain
arithmetic leaves enough registers for the wider groups. The lead mask is the inverse of "continuation"
(`b0 >= 0x80 && b0 < 0xC0`), ANDed with `idx < rem` and built from `vid.v`.
The whole vector is stored; lanes past the count are overwritten by the next
store, which is why `dst` must be at least `src.len()` long.

This replaced an earlier design that computed full u32 candidates four times
(four SEW=32 iterations per block). The byte-domain form does that arithmetic
once at SEW=8 and reduces the non-ASCII benchmark rows from 0.05 to 0.07 GB/s
on the X60, with ASCII unchanged.

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
| RVV slice fn | 0.61 | 0.09 | 0.09 | 0.09 | 0.09 | – |
| RVV lossy_into (C1) | 0.82 | 0.09 | 0.09 | 0.09 | 0.09 | 0.08 |

RVV wins on ASCII (1.53x for the slice function, 2.0x for `lossy_into`) and the
`Decoder`'s ASCII path is 2.19 against scalar's 0.69. Non-ASCII input is now
about equal to scalar at 0.09 GB/s: removing the redundant range checks and
halving the widen loop with LMUL=2 lifted it from 0.07. CJK remains the one
class where scalar is ahead (0.11 against 0.09). ASCII varies by about 20%
between runs on this machine (the table above and `bench/results/riscv.md`
come from different runs); the back-to-back comparison used for the commit
numbers had 0.61 for both the previous and this version. Dirty input is about even in lossy mode; the
`Decoder` is slower there (0.02 against 0.05), the same restart cost described
in [the main architecture page](../architecture.md).

The validator was changed from a ~90-instruction compare tree to three
`vrgather.vv` lookups with mask-register boolean logic. A microbenchmark of
2,000,000 calls on 32-byte CJK input ran in 236 ms against 363 ms for the
compare tree (1.54x) on this machine, same release build and tool. On the
first end-to-end run the non-ASCII rows stayed at 0.05 GB/s because the SEW=32
decode loop dominated; after the byte-domain decode and LMUL=2 they reached
0.09 GB/s.

The ASCII path needed two changes: a cheap `vfirst.m` non-ASCII test before the
full validator (running it on every block had made ASCII 0.15 GB/s), and
widening with `e32, m4` in one instruction group instead of an m1 loop. The
first version used `v1`/`v2` for the widening, which overlap the `v0..=v3`
register group and trap with SIGILL; using `v4` as the byte source and `v8` as
the widened destination avoids the overlap. LMUL=4 with `vcompress.vm` does trap
on this CPU, so decoding uses m2. Going VLA kept throughput on this machine,
where a full register is already 32 bytes; it should use wider registers of
other implementations without a rewrite.

## Where the performance is hiding

The remaining gap to the NEON and VSX kernels (1.4–2.8x over scalar on their
machines, against roughly 1x here on non-ASCII) is not in the algorithms: this
kernel now uses the same table lookups, byte-domain arithmetic and mask-driven
compaction. It is in what the compiler lets the assembly do.

- **Tables reload per block.** `vle8.v` loads B1_HIGH, B1_LOW and B2_HIGH into
  vector registers on every 32-byte block. NEON loads its three tables once at
  kernel entry. Keeping them across blocks needs vector values in Rust locals,
  which today's `asm!` cannot express (vreg is clobber-only) or a hand-written
  assembly function that owns the whole loop.
- **Stack round-trip in decode.** Four 32-byte buffers carry lo/hi/top/b0 from
  SEW=8 to SEW=32 because no vector value can cross the SEW change inside Rust.
  In NEON a `tbl` with a mask-indexed table compacts and widens in one step.
- **Every `asm!` is an optimisation barrier.** LLVM cannot schedule across the
  helpers, cannot allocate registers across them, and inserts its own `vsetvli`
  when it spills. The intrinsic-based kernels have none of these barriers.
- **`vsetvli` state is deliberately not modelled.** Rust emits `~{vtype}` and
  `~{vl}` clobbers for RISC-V vector asm, so LLVM assumes the state does not
  survive a block. A compiler change that models `vtype`/`vl` as trackable
  state, or real RVV intrinsics with scalable vector types (the work behind
  `rust-lang/stdarch#1817` and RFC #3838), would let the helpers be ordinary
  Rust code. The prototype vreg work in rustc PR #166108 helps with operand
  naming but does not change this.
- **LMUL=4 decode is blocked by register groups.** At `e32, m4` a mask group
  overlaps the widened data group when `vcompress.vm` runs, raising SIGILL on
  the X60. Fixed vector types with proper LMUL-aware allocation would let the
  SEW=32 loop cover the whole block in one iteration.
- **`vrgather.vv` is a general gather**, not a dedicated table shuffle like
  NEON `tbl` or VSX `vec_perm`. This one is hardware, not compiler.

A hand-written kernel via `global_asm!` could address the first three points
today: one function owns the block loop, so tables, constants and pointers can
be hoisted into registers once, and no compiler barrier sits between the
validation and decode phases. The cost is that the error path must still reach
`scalar::run` and `scalar::step` — writing that boundary in assembly means
matching the Rust calling convention by hand, or duplicating the error logic
(and the crate deliberately keeps error semantics in `scalar`). The practical
shape would be one global-asm function that loops over whole blocks until an
error, returning `(i, o, error_position)` to Rust, which then resolves the
error and re-enters. That is a larger change than a tuning pass and has not
been attempted here.
