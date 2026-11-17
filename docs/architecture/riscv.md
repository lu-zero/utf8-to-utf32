# riscv64: the RVV kernel

`src/riscv64.rs` holds one kernel, `rvv_impl`. It is gated behind the `rvv`
Cargo feature and needs a nightly toolchain (`riscv_target_feature` and, with
`std`, `stdarch_riscv_feature_detection`). Default stable builds do not compile
it.

Rust's `core::arch` has no RVV intrinsics yet, so the vector work is inline
assembly. The hot loop — validation, ASCII widening and decode — is one
`global_asm!` function, `rvv_blocks`, which owns the block loop and returns to
Rust only on an error or at the end of the input. That lets the three lookup
tables be loaded into vector registers **once** instead of once per block, and
removes the compiler barriers between the phases. The error path stays in Rust
so `scalar::run` and `scalar::step` keep ownership of the error semantics.

Upstream tracks the missing intrinsics in `rust-lang/stdarch#1817`, blocked on
scalable vector types (`rust-lang/rust#145052`, RFC #3838). SVE has them on
nightly, RVV does not. Issue #1 in this repository records the compiler-side
gaps and the rationale for the global-asm approach.

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
2. ASCII is widened to u32 in one `e32, m4` group.
3. Otherwise validation runs at SEW=8 over **n** lanes and produces the first
   invalid position. Using `n` rather than `adv` here is essential: see below.
4. Decode computes candidate code points at SEW=8, then two unrolled SEW=32
   groups widen them with `vzext.vf4`, select a lead mask and compact with
   `vcompress.vm`.
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
missed. For the term to work, validation must run with **`vl = n`, not
`vl = adv`**: a claim by a lead at `adv - 1` lands on lane `adv`, which would be
invisible if `adv` were the vector length. `adv` is applied only as a mask on
the error result. Getting this wrong made the validator silently accept invalid
input when the hot loop was first moved into global asm.

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

At SEW=32 with **LMUL=2** the second phase has two unrolled groups, each
covering 16 bytes. Bytes from each buffer are widened with `vzext.vf4`,
combined as `lo | hi << 8 | top << 16`, and compacted with `vcompress.vm` under
a lead mask rebuilt from the widened `b0`. `vpopc.m` returns the character count
and the destination pointer advances by four times that count. LMUL=2 gives 16
u32 lanes at VLEN=256, so the two groups cover a 32-byte block. The lead mask is
the inverse of "continuation" (`b0 >= 0x80 && b0 < 0xC0`), ANDed with `idx < vl`
and built from `vid.v`. The whole vector is stored; lanes past the count are
overwritten by the next store, which is why `dst` must be at least `src.len()`
long.

This replaced an earlier design that computed full u32 candidates four times
(four SEW=32 iterations per block). The byte-domain form does that arithmetic
once at SEW=8; combined with the global-asm loop it took the non-ASCII rows from
0.05 to 0.10–0.11 GB/s on the X60.

An LMUL=4 variant would cover the block in one SEW=32 group. Widening alone
works at `e32, m4` with non-overlapping registers (the ASCII path uses it), but
`vcompress.vm` at m4 needs a mask group that overlaps the data group on this
CPU and traps, so decode stays at m2.

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
| RVV slice fn | 0.49 | 0.11 | 0.10 | 0.10 | 0.10 | – |
| RVV lossy_into (C1) | 0.81 | 0.11 | 0.10 | 0.10 | 0.10 | 0.08 |

The figures above are one run; the `bench/results/riscv.md` file is another,
and differ by a few hundredths from run to run on this machine. The
global-asm loop lifts every non-ASCII class from 0.07 to 0.10–0.11 GB/s
back-to-back on the same machine; ASCII is essentially unchanged (0.47 to 0.49,
within run variance) because that path was already a single tight asm block.
RVV now beats scalar on Latin (0.11 against 0.09), box (0.10 against 0.08) and
emoji (0.10 against 0.09), and is close behind on CJK (0.10 against 0.11) for
the reason given above. On ASCII it stays ahead at 0.49 against scalar's 0.40;
the `Decoder`'s ASCII path is 2.19 against scalar's 0.69.
Dirty input is about even in lossy mode; the `Decoder` is slower there (0.03
against 0.05), the same restart cost described in
[the main architecture page](../architecture.md).

The validator was changed from a ~90-instruction compare tree to three
`vrgather.vv` lookups with mask-register boolean logic. A microbenchmark of
2,000,000 calls on 32-byte CJK input ran in 236 ms against 363 ms for the
compare tree (1.54x) on this machine, same release build and tool. That gain is
not visible in the end-to-end table: the SEW=32 decode dominated until the
byte-domain rewrite, and the global-asm loop then lifted the non-ASCII rows to
0.10–0.11.

The ASCII path needed two changes: a cheap `vfirst.m` non-ASCII test before the
full validator (running it on every block had made ASCII 0.15 GB/s), and
widening with `e32, m4` in one instruction group instead of an m1 loop. The
source and destination must not share the `v0..=v3` register group: `v1`/`v2`
traps with SIGILL, while the first version's `v4`/`v8` and the current
global-asm `v16`/`v20` are safe. Going VLA kept throughput on this machine,
where a full register is already 32 bytes; it should use wider registers of
other implementations without a rewrite.

## Why the hot loop is `global_asm!`

The first implementation used separate `asm!` helpers per phase. That costs
three things the intrinsic-based NEON and VSX kernels do not pay:

- the three lookup tables reload on **every block** (`vle8.v` into fresh
  registers), where NEON loads them once at kernel entry;
- every `asm!` is a compiler barrier, so LLVM cannot schedule or allocate
  registers across the phases and emits its own `vsetvli` when it spills;
- the constants and pointers are re-materialised per helper.

`rvv_blocks` fixes all three: one function owns the block loop, so the tables
are loaded once at entry into `v28..=v30`, constants stay in registers, and no
compiler barrier sits between validation and decode. It returns
`(i, o, first_bad)` to Rust on an error; Rust computes the cut, uses the Rust
`decode_block` helper for the valid prefix, then `scalar::run`/`step` resolve
the error and re-enter. Error semantics therefore stay in `scalar`.

The source still needs `.option arch, +v` — LLVM does not enable the extension
for a bare `global_asm!` block. The tables are Rust statics referenced with
`sym` operands rather than in an assembly `.rodata` section.

## What still waits on compiler support

- **The decode stack buffers.** Four 32-byte buffers carry lo/hi/top/b0 from
  SEW=8 to SEW=32. Inside one asm function they could stay in vector registers,
  but the extension state and register groups have to be managed by hand;
  LMUL=2 groups overlap easily (writing `vle8.v v9` after `vzext.vf4 v8` corrupts
  the `v8..=v9` widened value), and a real mistake here produces SIGILL or
  SIGBUS rather than a compile error.
- **`vsetvli` state is not modelled by rustc.** Emitting `~{vtype}`/`~{vl}`
  clobbers means LLVM assumes the state dies at each `asm!` boundary. Real RVV
  intrinsics with scalable vector types (the work behind
  `rust-lang/stdarch#1817`, `rust-lang/rust#145052` and RFC #3838) would let the
  kernel be ordinary Rust like NEON/VSX. The vreg prototype in rustc PR #166108
  helps operand naming but does not model this state.
- **`vrgather.vv` is a general gather**, not a dedicated table shuffle like
  NEON `tbl` or VSX `vec_perm`. This one is hardware, not compiler.
