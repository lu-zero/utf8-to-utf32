# powerpc64: the VSX kernel

`src/powerpc64.rs` holds one kernel, `vsx_impl`. It is gated behind the `vsx`
Cargo feature and needs a nightly toolchain (`stdarch_powerpc`,
`powerpc_target_feature` and, with `std`, `stdarch_powerpc_feature_detection`).
Default stable builds do not compile it. The kernel enables `power9-vector`,
so `vec_xl`/`vec_xst` lower to `lxv`/`stxv` instead of POWER8's `lxvd2x`/
`stxvd2x`. `Level::Vsx::is_supported()` checks `is_powerpc_feature_detected!("power9")`
with `std`, or `cfg!(target_feature = "power9-vector")` without it, so POWER8
falls back to scalar.

The design mirrors the [NEON kernel](aarch64.md): 64-byte blocks that start on
a character boundary, Keiser–Lemire lookup validation, one 64-bit lead mask,
byte-domain decoding and table-driven compaction. stdarch exposes no
`movemask` intrinsic for PowerPC, so the 16-bit masks come from a SWAR
multiply on the compare bytes.

## Overview

The loop runs while `i + 80 <= len`: 64 bytes for the block and 16 for the
decoders' shifted loads. Anything shorter goes to the scalar tail.

1. An all-ASCII block is widened with four `vec_perm` tables, four bytes to four
   u32 lanes each.
2. Otherwise each 16-byte vector is validated against its predecessor with
   three `vec_perm` table lookups, plus two more lookups and two saturating
   subtracts when the block has a byte `>= 0xE0`). The lead mask is built by
   comparing the block as signed bytes: `0x80..=0xBF` is `-128..=-65`, so one
   `vec_cmpgt` against `-64` marks continuations.
3. A character cut by the end of the block shortens the advance to 63, 62 or
   61 bytes, decided from the last three bytes. The next block starts at its
   lead.
4. Quarters without a byte `>= 0xF0` decode in the byte domain: high and low
   bytes of every candidate code point, then two `vec_perm` interleaves and a
   mask-indexed compaction straight into u32 lanes. Quarters with 4-byte leads
   add a third byte and compact four positions at a time.
5. A failing block decodes the valid prefix, then `scalar::run`/`step` resolve
   one character and cover a growing stretch (32 bytes, doubled per consecutive
   failing block, capped at 1 KiB) before SIMD retries.

## Masks without movemask

`vec_cmpgt` leaves 0x00 or 0xFF in each byte. AND with 0x80 leaves byte `k` as
0 or 0x80, i.e. bit `7 + 8k` of the 64-bit word. Reading the two halves
little-endian and multiplying each by `0x0002040810204081` moves that bit into
bit `k` of the top byte. Two multiplies produce one 16-bit mask per vector;
four vectors give the 64-bit lead mask.

## Measured behaviour

Measured on `sol`, a POWER9 ppc64le machine, with Rust 1.102 nightly, LTO
release and the `vsx` feature. Each figure is GB/s of UTF-8 input, the median
of 7 runs of about 1 MB. The full table, including the other implementations,
is in `bench/results/ppc64.md`; it was produced with
`cargo +nightly run --release -p utf32-bench --features utf8-to-utf32/vsx`.

| GB/s | ascii | latin | cjk | box | emoji |
|---|---|---|---|---|---|
| scalar | 2.01 | 0.58 | 0.54 | 0.47 | 0.42 |
| VSX | 5.56 | 0.78 | 0.78 | 0.71 | 0.59 |

VSX wins on every input class here, from 1.34x on Latin to 2.8x on ASCII. Three
changes got it there. The first version of `decode_wide` interleaved and
compacted in two separate `vec_perm` stages (10 permutes per quarter) and ran
emoji at 0.24 GB/s, below scalar; folding both into one mask-indexed table
(6 permutes) raised it to 0.48. Enabling the POWER9 vector instructions then
raised ASCII from 3.33 to 5.56 and emoji to 0.59. The four validation results
are also OR-ed before a single `vec_any_ne`, which removes three compare-records
per block.

On P9, `vperm` and `vbpermq` are 3-cycle PM operations with only two PM units
per chip. Using `vbpermq` for the movemask would add two internal nightly
features (`link_llvm_intrinsics`, `simd_ffi`), so the SWAR multiply stays.

Three lookups per 16 bytes is the floor for a 128-bit shuffle: one permute has
a single index per lane, so it can read either 16-byte table but not combine
two independent nibble lookups in the same lane, and a 256-byte table needs at
least eight permutes. A compare-based replacement for one table was correct but
slower (latin 0.80 to 0.64 GB/s, `utf32-bench`, median of 7, `sol`).

![GB/s by implementation and input type on POWER9](../performance-ppc64.svg)

## Why the Decoder regresses on error-dense input

On the `bench/` dirty input the VSX `Decoder` runs at 0.10 GB/s against scalar's
0.18 (same machine and method as the table above). An instrumented run over
that input counted 60,000 kernel entries for 30,000 errors: the `Decoder`
starts a fresh 16 KiB window after every error and the kernel stops at the
first bad byte, so each entry validates only the first 64-byte block.

For that block the kernel does four unaligned loads, up to four validation
calls (three `vec_perm` lookups each, plus two shifts and two saturating
subtracts when a byte is `>= 0xE0`), builds the lead mask with the SWAR
multiply, decodes the valid prefix, and only then calls `scalar::run` to
resolve one character. The scalar kernel goes straight to `decode_one`.

This is the same tradeoff NEON makes on dirty input (0.21 against scalar's
0.33 on its machine), not a VSX-specific bug.

The backoff in the error path does not help here. It runs only after
`scalar::step` returns, and `step` returns the error immediately in `STRICT`
mode, which is what the `Decoder` uses. It pays off in the lossy modes, where
it lets SIMD skip a stretch after an isolated error.

Two changes were considered and not made. Checking the first byte with
`scalar::decode_one` before SIMD validation would skip this setup when the
window starts on an error, but adds a scalar check to every clean block and
needs a measurement on clean text first.

Letting the `Decoder` consume lossy kernel output, or adding a kernel mode
that returns all pieces of a window in one call, would remove the restarts,
but both change the kernel contract and need an explicit design decision.

The ASCII `Decoder`, by contrast, is faster than scalar (13.46 against 10.45)
after moving the whole-vector ASCII test ahead of the SWAR movemask in
`ascii_len_vsx` and enabling the POWER9 instructions.

## Correctness

`tests/conformance.rs` runs every supported level, so VSX is checked against
`core::str::from_utf8` and `utf8_chunks` on random and structured inputs with
block-boundary sweeps. `tests/exhaustive_neon.rs` also runs every level, but its
64-byte buffers never enter the SIMD loop, so on powerpc64 it only exercises the
scalar path. Run `cargo +nightly test --release --features vsx` on a POWER9
machine after a change to the validator, the tables or a decode path.
