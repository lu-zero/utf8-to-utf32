//! aarch64 NEON SIMD kernel.
//!
//! The input is processed in 64-byte blocks that always start on a character
//! boundary:
//!
//! 1. An all-ASCII block is widened directly.
//! 2. Otherwise the block is validated with the Keiser–Lemire lookup scheme (three
//!    `tbl` lookups on the nibbles of adjacent bytes) and one 64-bit lead-byte
//!    mask is built. If a character is cut by the end of the block, the block is
//!    shortened (decided from its last three bytes only) so the next one starts there.
//! 3. Blocks without 4-byte leads are decoded in the byte domain, 16 positions
//!    at a time (high and low bytes of every candidate 16-bit code point, using
//!    shift-and-insert), then compacted with `tbl` straight into u32 lanes.
//!    Quarters with 4-byte leads add a third output byte and compact 4 positions
//!    at a time.
//! 4. A block that fails validation still has the characters before the first
//!    bad lane decoded with SIMD; the scalar path then resolves exactly one
//!    character (reporting or replacing the error) and SIMD restarts after it.

#![cfg_attr(not(feature = "alloc"), allow(dead_code))]

use crate::scalar;
use crate::Utf32Error;
use core::arch::aarch64::*;

// ---- Keiser–Lemire validation tables -------------------------------------

const TOO_SHORT: u8 = 1 << 0;
const TOO_LONG: u8 = 1 << 1;
const OVERLONG_3: u8 = 1 << 2;
const TOO_LARGE: u8 = 1 << 3;
const SURROGATE: u8 = 1 << 4;
const OVERLONG_2: u8 = 1 << 5;
const TOO_LARGE_1000: u8 = 1 << 6;
const OVERLONG_4: u8 = 1 << 6;
const TWO_CONTS: u8 = 1 << 7;
const CARRY: u8 = TOO_SHORT | TOO_LONG | TWO_CONTS;

/// Indexed by the high nibble of the previous byte.
static B1_HIGH: [u8; 16] = [
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TWO_CONTS,
    TWO_CONTS,
    TWO_CONTS,
    TWO_CONTS,
    TOO_SHORT | OVERLONG_2,
    TOO_SHORT,
    TOO_SHORT | OVERLONG_3 | SURROGATE,
    TOO_SHORT | TOO_LARGE | TOO_LARGE_1000 | OVERLONG_4,
];

/// Indexed by the low nibble of the previous byte.
static B1_LOW: [u8; 16] = [
    CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4,
    CARRY | OVERLONG_2,
    CARRY,
    CARRY,
    CARRY | TOO_LARGE,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
];

/// Indexed by the high nibble of the current byte.
static B2_HIGH: [u8; 16] = [
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE_1000 | OVERLONG_4,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
];

// ---- Compaction tables ---------------------------------------------------

/// For an 8-position group mask `m`: two `tbl` index vectors that pick the
/// (lo, hi) byte pair of the k-th selected position out of a `zip(lo, hi)`
/// vector, into the low two bytes of u32 lane k % 4 (upper bytes zeroed).
/// `[m][0]` yields selected positions 0..4, `[m][1]` positions 4..8.
const fn tbl_pairs() -> [[[u8; 16]; 2]; 256] {
    let mut t = [[[0xFFu8; 16]; 2]; 256];
    let mut m = 0;
    while m < 256 {
        let mut k = 0;
        let mut pos = 0;
        while pos < 8 {
            if m & (1 << pos) != 0 {
                let (v, lane) = (k / 4, k % 4);
                t[m][v][lane * 4] = (pos * 2) as u8;
                t[m][v][lane * 4 + 1] = (pos * 2 + 1) as u8;
                k += 1;
            }
            pos += 1;
        }
        m += 1;
    }
    t
}
static TBL_PAIRS: [[[u8; 16]; 2]; 256] = tbl_pairs();

/// For a 4-position group mask `m`: `tbl` indices compacting the selected u32 lanes.
const fn tbl_u32() -> [[u8; 16]; 16] {
    let mut t = [[0xFFu8; 16]; 16];
    let mut m = 0;
    while m < 16 {
        let mut k = 0;
        let mut lane = 0;
        while lane < 4 {
            if m & (1 << lane) != 0 {
                let mut b = 0;
                while b < 4 {
                    t[m][k * 4 + b] = (lane * 4 + b) as u8;
                    b += 1;
                }
                k += 1;
            }
            lane += 1;
        }
        m += 1;
    }
    t
}
/// Population count of every byte value (scalar `count_ones` bounces through the vector unit on aarch64).
const fn cnt8() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = (i as u32).count_ones() as u8;
        i += 1;
    }
    t
}
static CNT8: [u8; 256] = cnt8();

static TBL_U32: [[u8; 16]; 16] = tbl_u32();

/// Bytes handled by the scalar path after the first invalid block. Each further
/// invalid block in a row doubles it, up to `ERR_STRETCH << ERR_BACKOFF_MAX`; a
/// valid block resets it. Dense errors therefore converge to scalar speed.
const ERR_STRETCH: usize = 32;
const ERR_BACKOFF_MAX: u32 = 5;

const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

// ---- Helpers -------------------------------------------------------------

/// Bit j set iff byte j of the compare result `v` (0x00 / 0xFF lanes) is set.
#[inline(always)]
unsafe fn movemask(v: uint8x16_t) -> u32 {
    let t = vandq_u8(v, vld1q_u8(BITS.as_ptr()));
    let s = vpaddq_u8(t, t);
    let s = vpaddq_u8(s, s);
    let s = vpaddq_u8(s, s);
    vgetq_lane_u16::<0>(vreinterpretq_u16_u8(s)) as u32
}

/// 64-bit version of [`movemask`] for four compare results.
#[inline(always)]
unsafe fn movemask64(v: [uint8x16_t; 4]) -> u64 {
    let w = vld1q_u8(BITS.as_ptr());
    let pa = vpaddq_u8(vandq_u8(v[0], w), vandq_u8(v[1], w));
    let pb = vpaddq_u8(vandq_u8(v[2], w), vandq_u8(v[3], w));
    let pc = vpaddq_u8(pa, pb);
    let pd = vpaddq_u8(pc, pc);
    vgetq_lane_u64::<0>(vreinterpretq_u64_u8(pd))
}

#[inline(always)]
unsafe fn store_ascii16(dst: *mut u32, v: uint8x16_t) {
    let lo = vmovl_u8(vget_low_u8(v));
    let hi = vmovl_high_u8(v);
    vst1q_u32(dst, vmovl_u16(vget_low_u16(lo)));
    vst1q_u32(dst.add(4), vmovl_high_u16(lo));
    vst1q_u32(dst.add(8), vmovl_u16(vget_low_u16(hi)));
    vst1q_u32(dst.add(12), vmovl_high_u16(hi));
}

struct Tables {
    b1h: uint8x16_t,
    b1l: uint8x16_t,
    b2h: uint8x16_t,
}

/// Non-zero lanes mark invalid sequences in `a`, whose preceding 16 bytes are
/// `prev` (zero at a character boundary). Truncated sequences at the end of `a`
/// are not reported. `LONG` enables the check for continuation bytes required by
/// 3/4-byte leads two or three bytes back; it can be off when no byte in the
/// block is >= 0xE0.
#[inline(always)]
unsafe fn validate<const LONG: bool>(prev: uint8x16_t, a: uint8x16_t, t: &Tables) -> uint8x16_t {
    let p1 = vextq_u8::<15>(prev, a);
    let sc = vandq_u8(
        vandq_u8(
            vqtbl1q_u8(t.b1h, vshrq_n_u8::<4>(p1)),
            vqtbl1q_u8(t.b1l, vandq_u8(p1, vdupq_n_u8(0x0F))),
        ),
        vqtbl1q_u8(t.b2h, vshrq_n_u8::<4>(a)),
    );
    if !LONG {
        return sc;
    }
    let p2 = vextq_u8::<14>(prev, a);
    let p3 = vextq_u8::<13>(prev, a);
    // Bit 7 set where a continuation byte is required by a 3/4-byte lead 2/3 back.
    let must = vandq_u8(
        vorrq_u8(
            vqsubq_u8(p2, vdupq_n_u8(0x60)),
            vqsubq_u8(p3, vdupq_n_u8(0x70)),
        ),
        vdupq_n_u8(0x80),
    );
    veorq_u8(must, sc)
}

/// Decodes the characters of the window at `p` whose lead bytes are set in
/// `lm` (16 bits), all of which are 1..=3 bytes long. Returns the new output index.
#[inline(always)]
unsafe fn decode_bmp(p: *const u8, a: uint8x16_t, lm: u32, dst: *mut u32, mut o: usize) -> usize {
    let v1 = vld1q_u8(p.add(1));
    let v2 = vld1q_u8(p.add(2));
    let m2 = vcgeq_u8(a, vdupq_n_u8(0xC0));
    let m3 = vcgeq_u8(a, vdupq_n_u8(0xE0));
    // 2-byte: (a & 0x1F) << 6 | (v1 & 0x3F)
    let lo2 = vsliq_n_u8::<6>(v1, a);
    let hi2 = vandq_u8(vshrq_n_u8::<2>(a), vdupq_n_u8(0x07));
    // 3-byte: (a & 0x0F) << 12 | (v1 & 0x3F) << 6 | (v2 & 0x3F)
    let lo3 = vsliq_n_u8::<6>(v2, v1);
    let hi3 = vsliq_n_u8::<4>(vshrq_n_u8::<2>(vandq_u8(v1, vdupq_n_u8(0x3C))), a);
    let lo = vbslq_u8(m3, lo3, vbslq_u8(m2, lo2, a));
    let hi = vbslq_u8(m3, hi3, vandq_u8(m2, hi2));
    let c0 = vzip1q_u8(lo, hi);
    let c1 = vzip2q_u8(lo, hi);
    let m0 = (lm & 0xFF) as usize;
    let m1 = (lm >> 8) as usize;
    let d = dst as *mut u8;
    let t = &TBL_PAIRS[m0];
    vst1q_u8(d.add(o * 4), vqtbl1q_u8(c0, vld1q_u8(t[0].as_ptr())));
    let n0 = CNT8[m0] as usize;
    if n0 > 4 {
        vst1q_u8(d.add(o * 4 + 16), vqtbl1q_u8(c0, vld1q_u8(t[1].as_ptr())));
    }
    o += n0;
    let t = &TBL_PAIRS[m1];
    vst1q_u8(d.add(o * 4), vqtbl1q_u8(c1, vld1q_u8(t[0].as_ptr())));
    let n1 = CNT8[m1] as usize;
    if n1 > 4 {
        vst1q_u8(d.add(o * 4 + 16), vqtbl1q_u8(c1, vld1q_u8(t[1].as_ptr())));
    }
    o + n1
}

/// Like [`decode_bmp`] but for windows that may contain 4-byte characters:
/// every position gets three output bytes, which are interleaved into u32 lanes
/// and compacted four positions at a time.
#[inline(always)]
unsafe fn decode_wide(p: *const u8, a: uint8x16_t, lm: u32, dst: *mut u32, mut o: usize) -> usize {
    let v1 = vld1q_u8(p.add(1));
    let v2 = vld1q_u8(p.add(2));
    let v3 = vld1q_u8(p.add(3));
    let m2 = vcgeq_u8(a, vdupq_n_u8(0xC0));
    let m3 = vcgeq_u8(a, vdupq_n_u8(0xE0));
    let m4 = vcgeq_u8(a, vdupq_n_u8(0xF0));
    let lo2 = vsliq_n_u8::<6>(v1, a);
    let hi2 = vandq_u8(vshrq_n_u8::<2>(a), vdupq_n_u8(0x07));
    let lo3 = vsliq_n_u8::<6>(v2, v1);
    let hi3 = vsliq_n_u8::<4>(vshrq_n_u8::<2>(vandq_u8(v1, vdupq_n_u8(0x3C))), a);
    // 4-byte: (a & 7) << 18 | (v1 & 0x3F) << 12 | (v2 & 0x3F) << 6 | (v3 & 0x3F)
    let lo4 = vsliq_n_u8::<6>(v3, v2);
    let mid4 = vsliq_n_u8::<4>(vshrq_n_u8::<2>(vandq_u8(v2, vdupq_n_u8(0x3C))), v1);
    let top4 = vsliq_n_u8::<2>(
        vshrq_n_u8::<4>(vandq_u8(v1, vdupq_n_u8(0x30))),
        vandq_u8(a, vdupq_n_u8(0x07)),
    );
    let lo = vbslq_u8(m4, lo4, vbslq_u8(m3, lo3, vbslq_u8(m2, lo2, a)));
    let mid = vbslq_u8(m4, mid4, vbslq_u8(m3, hi3, vandq_u8(m2, hi2)));
    let top = vandq_u8(m4, top4);
    let z = vdupq_n_u8(0);
    let lm_lo = vreinterpretq_u16_u8(vzip1q_u8(lo, mid));
    let lm_hi = vreinterpretq_u16_u8(vzip2q_u8(lo, mid));
    let t_lo = vreinterpretq_u16_u8(vzip1q_u8(top, z));
    let t_hi = vreinterpretq_u16_u8(vzip2q_u8(top, z));
    let x = [
        vzip1q_u16(lm_lo, t_lo),
        vzip2q_u16(lm_lo, t_lo),
        vzip1q_u16(lm_hi, t_hi),
        vzip2q_u16(lm_hi, t_hi),
    ];
    let mut g = 0;
    while g < 4 {
        let gm = ((lm >> (4 * g)) & 0xF) as usize;
        let packed = vqtbl1q_u8(vreinterpretq_u8_u16(x[g]), vld1q_u8(TBL_U32[gm].as_ptr()));
        vst1q_u8(dst.add(o) as *mut u8, packed);
        o += CNT8[gm] as usize;
        g += 1;
    }
    o
}

/// Decodes the characters of a 64-byte block whose lead bytes are set in `lm`.
/// `mx` is the maximum byte of the block.
#[inline(always)]
unsafe fn decode_block(
    p: *const u8,
    a: [uint8x16_t; 4],
    lm: u64,
    mx: u8,
    dst: *mut u32,
    mut o: usize,
) -> usize {
    let mut q = 0;
    while q < 4 {
        let m = ((lm >> (16 * q)) & 0xFFFF) as u32;
        if m == 0 {
            q += 1;
            continue;
        }
        if mx < 0xF0 || vmaxvq_u8(a[q]) < 0xF0 {
            o = decode_bmp(p.add(16 * q), a[q], m, dst, o);
        } else {
            o = decode_wide(p.add(16 * q), a[q], m, dst, o);
        }
        q += 1;
    }
    o
}

#[target_feature(enable = "neon")]
unsafe fn neon_impl(src: &[u8], dst: *mut u32, mode: u8) -> Result<usize, Utf32Error> {
    let len = src.len();
    let p = src.as_ptr();
    let (mut i, mut o) = (0usize, 0usize);
    let tables = Tables {
        b1h: vld1q_u8(B1_HIGH.as_ptr()),
        b1l: vld1q_u8(B1_LOW.as_ptr()),
        b2h: vld1q_u8(B2_HIGH.as_ptr()),
    };
    let mut streak = 0u32;
    let z = vdupq_n_u8(0);
    let c64 = vdupq_n_s8(-64);
    // 64 bytes plus the lookahead of the decoders.
    while i + 80 <= len {
        let pi = p.add(i);
        let a = [
            vld1q_u8(pi),
            vld1q_u8(pi.add(16)),
            vld1q_u8(pi.add(32)),
            vld1q_u8(pi.add(48)),
        ];
        let mx = vmaxvq_u8(vmaxq_u8(vmaxq_u8(a[0], a[1]), vmaxq_u8(a[2], a[3])));
        if mx < 0x80 {
            store_ascii16(dst.add(o), a[0]);
            store_ascii16(dst.add(o + 16), a[1]);
            store_ascii16(dst.add(o + 32), a[2]);
            store_ascii16(dst.add(o + 48), a[3]);
            i += 64;
            o += 64;
            continue;
        }

        let e = if mx < 0xE0 {
            [
                validate::<false>(z, a[0], &tables),
                validate::<false>(a[0], a[1], &tables),
                validate::<false>(a[1], a[2], &tables),
                validate::<false>(a[2], a[3], &tables),
            ]
        } else {
            [
                validate::<true>(z, a[0], &tables),
                validate::<true>(a[0], a[1], &tables),
                validate::<true>(a[1], a[2], &tables),
                validate::<true>(a[2], a[3], &tables),
            ]
        };
        let bad = vmaxvq_u8(vorrq_u8(vorrq_u8(e[0], e[1]), vorrq_u8(e[2], e[3]))) != 0;
        let cs = |v: uint8x16_t| vcltq_s8(vreinterpretq_s8_u8(v), c64);
        let leads = !movemask64([cs(a[0]), cs(a[1]), cs(a[2]), cs(a[3])]);

        if bad {
            let nz = |v: uint8x16_t| vcgtq_u8(v, z);
            let badm = movemask64([nz(e[0]), nz(e[1]), nz(e[2]), nz(e[3])]);
            let q = badm.trailing_zeros();
            // Characters starting before the last lead below the first bad lane are valid.
            let below = leads & ((1u64 << q) - 1);
            let cut = if below == 0 {
                0
            } else {
                63 - below.leading_zeros() as usize
            };
            let lm = leads & ((1u64 << cut) - 1);
            o = decode_block(pi, a, lm, mx, dst, o);
            i += cut;
            // Errors tend to cluster: let scalar cover a stretch before SIMD retries.
            let stop = (i + (ERR_STRETCH << streak)).min(len);
            streak = (streak + 1).min(ERR_BACKOFF_MAX);
            (i, o) = scalar::step(scalar::run(src, i, i + 1, dst, o), src, dst, mode)?;
            while i < stop {
                (i, o) = scalar::step(scalar::run(src, i, stop, dst, o), src, dst, mode)?;
            }
            continue;
        }

        streak = 0;
        // Does a character start in the last three bytes and run past the block?
        let adv = if *pi.add(63) >= 0xC0 {
            63
        } else if *pi.add(62) >= 0xE0 {
            62
        } else if *pi.add(61) >= 0xF0 {
            61
        } else {
            64
        };
        let lm = if adv == 64 {
            leads
        } else {
            leads & ((1u64 << adv) - 1)
        };
        o = decode_block(pi, a, lm, mx, dst, o);
        i += adv;
    }
    scalar::finish(src, i, dst, o, mode)
}

pub(crate) unsafe fn convert_neon(
    src: &[u8],
    dst: *mut u32,
    mode: u8,
) -> Result<usize, Utf32Error> {
    neon_impl(src, dst, mode)
}

#[cfg(feature = "alloc")]
#[target_feature(enable = "neon")]
unsafe fn ascii_len_neon_impl(src: &[u8]) -> usize {
    let mut i = 0;
    while i + 16 <= src.len() {
        let v = vld1q_u8(src.as_ptr().add(i));
        if vmaxvq_u8(v) >= 0x80 {
            let m = movemask(vcltzq_s8(vreinterpretq_s8_u8(v)));
            return i + m.trailing_zeros() as usize;
        }
        i += 16;
    }
    i + scalar::ascii_len(&src[i..])
}
#[cfg(feature = "alloc")]
pub(crate) unsafe fn ascii_len_neon(src: &[u8]) -> usize {
    ascii_len_neon_impl(src)
}
