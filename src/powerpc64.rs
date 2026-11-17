//! powerpc64 VSX SIMD kernel (nightly, `stdarch_powerpc`).
//!
//! Mirrors the aarch64 kernel: 64-byte blocks starting on a character
//! boundary, Keiser-Lemire validation with `vec_perm` lookups, a 64-bit lead
//! mask, byte-domain decoding and `vec_perm` compaction. The scalar module
//! owns the error semantics; this kernel only decides that a block is bad.

#![cfg_attr(not(feature = "alloc"), allow(dead_code))]

use crate::scalar;
use crate::Utf32Error;
use core::arch::powerpc64::*;
use core::mem::transmute;

type V8 = vector_unsigned_char;

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

// vec_perm selects from (a[0..16], b[0..16]) in lane order.
const P1: [u8; 16] = [
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
];
const P2: [u8; 16] = [
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
];
const P3: [u8; 16] = [
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
];
const ZIP1: [u8; 16] = [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23];
const ZIP2: [u8; 16] = [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31];

const W0: [u8; 16] = [0, 16, 16, 16, 1, 16, 16, 16, 2, 16, 16, 16, 3, 16, 16, 16];
const W1: [u8; 16] = [4, 16, 16, 16, 5, 16, 16, 16, 6, 16, 16, 16, 7, 16, 16, 16];
const W2: [u8; 16] = [8, 16, 16, 16, 9, 16, 16, 16, 10, 16, 16, 16, 11, 16, 16, 16];
const W3: [u8; 16] = [
    12, 16, 16, 16, 13, 16, 16, 16, 14, 16, 16, 16, 15, 16, 16, 16,
];

/// For an 8-position group mask: two `vec_perm` controls picking the (lo, hi)
/// byte pair of the k-th selected position into a u32 lane. Unused bytes point
/// into the zero vector (indices 16..=31).
const fn tbl_pairs() -> [[[u8; 16]; 2]; 256] {
    let mut t = [[[16u8; 16]; 2]; 256];
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

/// Four groups × 16 masks: compact four u32 lanes straight out of the
/// interleaved (lo, mid) and (top, zero) vectors. Index 17 is a zero byte.
const fn tbl_wide() -> [[[u8; 16]; 16]; 4] {
    let mut t = [[[17u8; 16]; 16]; 4];
    let mut g = 0;
    while g < 4 {
        let mut m = 0;
        while m < 16 {
            let (mut k, mut lane) = (0, 0);
            while lane < 4 {
                if m & (1 << lane) != 0 {
                    let base = 8 * (g % 2) + 2 * lane;
                    t[g][m][k * 4] = base as u8;
                    t[g][m][k * 4 + 1] = (base + 1) as u8;
                    t[g][m][k * 4 + 2] = (16 + base) as u8;
                    t[g][m][k * 4 + 3] = (17 + base) as u8;
                    k += 1;
                }
                lane += 1;
            }
            m += 1;
        }
        g += 1;
    }
    t
}
static TBL_WIDE: [[[u8; 16]; 16]; 4] = tbl_wide();

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

// Inherited from the NEON kernel and kept on error-dense box drawing, not tuned on POWER9.
const ERR_STRETCH: usize = 32;
const ERR_BACKOFF_MAX: u32 = 5;

/// Whether every byte of `v` is below `n` (unsigned horizontal maximum).
#[inline(always)]
unsafe fn all_below(v: V8, n: u8) -> bool {
    vec_all_lt(v, vec_splats(n))
}

#[inline(always)]
unsafe fn movemask(v: V8, c80: V8) -> u16 {
    let b: [u8; 16] = transmute(vec_and(v, c80));
    let lo = u64::from_le_bytes(b[..8].try_into().unwrap());
    let hi = u64::from_le_bytes(b[8..].try_into().unwrap());
    ((lo.wrapping_mul(0x0002_0408_1020_4081) >> 56) as u16)
        | (((hi.wrapping_mul(0x0002_0408_1020_4081) >> 56) as u16) << 8)
}

#[inline(always)]
unsafe fn movemask64(v: [V8; 4], c80: V8) -> u64 {
    movemask(v[0], c80) as u64
        | (movemask(v[1], c80) as u64) << 16
        | (movemask(v[2], c80) as u64) << 32
        | (movemask(v[3], c80) as u64) << 48
}

struct Tables {
    b1h: V8,
    b1l: V8,
    b2h: V8,
}

#[inline(always)]
unsafe fn validate<const LONG: bool>(prev: V8, a: V8, t: &Tables) -> V8 {
    let p1 = vec_perm(prev, a, transmute(P1));
    let sc = vec_and(
        vec_and(
            vec_perm(t.b1h, t.b1h, vec_sr(p1, vec_splats(4u8))),
            vec_perm(t.b1l, t.b1l, vec_and(p1, vec_splats(0x0Fu8))),
        ),
        vec_perm(t.b2h, t.b2h, vec_sr(a, vec_splats(4u8))),
    );
    if !LONG {
        return sc;
    }
    let p2 = vec_perm(prev, a, transmute(P2));
    let p3 = vec_perm(prev, a, transmute(P3));
    // Bit 7 is set where a continuation byte is required 2/3 bytes back.
    let must = vec_and(
        vec_or(
            vec_subs(p2, vec_splats(0x60u8)),
            vec_subs(p3, vec_splats(0x70u8)),
        ),
        vec_splats(0x80u8),
    );
    vec_xor(must, sc)
}

#[inline(always)]
unsafe fn store_ascii16(dst: *mut u32, a: V8) {
    let z: V8 = transmute([0u8; 16]);
    let ctrls = [W0, W1, W2, W3];
    let mut k = 0;
    while k < 4 {
        let q = vec_perm(a, z, transmute(ctrls[k]));
        vec_xst(q, 0, dst.add(k * 4) as *mut u8);
        k += 1;
    }
}

#[inline(always)]
unsafe fn decode_bmp(p: *const u8, a: V8, lm: u32, dst: *mut u32, mut o: usize) -> usize {
    let v1: V8 = vec_xl(1, p);
    let v2: V8 = vec_xl(2, p);
    let z: V8 = transmute([0u8; 16]);
    let s6 = vec_splats(6u8);
    let c3f = vec_splats(0x3Fu8);
    let lo2 = vec_or(vec_sl(a, s6), vec_and(v1, c3f));
    let hi2 = vec_and(vec_sr(a, vec_splats(2u8)), vec_splats(0x07u8));
    let lo3 = vec_or(vec_sl(v1, s6), vec_and(v2, c3f));
    let hi3 = vec_or(
        vec_sl(a, vec_splats(4u8)),
        vec_sr(vec_and(v1, vec_splats(0x3Cu8)), vec_splats(2u8)),
    );
    let m2 = vec_cmpgt(a, vec_splats(0xBFu8));
    let m3 = vec_cmpgt(a, vec_splats(0xDFu8));
    let lo = vec_sel(vec_sel(a, lo2, m2), lo3, m3);
    let hi = vec_sel(vec_sel(z, hi2, m2), hi3, m3);
    let c = [
        vec_perm(lo, hi, transmute(ZIP1)),
        vec_perm(lo, hi, transmute(ZIP2)),
    ];
    let masks = [(lm & 0xFF) as usize, ((lm >> 8) & 0xFF) as usize];
    let d = dst as *mut u8;
    let mut g = 0;
    while g < 2 {
        let m = masks[g];
        if m != 0 {
            let t = &TBL_PAIRS[m];
            vec_xst(vec_perm(c[g], z, transmute(t[0])), 0, d.add(o * 4));
            let n = CNT8[m] as usize;
            if n > 4 {
                vec_xst(vec_perm(c[g], z, transmute(t[1])), 0, d.add(o * 4 + 16));
            }
            o += n;
        }
        g += 1;
    }
    o
}

#[inline(always)]
unsafe fn decode_wide(p: *const u8, a: V8, lm: u32, dst: *mut u32, mut o: usize) -> usize {
    let v1: V8 = vec_xl(1, p);
    let v2: V8 = vec_xl(2, p);
    let v3: V8 = vec_xl(3, p);
    let z: V8 = transmute([0u8; 16]);
    let s6 = vec_splats(6u8);
    let c3f = vec_splats(0x3Fu8);
    let lo2 = vec_or(vec_sl(a, s6), vec_and(v1, c3f));
    let hi2 = vec_and(vec_sr(a, vec_splats(2u8)), vec_splats(0x07u8));
    let lo3 = vec_or(vec_sl(v1, s6), vec_and(v2, c3f));
    let hi3 = vec_or(
        vec_sl(a, vec_splats(4u8)),
        vec_sr(vec_and(v1, vec_splats(0x3Cu8)), vec_splats(2u8)),
    );
    let lo4 = vec_or(vec_sl(v2, s6), vec_and(v3, c3f));
    let mid4 = vec_or(
        vec_sl(v1, vec_splats(4u8)),
        vec_sr(vec_and(v2, vec_splats(0x3Cu8)), vec_splats(2u8)),
    );
    let top4 = vec_or(
        vec_sl(vec_and(a, vec_splats(0x07u8)), vec_splats(2u8)),
        vec_sr(vec_and(v1, vec_splats(0x30u8)), vec_splats(4u8)),
    );
    let m2 = vec_cmpgt(a, vec_splats(0xBFu8));
    let m3 = vec_cmpgt(a, vec_splats(0xDFu8));
    let m4 = vec_cmpgt(a, vec_splats(0xEFu8));
    let lo = vec_sel(vec_sel(vec_sel(a, lo2, m2), lo3, m3), lo4, m4);
    let mid = vec_sel(vec_sel(vec_sel(z, hi2, m2), hi3, m3), mid4, m4);
    let top = vec_sel(z, top4, m4);
    let lm_lo = vec_perm(lo, mid, transmute(ZIP1));
    let lm_hi = vec_perm(lo, mid, transmute(ZIP2));
    let t_lo = vec_perm(top, z, transmute(ZIP1));
    let t_hi = vec_perm(top, z, transmute(ZIP2));
    let pairs = [(lm_lo, t_lo), (lm_hi, t_hi)];
    let mut g = 0;
    while g < 4 {
        let gm = ((lm >> (4 * g)) & 0xF) as usize;
        if gm != 0 {
            let (lo_mid, top_zero) = pairs[g / 2];
            let packed = vec_perm(lo_mid, top_zero, transmute(TBL_WIDE[g][gm]));
            vec_xst(packed, 0, dst.add(o) as *mut u8);
            o += CNT8[gm] as usize;
        }
        g += 1;
    }
    o
}

#[inline(always)]
unsafe fn decode_block(
    p: *const u8,
    a: [V8; 4],
    lm: u64,
    wide: bool,
    dst: *mut u32,
    mut o: usize,
) -> usize {
    let mut q = 0;
    while q < 4 {
        let m = ((lm >> (16 * q)) & 0xFFFF) as u32;
        if m != 0 {
            if wide && !all_below(a[q], 0xF0) {
                o = decode_wide(p.add(16 * q), a[q], m, dst, o);
            } else {
                o = decode_bmp(p.add(16 * q), a[q], m, dst, o);
            }
        }
        q += 1;
    }
    o
}

#[target_feature(enable = "vsx,power9-vector")]
unsafe fn vsx_impl(src: &[u8], dst: *mut u32, mode: u8) -> Result<usize, Utf32Error> {
    let len = src.len();
    let p = src.as_ptr();
    let (mut i, mut o) = (0usize, 0usize);
    let tables = Tables {
        b1h: vec_xl(0, B1_HIGH.as_ptr()),
        b1l: vec_xl(0, B1_LOW.as_ptr()),
        b2h: vec_xl(0, B2_HIGH.as_ptr()),
    };
    let z: V8 = transmute([0u8; 16]);
    let c80 = vec_splats(0x80u8);
    let mut streak = 0u32;
    while i + 80 <= len {
        let pi = p.add(i);
        let a = [
            vec_xl(0, pi),
            vec_xl(16, pi),
            vec_xl(32, pi),
            vec_xl(48, pi),
        ];
        let ascii = all_below(a[0], 0x80)
            && all_below(a[1], 0x80)
            && all_below(a[2], 0x80)
            && all_below(a[3], 0x80);
        if ascii {
            store_ascii16(dst.add(o), a[0]);
            store_ascii16(dst.add(o + 16), a[1]);
            store_ascii16(dst.add(o + 32), a[2]);
            store_ascii16(dst.add(o + 48), a[3]);
            i += 64;
            o += 64;
            continue;
        }
        let wide = !(all_below(a[0], 0xF0)
            && all_below(a[1], 0xF0)
            && all_below(a[2], 0xF0)
            && all_below(a[3], 0xF0));

        let long = !(all_below(a[0], 0xE0)
            && all_below(a[1], 0xE0)
            && all_below(a[2], 0xE0)
            && all_below(a[3], 0xE0));
        let e = if long {
            [
                validate::<true>(z, a[0], &tables),
                validate::<true>(a[0], a[1], &tables),
                validate::<true>(a[1], a[2], &tables),
                validate::<true>(a[2], a[3], &tables),
            ]
        } else {
            [
                validate::<false>(z, a[0], &tables),
                validate::<false>(a[0], a[1], &tables),
                validate::<false>(a[1], a[2], &tables),
                validate::<false>(a[2], a[3], &tables),
            ]
        };
        let all_e = vec_or(vec_or(e[0], e[1]), vec_or(e[2], e[3]));
        let bad = vec_any_ne(all_e, z);
        // A byte is a continuation iff it is 0x80..=0xBF, i.e. -128..=-65
        // as i8. Every i8 is >= -128, so one signed compare suffices.
        let c_neg64 = vec_splats(-64i8);
        let cont_signed = |v: V8| -> V8 {
            let s: vector_signed_char = transmute(v);
            transmute(vec_cmpgt(c_neg64, s))
        };
        let cont: [V8; 4] = [
            cont_signed(a[0]),
            cont_signed(a[1]),
            cont_signed(a[2]),
            cont_signed(a[3]),
        ];
        let leads = !movemask64(cont, c80);

        if bad {
            let badm = movemask64(
                [
                    transmute::<vector_bool_char, V8>(vec_cmpgt(e[0], z)),
                    transmute::<vector_bool_char, V8>(vec_cmpgt(e[1], z)),
                    transmute::<vector_bool_char, V8>(vec_cmpgt(e[2], z)),
                    transmute::<vector_bool_char, V8>(vec_cmpgt(e[3], z)),
                ],
                c80,
            );
            let q = badm.trailing_zeros();
            let below = leads & ((1u64 << q) - 1);
            let cut = if below == 0 {
                0
            } else {
                63 - below.leading_zeros() as usize
            };
            let lm = leads & ((1u64 << cut) - 1);
            o = decode_block(pi, a, lm, wide, dst, o);
            i += cut;
            let stop = (i + (ERR_STRETCH << streak)).min(len);
            streak = (streak + 1).min(ERR_BACKOFF_MAX);
            (i, o) = scalar::step(scalar::run(src, i, i + 1, dst, o), src, dst, mode)?;
            while i < stop {
                (i, o) = scalar::step(scalar::run(src, i, stop, dst, o), src, dst, mode)?;
            }
            continue;
        }

        streak = 0;
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
        o = decode_block(pi, a, lm, wide, dst, o);
        i += adv;
    }
    scalar::finish(src, i, dst, o, mode)
}

pub(crate) unsafe fn convert_vsx(src: &[u8], dst: *mut u32, mode: u8) -> Result<usize, Utf32Error> {
    vsx_impl(src, dst, mode)
}

#[cfg(feature = "alloc")]
#[target_feature(enable = "vsx,power9-vector")]
unsafe fn ascii_len_vsx_impl(src: &[u8]) -> usize {
    let mut i = 0;
    let c80 = vec_splats(0x80u8);
    let c7f = vec_splats(0x7Fu8);
    while i + 16 <= src.len() {
        let v: V8 = vec_xl(0, src.as_ptr().add(i));
        if !all_below(v, 0x80) {
            let gt: V8 = transmute(vec_cmpgt(v, c7f));
            return i + movemask(gt, c80).trailing_zeros() as usize;
        }
        i += 16;
    }
    i + scalar::ascii_len(&src[i..])
}

#[cfg(feature = "alloc")]
pub(crate) unsafe fn ascii_len_vsx(src: &[u8]) -> usize {
    ascii_len_vsx_impl(src)
}
