//! x86 / x86_64 SIMD kernels. All share the same scheme:
//!
//! 1. ASCII fast path: if the next bytes have no high bit, widen them
//!    (`pmovzxbd`) straight to u32.
//! 2. Otherwise take a block of P positions (60 bytes at every level) plus 3 bytes of
//!    lookahead and build u64 bitmasks from byte compares (high bit, at least
//!    C0, C2, E0, F0, F5; for 3/4-byte leads E0/ED/F0/F4 also whether the next
//!    byte is at least A0 / 90). The block is valid iff the continuation mask
//!    equals the mask of positions claimed by leads, there are no bad leads,
//!    and no overlong, surrogate or above-U+10FFFF second bytes.
//! 3. Decode. If the block has no 4-byte leads, a u16-lane path is used
//!    (8/16/32 lanes per register; AVX-512 compacts with VBMI2 `vpcompressw`).
//!    Otherwise (AVX2, AVX-512): gather bytes j..j+3 into u32 lane j (pshufb / vpermb), compute
//!    the 1/2/3/4-byte code point per lane with shifts/masks/blends, then
//!    compact lanes that start a character (vpermd table / vpcompressd) and store.
//!    SSE4.1 keeps u16 lanes and carries bits 16..=20 in a second vector.
//! 4. A block failing the check is decoded by the scalar path, which reports
//!    the exact error. Tails shorter than the load width are scalar.

// ascii_len_* are only used by the alloc-gated Decoder.
#![cfg_attr(not(feature = "alloc"), allow(dead_code))]

use crate::scalar;
use crate::Utf32Error;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::mask::{plan, plan_err, ranges_bad, ERR_STREAK_MAX, PREFIX};

// ---------- tables ----------

/// pshufb masks compacting u16 lanes selected by an 8-bit mask.
const fn sse16_tbl() -> [[u8; 16]; 256] {
    let mut t = [[0x80u8; 16]; 256];
    let mut m = 0;
    while m < 256 {
        let mut k = 0;
        let mut lane = 0;
        while lane < 8 {
            if m & (1 << lane) != 0 {
                t[m][k * 2] = (lane * 2) as u8;
                t[m][k * 2 + 1] = (lane * 2 + 1) as u8;
                k += 1;
            }
            lane += 1;
        }
        m += 1;
    }
    t
}
static SSE16_TBL: [[u8; 16]; 256] = sse16_tbl();

const fn avx2_tbl() -> [[u8; 8]; 256] {
    let mut t = [[0u8; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let mut k = 0;
        let mut lane = 0;
        while lane < 8 {
            if m & (1 << lane) != 0 {
                t[m][k] = lane as u8;
                k += 1;
            }
            lane += 1;
        }
        m += 1;
    }
    t
}
static AVX2_TBL: [[u8; 8]; 256] = avx2_tbl();

const fn vbmi_idx(g: usize) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut j = 0;
    while j < 16 {
        let mut k = 0;
        while k < 4 {
            t[j * 4 + k] = ((g + j + k) & 63) as u8;
            k += 1;
        }
        j += 1;
    }
    t
}
static VBMI_IDX: [[u8; 64]; 4] = [vbmi_idx(0), vbmi_idx(16), vbmi_idx(32), vbmi_idx(48)];

/// u16 lane j = bytes (g+j, g+j+1) for IDX16[.][0], and (g+j+2, _) for IDX16[.][1].
const fn vbmi_idx16(g: usize, off: usize) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut j = 0;
    while j < 32 {
        t[j * 2] = ((g + j + off) & 63) as u8;
        t[j * 2 + 1] = ((g + j + off + 1) & 63) as u8;
        j += 1;
    }
    t
}
static VBMI_IDX16: [[[u8; 64]; 2]; 2] = [
    [vbmi_idx16(0, 0), vbmi_idx16(0, 2)],
    [vbmi_idx16(32, 0), vbmi_idx16(32, 2)],
];

static PAT8: [u8; 32] = [
    0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, //
    4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10,
];

/// Signed-compare threshold so that `(b as i8) > t` <=> `b >= x` for high-bit bytes.
const fn th(x: u8) -> i8 {
    (x as i8) - 1
}

// ---------- SSE4.1 ----------

/// Positions per SSE4.1 block: four 16-byte vectors give 64-bit masks, less the 3-byte lookahead.
const SSE41_P: usize = 60;

/// Decodes the characters starting at `leads` (bit k = byte `i + k`) of a validated
/// block that holds 4-byte characters, 8 positions at a time: the low 16 bits of each
/// code point in u16 lanes, bits 16..=20 in a second vector, interleaved into u32 on
/// store. Out of line so the 1- to 3-byte loop keeps its register allocation.
///
/// Safety: as for the `sse41_impl` decode paths; `src[i..i + 72]` must be readable.
#[target_feature(enable = "sse2,ssse3,sse4.1,popcnt")]
#[inline(never)]
unsafe fn sse41_wide(p: *const u8, i: usize, dst: *mut u32, mut o: usize, leads: u64) -> usize {
    const P: usize = SSE41_P;
    let m3f = _mm_set1_epi16(0x3F);
    let mut g = 0;
    while g < P {
        let gm = ((leads >> g) & 0xFF) as usize;
        if gm != 0 {
            let ld =
                |k: usize| _mm_cvtepu8_epi16(_mm_loadl_epi64(p.add(i + g + k) as *const __m128i));
            let t0 = ld(0);
            let b1 = _mm_and_si128(ld(1), m3f);
            let b2 = _mm_and_si128(ld(2), m3f);
            let b3 = _mm_and_si128(ld(3), m3f);
            let cp2 = _mm_or_si128(
                _mm_slli_epi16::<6>(_mm_and_si128(t0, _mm_set1_epi16(0x1F))),
                b1,
            );
            let b12 = _mm_or_si128(_mm_slli_epi16::<6>(b1), b2);
            let cp3 = _mm_or_si128(_mm_slli_epi16::<12>(t0), b12);
            let cp4 = _mm_or_si128(_mm_slli_epi16::<6>(b12), b3);
            let is4 = _mm_cmpgt_epi16(t0, _mm_set1_epi16(0xEF));
            let mut r = _mm_blendv_epi8(t0, cp2, _mm_cmpgt_epi16(t0, _mm_set1_epi16(0xBF)));
            r = _mm_blendv_epi8(r, cp3, _mm_cmpgt_epi16(t0, _mm_set1_epi16(0xDF)));
            r = _mm_blendv_epi8(r, cp4, is4);
            let top = _mm_and_si128(
                _mm_or_si128(
                    _mm_slli_epi16::<2>(_mm_and_si128(t0, _mm_set1_epi16(0x07))),
                    _mm_srli_epi16::<4>(b1),
                ),
                is4,
            );
            let sh = _mm_loadu_si128(SSE16_TBL[gm].as_ptr() as *const __m128i);
            let r = _mm_shuffle_epi8(r, sh);
            let top = _mm_shuffle_epi8(top, sh);
            _mm_storeu_si128(dst.add(o) as *mut __m128i, _mm_unpacklo_epi16(r, top));
            _mm_storeu_si128(dst.add(o + 4) as *mut __m128i, _mm_unpackhi_epi16(r, top));
            o += gm.count_ones() as usize;
        }
        g += 8;
    }
    o
}

#[target_feature(enable = "sse2,ssse3,sse4.1,popcnt")]
unsafe fn sse41_impl(src: &[u8], dst: *mut u32, mode: u8) -> Result<usize, Utf32Error> {
    let len = src.len();
    let p = src.as_ptr();
    let (mut i, mut o) = (0usize, 0usize);
    let mut streak = 0u32;
    // Four vectors give 64-bit masks; the u32 path's last 16-byte load ends at i + P + 12.
    while i + 72 <= len {
        let a = _mm_loadu_si128(p.add(i) as *const __m128i);
        let b = _mm_loadu_si128(p.add(i + 16) as *const __m128i);
        let c = _mm_loadu_si128(p.add(i + 32) as *const __m128i);
        let d = _mm_loadu_si128(p.add(i + 48) as *const __m128i);
        macro_rules! mm {
            ($v:ident => $e:expr) => {{
                let m = |$v: __m128i| _mm_movemask_epi8($e) as u32 as u64;
                m(a) | (m(b) << 16) | (m(c) << 32) | (m(d) << 48)
            }};
        }
        let hi = mm!(x => x);
        let tz = (hi.trailing_zeros() & !3) as usize;
        if tz >= 4 {
            let mut k = 0;
            while k < tz {
                let x = _mm_cvtsi32_si128((p.add(i + k) as *const i32).read_unaligned());
                _mm_storeu_si128(dst.add(o + k) as *mut __m128i, _mm_cvtepu8_epi32(x));
                k += 4;
            }
            i += tz;
            o += tz;
            continue;
        }
        macro_rules! ge {
            ($x:expr) => {
                mm!(v => _mm_cmpgt_epi8(v, _mm_set1_epi8(th($x)))) & hi
            };
        }
        macro_rules! eq {
            ($x:expr) => {
                mm!(v => _mm_cmpeq_epi8(v, _mm_set1_epi8($x as u8 as i8)))
            };
        }
        const P: u32 = SSE41_P as u32;
        let (c0, e0, f0) = (ge!(0xC0), ge!(0xE0), ge!(0xF0));
        macro_rules! dec16 {
            ($leads:expr) => {{
                let leads: u64 = $leads;
                // <= 3-byte chars only: code points fit in u16, 8 lanes per register.
                let m3f = _mm_set1_epi16(0x3F);
                let mut g = 0;
                while g < P as usize {
                    let gm = ((leads >> g) & 0xFF) as usize;
                    if gm != 0 {
                        let t0 = _mm_cvtepu8_epi16(_mm_loadl_epi64(p.add(i + g) as *const __m128i));
                        let b1 = _mm_and_si128(
                            _mm_cvtepu8_epi16(_mm_loadl_epi64(p.add(i + g + 1) as *const __m128i)),
                            m3f,
                        );
                        let b2 = _mm_and_si128(
                            _mm_cvtepu8_epi16(_mm_loadl_epi64(p.add(i + g + 2) as *const __m128i)),
                            m3f,
                        );
                        let cp2 = _mm_or_si128(
                            _mm_slli_epi16::<6>(_mm_and_si128(t0, _mm_set1_epi16(0x1F))),
                            b1,
                        );
                        let cp3 = _mm_or_si128(
                            _mm_or_si128(_mm_slli_epi16::<12>(t0), _mm_slli_epi16::<6>(b1)),
                            b2,
                        );
                        let mut r =
                            _mm_blendv_epi8(t0, cp2, _mm_cmpgt_epi16(t0, _mm_set1_epi16(0xBF)));
                        r = _mm_blendv_epi8(r, cp3, _mm_cmpgt_epi16(t0, _mm_set1_epi16(0xDF)));
                        let r = _mm_shuffle_epi8(
                            r,
                            _mm_loadu_si128(SSE16_TBL[gm].as_ptr() as *const __m128i),
                        );
                        _mm_storeu_si128(dst.add(o) as *mut __m128i, _mm_cvtepu16_epi32(r));
                        _mm_storeu_si128(
                            dst.add(o + 4) as *mut __m128i,
                            _mm_cvtepu16_epi32(_mm_srli_si128::<8>(r)),
                        );
                        o += gm.count_ones() as usize;
                    }
                    g += 8;
                }
            }};
        }
        let pm = (1u64 << P) - 1;
        let planned = match plan(hi, c0, ge!(0xC2), e0, f0, ge!(0xF5), P) {
            Ok(x) if e0 & pm == 0 => Ok(x),
            Ok(x) => {
                let rb = ranges_bad(
                    eq!(0xE0),
                    eq!(0xED),
                    eq!(0xF0),
                    eq!(0xF4),
                    ge!(0xA0),
                    ge!(0x90),
                ) & pm;
                if rb == 0 {
                    Ok(x)
                } else {
                    plan_err(rb, hi & !c0, pm)
                }
            }
            Err(bad) if PREFIX => {
                let rb = if e0 & pm == 0 {
                    0
                } else {
                    ranges_bad(
                        eq!(0xE0),
                        eq!(0xED),
                        eq!(0xF0),
                        eq!(0xF4),
                        ge!(0xA0),
                        ge!(0x90),
                    )
                };
                plan_err(bad | (rb & pm), hi & !c0, pm)
            }
            Err(_) => Err((0, 0)),
        };
        match planned {
            Ok((leads, adv)) if f0 & ((1u64 << P) - 1) == 0 => {
                streak = 0;
                dec16!(leads);
                i += adv;
            }
            Ok((leads, adv)) => {
                streak = 0;
                o = sse41_wide(p, i, dst, o, leads);
                i += adv;
            }
            Err((leads, cut)) => {
                // With PREFIX, decode the valid prefix with SIMD first. Errors cluster, so stay
                // in scalar for a stretch that doubles with each consecutive failing block.
                let end = (i + ((P as usize) << streak)).min(len);
                if PREFIX && leads != 0 {
                    if f0 & ((1u64 << cut) - 1) == 0 {
                        dec16!(leads);
                    } else {
                        o = sse41_wide(p, i, dst, o, leads);
                    }
                }
                if PREFIX {
                    i += cut;
                }
                while i < end {
                    (i, o) = scalar::step(scalar::run(src, i, end, dst, o), src, dst, mode)?;
                }
                streak = (streak + 1).min(ERR_STREAK_MAX);
            }
        }
    }
    scalar::finish(src, i, dst, o, mode)
}

pub(crate) unsafe fn convert_sse41(
    src: &[u8],
    dst: *mut u32,
    mode: u8,
) -> Result<usize, Utf32Error> {
    sse41_impl(src, dst, mode)
}

#[target_feature(enable = "sse2")]
unsafe fn ascii_len_sse2_impl(src: &[u8]) -> usize {
    let mut i = 0;
    while i + 16 <= src.len() {
        let m = _mm_movemask_epi8(_mm_loadu_si128(src.as_ptr().add(i) as *const __m128i)) as u32;
        if m != 0 {
            return i + m.trailing_zeros() as usize;
        }
        i += 16;
    }
    i + scalar::ascii_len(&src[i..])
}
pub(crate) unsafe fn ascii_len_sse2(src: &[u8]) -> usize {
    ascii_len_sse2_impl(src)
}

// ---------- AVX2 ----------

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dec_avx2(v: __m256i) -> __m256i {
    let s1 = |x: i32| _mm256_set1_epi32(x);
    let t0 = _mm256_and_si256(v, s1(0xFF));
    let m3f = s1(0x3F);
    let b1 = _mm256_and_si256(_mm256_srli_epi32::<8>(v), m3f);
    let b2 = _mm256_and_si256(_mm256_srli_epi32::<16>(v), m3f);
    let b3 = _mm256_and_si256(_mm256_srli_epi32::<24>(v), m3f);
    let cp2 = _mm256_or_si256(_mm256_slli_epi32::<6>(_mm256_and_si256(t0, s1(0x1F))), b1);
    let b12 = _mm256_or_si256(_mm256_slli_epi32::<6>(b1), b2);
    let cp3 = _mm256_or_si256(_mm256_slli_epi32::<12>(_mm256_and_si256(t0, s1(0x0F))), b12);
    let cp4 = _mm256_or_si256(
        _mm256_or_si256(
            _mm256_slli_epi32::<18>(_mm256_and_si256(t0, s1(0x07))),
            _mm256_slli_epi32::<6>(b12),
        ),
        b3,
    );
    let mut r = _mm256_blendv_epi8(t0, cp2, _mm256_cmpgt_epi32(t0, s1(0xBF)));
    r = _mm256_blendv_epi8(r, cp3, _mm256_cmpgt_epi32(t0, s1(0xDF)));
    _mm256_blendv_epi8(r, cp4, _mm256_cmpgt_epi32(t0, s1(0xEF)))
}

#[target_feature(enable = "avx2,popcnt")]
unsafe fn avx2_impl(src: &[u8], dst: *mut u32, mode: u8) -> Result<usize, Utf32Error> {
    let len = src.len();
    let p = src.as_ptr();
    let (mut i, mut o) = (0usize, 0usize);
    let mut streak = 0u32;
    let pat = _mm256_loadu_si256(PAT8.as_ptr() as *const __m256i);
    // The u32 path's last 16-byte load ends at i + P + 12.
    while i + 72 <= len {
        let a = _mm256_loadu_si256(p.add(i) as *const __m256i);
        let b = _mm256_loadu_si256(p.add(i + 32) as *const __m256i);
        macro_rules! mm {
            ($x:expr, $y:expr) => {
                (_mm256_movemask_epi8($x) as u32 as u64)
                    | ((_mm256_movemask_epi8($y) as u32 as u64) << 32)
            };
        }
        let hi = mm!(a, b);
        let tz = (hi.trailing_zeros() & !7) as usize;
        if tz >= 8 {
            let mut k = 0;
            while k < tz {
                let x = _mm_loadl_epi64(p.add(i + k) as *const __m128i);
                _mm256_storeu_si256(dst.add(o + k) as *mut __m256i, _mm256_cvtepu8_epi32(x));
                k += 8;
            }
            i += tz;
            o += tz;
            continue;
        }
        macro_rules! ge {
            ($x:expr) => {
                mm!(
                    _mm256_cmpgt_epi8(a, _mm256_set1_epi8(th($x))),
                    _mm256_cmpgt_epi8(b, _mm256_set1_epi8(th($x)))
                ) & hi
            };
        }
        macro_rules! eq {
            ($x:expr) => {
                mm!(
                    _mm256_cmpeq_epi8(a, _mm256_set1_epi8($x as u8 as i8)),
                    _mm256_cmpeq_epi8(b, _mm256_set1_epi8($x as u8 as i8))
                )
            };
        }
        const P: u32 = 60;
        let (c0, e0, f0) = (ge!(0xC0), ge!(0xE0), ge!(0xF0));
        macro_rules! dec16 {
            ($leads:expr) => {{
                let leads: u64 = $leads;
                let m3f = _mm256_set1_epi16(0x3F);
                let mut g = 0;
                while g < P as usize {
                    let gm = ((leads >> g) & 0xFFFF) as usize;
                    if gm != 0 {
                        let ld = |k: usize| {
                            _mm256_cvtepu8_epi16(
                                _mm_loadu_si128(p.add(i + g + k) as *const __m128i),
                            )
                        };
                        let t0 = ld(0);
                        let b1 = _mm256_and_si256(ld(1), m3f);
                        let b2 = _mm256_and_si256(ld(2), m3f);
                        let cp2 = _mm256_or_si256(
                            _mm256_slli_epi16::<6>(_mm256_and_si256(t0, _mm256_set1_epi16(0x1F))),
                            b1,
                        );
                        let cp3 = _mm256_or_si256(
                            _mm256_or_si256(
                                _mm256_slli_epi16::<12>(t0),
                                _mm256_slli_epi16::<6>(b1),
                            ),
                            b2,
                        );
                        let mut r = _mm256_blendv_epi8(
                            t0,
                            cp2,
                            _mm256_cmpgt_epi16(t0, _mm256_set1_epi16(0xBF)),
                        );
                        r = _mm256_blendv_epi8(
                            r,
                            cp3,
                            _mm256_cmpgt_epi16(t0, _mm256_set1_epi16(0xDF)),
                        );
                        let (lo, hm) = (gm & 0xFF, gm >> 8);
                        let sh = _mm256_loadu2_m128i(
                            SSE16_TBL[hm].as_ptr() as *const __m128i,
                            SSE16_TBL[lo].as_ptr() as *const __m128i,
                        );
                        let r = _mm256_shuffle_epi8(r, sh);
                        let nlo = lo.count_ones() as usize;
                        _mm256_storeu_si256(
                            dst.add(o) as *mut __m256i,
                            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r)),
                        );
                        _mm256_storeu_si256(
                            dst.add(o + nlo) as *mut __m256i,
                            _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(r)),
                        );
                        o += nlo + hm.count_ones() as usize;
                    }
                    g += 16;
                }
            }};
        }
        macro_rules! dec32 {
            ($leads:expr) => {{
                let leads: u64 = $leads;
                let mut g = 0;
                while g < P as usize {
                    let gm = ((leads >> g) & 0xFF) as usize;
                    if gm != 0 {
                        let x = _mm_loadu_si128(p.add(i + g) as *const __m128i);
                        let r = dec_avx2(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(x), pat));
                        let idx = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                            AVX2_TBL[gm].as_ptr() as *const __m128i
                        ));
                        _mm256_storeu_si256(
                            dst.add(o) as *mut __m256i,
                            _mm256_permutevar8x32_epi32(r, idx),
                        );
                        o += gm.count_ones() as usize;
                    }
                    g += 8;
                }
            }};
        }
        let pm = (1u64 << P) - 1;
        let planned = match plan(hi, c0, ge!(0xC2), e0, f0, ge!(0xF5), P) {
            Ok(x) if e0 & pm == 0 => Ok(x),
            Ok(x) => {
                let rb = ranges_bad(
                    eq!(0xE0),
                    eq!(0xED),
                    eq!(0xF0),
                    eq!(0xF4),
                    ge!(0xA0),
                    ge!(0x90),
                ) & pm;
                if rb == 0 {
                    Ok(x)
                } else {
                    plan_err(rb, hi & !c0, pm)
                }
            }
            Err(bad) if PREFIX => {
                let rb = if e0 & pm == 0 {
                    0
                } else {
                    ranges_bad(
                        eq!(0xE0),
                        eq!(0xED),
                        eq!(0xF0),
                        eq!(0xF4),
                        ge!(0xA0),
                        ge!(0x90),
                    )
                };
                plan_err(bad | (rb & pm), hi & !c0, pm)
            }
            Err(_) => Err((0, 0)),
        };
        match planned {
            Ok((leads, adv)) if f0 & ((1u64 << P) - 1) == 0 => {
                streak = 0;
                dec16!(leads);
                i += adv;
            }
            Ok((leads, adv)) => {
                streak = 0;
                dec32!(leads);
                i += adv;
            }
            Err((leads, cut)) => {
                // With PREFIX, decode the valid prefix with SIMD first. Errors cluster, so stay
                // in scalar for a stretch that doubles with each consecutive failing block.
                let end = (i + ((P as usize) << streak)).min(len);
                if PREFIX && leads != 0 {
                    if f0 & ((1u64 << cut) - 1) == 0 {
                        dec16!(leads);
                    } else {
                        dec32!(leads);
                    }
                }
                if PREFIX {
                    i += cut;
                }
                while i < end {
                    (i, o) = scalar::step(scalar::run(src, i, end, dst, o), src, dst, mode)?;
                }
                streak = (streak + 1).min(ERR_STREAK_MAX);
            }
        }
    }
    scalar::finish(src, i, dst, o, mode)
}

pub(crate) unsafe fn convert_avx2(
    src: &[u8],
    dst: *mut u32,
    mode: u8,
) -> Result<usize, Utf32Error> {
    avx2_impl(src, dst, mode)
}

#[target_feature(enable = "avx2")]
unsafe fn ascii_len_avx2_impl(src: &[u8]) -> usize {
    let mut i = 0;
    while i + 32 <= src.len() {
        let m =
            _mm256_movemask_epi8(_mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i)) as u32;
        if m != 0 {
            return i + m.trailing_zeros() as usize;
        }
        i += 32;
    }
    i + scalar::ascii_len(&src[i..])
}
pub(crate) unsafe fn ascii_len_avx2(src: &[u8]) -> usize {
    ascii_len_avx2_impl(src)
}

// ---------- AVX-512 (F + BW + VBMI) ----------

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn dec_avx512(v: __m512i) -> __m512i {
    let s1 = |x: i32| _mm512_set1_epi32(x);
    let t0 = _mm512_and_si512(v, s1(0xFF));
    let m3f = s1(0x3F);
    let b1 = _mm512_and_si512(_mm512_srli_epi32::<8>(v), m3f);
    let b2 = _mm512_and_si512(_mm512_srli_epi32::<16>(v), m3f);
    let b3 = _mm512_and_si512(_mm512_srli_epi32::<24>(v), m3f);
    let cp2 = _mm512_or_si512(_mm512_slli_epi32::<6>(_mm512_and_si512(t0, s1(0x1F))), b1);
    let b12 = _mm512_or_si512(_mm512_slli_epi32::<6>(b1), b2);
    let cp3 = _mm512_or_si512(_mm512_slli_epi32::<12>(_mm512_and_si512(t0, s1(0x0F))), b12);
    let cp4 = _mm512_or_si512(
        _mm512_or_si512(
            _mm512_slli_epi32::<18>(_mm512_and_si512(t0, s1(0x07))),
            _mm512_slli_epi32::<6>(b12),
        ),
        b3,
    );
    let mut r = _mm512_mask_mov_epi32(t0, _mm512_cmpgt_epu32_mask(t0, s1(0xBF)), cp2);
    r = _mm512_mask_mov_epi32(r, _mm512_cmpgt_epu32_mask(t0, s1(0xDF)), cp3);
    _mm512_mask_mov_epi32(r, _mm512_cmpgt_epu32_mask(t0, s1(0xEF)), cp4)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2,popcnt")]
unsafe fn avx512_impl(src: &[u8], dst: *mut u32, mode: u8) -> Result<usize, Utf32Error> {
    let len = src.len();
    let p = src.as_ptr();
    let (mut i, mut o) = (0usize, 0usize);
    let mut streak = 0u32;
    let idx = [
        _mm512_loadu_si512(VBMI_IDX[0].as_ptr() as *const _),
        _mm512_loadu_si512(VBMI_IDX[1].as_ptr() as *const _),
        _mm512_loadu_si512(VBMI_IDX[2].as_ptr() as *const _),
        _mm512_loadu_si512(VBMI_IDX[3].as_ptr() as *const _),
    ];
    let ld = |a: &[u8; 64]| _mm512_loadu_si512(a.as_ptr() as *const _);
    let idx16 = [
        [ld(&VBMI_IDX16[0][0]), ld(&VBMI_IDX16[0][1])],
        [ld(&VBMI_IDX16[1][0]), ld(&VBMI_IDX16[1][1])],
    ];
    while i + 64 <= len {
        let z = _mm512_loadu_si512(p.add(i) as *const _);
        let hi = _mm512_movepi8_mask(z);
        let tz = (hi.trailing_zeros() & !15) as usize;
        if tz >= 4 {
            let mut k = 0;
            while k < tz {
                let x = _mm_loadu_si128(p.add(i + k) as *const __m128i);
                _mm512_storeu_si512(dst.add(o + k) as *mut _, _mm512_cvtepu8_epi32(x));
                k += 16;
            }
            i += tz;
            o += tz;
            continue;
        }
        macro_rules! ge {
            ($x:expr) => {
                _mm512_cmpge_epu8_mask(z, _mm512_set1_epi8($x as u8 as i8))
            };
        }
        macro_rules! eq {
            ($x:expr) => {
                _mm512_cmpeq_epi8_mask(z, _mm512_set1_epi8($x as u8 as i8))
            };
        }
        const P: u32 = 60;
        let (c0, e0, f0) = (ge!(0xC0), ge!(0xE0), ge!(0xF0));
        macro_rules! dec16 {
            ($leads:expr) => {{
                let leads: u64 = $leads;
                let m3f = _mm512_set1_epi16(0x3F);
                for gi in 0..2 {
                    let gm = if gi == 0 {
                        leads as u32
                    } else {
                        ((leads >> 32) & 0x0FFF_FFFF) as u32
                    };
                    if gm != 0 {
                        let v01 = _mm512_permutexvar_epi8(idx16[gi][0], z);
                        let v2 = _mm512_permutexvar_epi8(idx16[gi][1], z);
                        let t0 = _mm512_and_si512(v01, _mm512_set1_epi16(0xFF));
                        let b1 = _mm512_and_si512(_mm512_srli_epi16::<8>(v01), m3f);
                        let b2 = _mm512_and_si512(v2, m3f);
                        let cp2 = _mm512_or_si512(
                            _mm512_slli_epi16::<6>(_mm512_and_si512(t0, _mm512_set1_epi16(0x1F))),
                            b1,
                        );
                        let cp3 = _mm512_or_si512(
                            _mm512_or_si512(
                                _mm512_slli_epi16::<12>(t0),
                                _mm512_slli_epi16::<6>(b1),
                            ),
                            b2,
                        );
                        let mut r = _mm512_mask_mov_epi16(
                            t0,
                            _mm512_cmpgt_epu16_mask(t0, _mm512_set1_epi16(0xBF)),
                            cp2,
                        );
                        r = _mm512_mask_mov_epi16(
                            r,
                            _mm512_cmpgt_epu16_mask(t0, _mm512_set1_epi16(0xDF)),
                            cp3,
                        );
                        let c = _mm512_maskz_compress_epi16(gm, r);
                        let n = gm.count_ones() as usize;
                        _mm512_storeu_si512(
                            dst.add(o) as *mut _,
                            _mm512_cvtepu16_epi32(_mm512_castsi512_si256(c)),
                        );
                        if n > 16 {
                            _mm512_storeu_si512(
                                dst.add(o + 16) as *mut _,
                                _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(c)),
                            );
                        }
                        o += n;
                    }
                }
            }};
        }
        macro_rules! dec32 {
            ($leads:expr) => {{
                let leads: u64 = $leads;
                for (gi, ix) in idx.iter().enumerate() {
                    let gm = (leads >> (gi * 16)) as u16;
                    if gm != 0 {
                        let r = dec_avx512(_mm512_permutexvar_epi8(*ix, z));
                        _mm512_storeu_si512(
                            dst.add(o) as *mut _,
                            _mm512_maskz_compress_epi32(gm, r),
                        );
                        o += gm.count_ones() as usize;
                    }
                }
            }};
        }
        let pm = (1u64 << P) - 1;
        let planned = match plan(hi, c0, ge!(0xC2), e0, f0, ge!(0xF5), P) {
            Ok(x) if e0 & ((1u64 << P) - 1) == 0 => Ok(x),
            Ok(x) => {
                let rb = ranges_bad(
                    eq!(0xE0),
                    eq!(0xED),
                    eq!(0xF0),
                    eq!(0xF4),
                    ge!(0xA0),
                    ge!(0x90),
                ) & pm;
                if rb == 0 {
                    Ok(x)
                } else {
                    plan_err(rb, hi & !c0, pm)
                }
            }
            Err(bad) if PREFIX => {
                let rb = if e0 & ((1u64 << P) - 1) == 0 {
                    0
                } else {
                    ranges_bad(
                        eq!(0xE0),
                        eq!(0xED),
                        eq!(0xF0),
                        eq!(0xF4),
                        ge!(0xA0),
                        ge!(0x90),
                    )
                };
                plan_err(bad | (rb & pm), hi & !c0, pm)
            }
            Err(_) => Err((0, 0)),
        };
        match planned {
            Ok((leads, adv)) if f0 & ((1u64 << P) - 1) == 0 => {
                streak = 0;
                dec16!(leads);
                i += adv;
            }
            Ok((leads, adv)) => {
                streak = 0;
                dec32!(leads);
                i += adv;
            }
            Err((leads, cut)) => {
                // With PREFIX, decode the valid prefix with SIMD first. Errors cluster, so stay
                // in scalar for a stretch that doubles with each consecutive failing block.
                let end = (i + ((P as usize) << streak)).min(len);
                if PREFIX && leads != 0 {
                    if f0 & ((1u64 << cut) - 1) == 0 {
                        dec16!(leads);
                    } else {
                        dec32!(leads);
                    }
                }
                if PREFIX {
                    i += cut;
                }
                while i < end {
                    (i, o) = scalar::step(scalar::run(src, i, end, dst, o), src, dst, mode)?;
                }
                streak = (streak + 1).min(ERR_STREAK_MAX);
            }
        }
    }
    scalar::finish(src, i, dst, o, mode)
}

pub(crate) unsafe fn convert_avx512(
    src: &[u8],
    dst: *mut u32,
    mode: u8,
) -> Result<usize, Utf32Error> {
    avx512_impl(src, dst, mode)
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn ascii_len_avx512_impl(src: &[u8]) -> usize {
    let mut i = 0;
    while i + 64 <= src.len() {
        let m = _mm512_movepi8_mask(_mm512_loadu_si512(src.as_ptr().add(i) as *const _));
        if m != 0 {
            return i + m.trailing_zeros() as usize;
        }
        i += 64;
    }
    i + scalar::ascii_len(&src[i..])
}
pub(crate) unsafe fn ascii_len_avx512(src: &[u8]) -> usize {
    ascii_len_avx512_impl(src)
}
