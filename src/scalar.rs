//! Portable scalar decoder; exact std error semantics.

use crate::Utf32Error;

/// Decode one non-ASCII sequence starting at `i`.
/// Ok((code point, length)) or Err(error_len) with `None` = truncated at end.
#[inline(always)]
pub(crate) fn decode_one(src: &[u8], i: usize) -> Result<(u32, usize), Option<u8>> {
    let len = src.len();
    let b0 = src[i];
    let get = |k: usize| if i + k < len { Some(src[i + k]) } else { None };
    let cont = |b: u8| b & 0xC0 == 0x80;
    match b0 {
        0x00..=0x7F => Ok((b0 as u32, 1)),
        0xC2..=0xDF => {
            let b1 = get(1).ok_or(None)?;
            if !cont(b1) {
                return Err(Some(1));
            }
            Ok((((b0 as u32 & 0x1F) << 6) | (b1 as u32 & 0x3F), 2))
        }
        0xE0..=0xEF => {
            let b1 = get(1).ok_or(None)?;
            let ok = match b0 {
                0xE0 => (0xA0..=0xBF).contains(&b1),
                0xED => (0x80..=0x9F).contains(&b1),
                _ => cont(b1),
            };
            if !ok {
                return Err(Some(1));
            }
            let b2 = get(2).ok_or(None)?;
            if !cont(b2) {
                return Err(Some(2));
            }
            Ok((
                ((b0 as u32 & 0x0F) << 12) | ((b1 as u32 & 0x3F) << 6) | (b2 as u32 & 0x3F),
                3,
            ))
        }
        0xF0..=0xF4 => {
            let b1 = get(1).ok_or(None)?;
            let ok = match b0 {
                0xF0 => (0x90..=0xBF).contains(&b1),
                0xF4 => (0x80..=0x8F).contains(&b1),
                _ => cont(b1),
            };
            if !ok {
                return Err(Some(1));
            }
            let b2 = get(2).ok_or(None)?;
            if !cont(b2) {
                return Err(Some(2));
            }
            let b3 = get(3).ok_or(None)?;
            if !cont(b3) {
                return Err(Some(3));
            }
            Ok((
                ((b0 as u32 & 0x07) << 18)
                    | ((b1 as u32 & 0x3F) << 12)
                    | ((b2 as u32 & 0x3F) << 6)
                    | (b3 as u32 & 0x3F),
                4,
            ))
        }
        _ => Err(Some(1)),
    }
}

/// Decode chars starting at `i` until `i >= until` (stops on a char boundary).
/// Safety: `dst` valid for `src.len()` writes and `o <= i`.
#[inline(always)]
pub(crate) unsafe fn run(
    src: &[u8],
    mut i: usize,
    until: usize,
    dst: *mut u32,
    mut o: usize,
) -> Result<(usize, usize), Utf32Error> {
    let len = src.len();
    let p = src.as_ptr();
    while i < until {
        if i + 8 <= len {
            let w = (p.add(i) as *const u64).read_unaligned();
            if w & 0x8080_8080_8080_8080 == 0 {
                for k in 0..8 {
                    *dst.add(o + k) = *p.add(i + k) as u32;
                }
                i += 8;
                o += 8;
                continue;
            }
        }
        let b0 = *p.add(i);
        if b0 < 0x80 {
            *dst.add(o) = b0 as u32;
            i += 1;
            o += 1;
            continue;
        }
        match decode_one(src, i) {
            Ok((cp, n)) => {
                *dst.add(o) = cp;
                i += n;
                o += 1;
            }
            Err(el) => {
                return Err(Utf32Error {
                    valid_up_to: i,
                    written: o,
                    error_len: el,
                })
            }
        }
    }
    Ok((i, o))
}

/// Resolve a decode error according to `mode` (0 strict, 1 lossy, 2 lossy + raw C1).
#[inline(always)]
pub(crate) unsafe fn step(
    r: Result<(usize, usize), Utf32Error>,
    src: &[u8],
    dst: *mut u32,
    mode: u8,
) -> Result<(usize, usize), Utf32Error> {
    match r {
        Ok(x) => Ok(x),
        Err(e) => fix(src, dst, e, mode),
    }
}

#[cold]
#[inline(never)]
unsafe fn fix(
    src: &[u8],
    dst: *mut u32,
    e: Utf32Error,
    mode: u8,
) -> Result<(usize, usize), Utf32Error> {
    if mode == crate::STRICT {
        return Err(e);
    }
    let i = e.valid_up_to;
    let (cp, n) = match e.error_len {
        None => (0xFFFD, src.len() - i),
        Some(n) => {
            let b = src[i];
            if mode == crate::LOSSY_C1 && n == 1 && (0x80..=0x9F).contains(&b) {
                (b as u32, 1)
            } else {
                (0xFFFD, n as usize)
            }
        }
    };
    *dst.add(e.written) = cp;
    Ok((i + n, e.written + 1))
}

/// Scalar tail, continuing past errors in lossy modes.
#[inline(always)]
pub(crate) unsafe fn finish(
    src: &[u8],
    mut i: usize,
    dst: *mut u32,
    mut o: usize,
    mode: u8,
) -> Result<usize, Utf32Error> {
    while i < src.len() {
        (i, o) = step(run(src, i, src.len(), dst, o), src, dst, mode)?;
    }
    Ok(o)
}

pub(crate) unsafe fn convert(src: &[u8], dst: *mut u32, mode: u8) -> Result<usize, Utf32Error> {
    finish(src, 0, dst, 0, mode)
}

#[cfg_attr(not(feature = "alloc"), allow(dead_code))]
pub(crate) unsafe fn ascii_len(src: &[u8]) -> usize {
    let mut i = 0;
    let p = src.as_ptr();
    while i + 8 <= src.len() {
        let w = (p.add(i) as *const u64).read_unaligned() & 0x8080_8080_8080_8080;
        if w != 0 {
            return i + (w.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < src.len() && src[i] < 0x80 {
        i += 1;
    }
    i
}
