//! Fast UTF-8 to UTF-32 conversion in pure Rust, with runtime SIMD dispatch.
//!
//! Converts UTF-8 bytes to Unicode scalar values (`u32`), validating as it
//! goes, with exactly the error semantics of [`core::str::from_utf8`]
//! (`valid_up_to` / `error_len`, maximal-subpart error lengths).
//!
//! # Implementations
//!
//! | [`Level`] | CPU features | Notes |
//! |---|---|---|
//! | `Scalar` | none | portable; also used for tails and error resolution |
//! | `Sse41` | SSSE3, SSE4.1, POPCNT | 16-byte blocks (x86/x86_64) |
//! | `Avx2` | AVX2, POPCNT | 32-byte blocks (x86/x86_64) |
//! | `Avx512` | AVX-512 F, BW, VBMI, VBMI2 | 48-byte blocks, `vpcompressw`/`vpcompressd` (x86/x86_64) |
//! | `Neon` | NEON | 64-byte blocks, `tbl` validation and compaction (aarch64) |
//! | `Vsx` | VSX (nightly, `vsx` feature) | 64-byte blocks, `vec_perm` validation and compaction (POWER9) |
//! | `Rvv` | RVV (nightly, `rvv` feature) | 32-byte blocks on VLEN=256, SEW=8 validation, SEW=32 compaction (riscv64) |
//!
//! With the `std` feature (default) the best level is detected at runtime and
//! cached as a function pointer; the `UTF32_LEVEL` environment variable
//! (`scalar`, `sse4.1`, `avx2`, `avx512`, `neon`, `vsx`, `rvv`) can force a level. Without
//! `std` the level is chosen from compile-time target features. `Vsx` needs a nightly
//! toolchain and the `vsx` Cargo feature. `Rvv` needs a nightly toolchain, the `rvv`
//! Cargo feature, the `V` ISA extension and VLEN=256.
//!
//! # Examples
//!
//! ```
//! use core::mem::MaybeUninit;
//! let src = "héllo ─ 中".as_bytes();
//! let mut dst = vec![MaybeUninit::<u32>::uninit(); src.len()];
//! let n = utf8_to_utf32::utf8_to_utf32(src, &mut dst).unwrap();
//! assert_eq!(n, 9);
//!
//! let mut out = Vec::new();
//! utf8_to_utf32::utf8_to_utf32_lossy_into(b"a\x9bb\xff", &mut out, true);
//! assert_eq!(out, [0x61, 0x9b, 0x62, 0xFFFD]);
//! ```
//!
//! # Features
//!
//! - `std` (default): runtime CPU feature detection; implies `alloc`.
//! - `alloc`: `Vec`-based APIs ([`utf8_to_utf32_lossy_into`], [`Decoder`]).
//! - `vsx`: POWER9 VSX kernel; needs a nightly toolchain. On POWER8 the level
//!   is unsupported and the scalar kernel runs instead.
//! - `rvv`: riscv64 RVV kernel; needs a nightly toolchain.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(
    all(target_arch = "powerpc64", feature = "vsx"),
    feature(stdarch_powerpc, powerpc_target_feature)
)]
#![cfg_attr(
    all(target_arch = "powerpc64", feature = "vsx", feature = "std"),
    feature(stdarch_powerpc_feature_detection)
)]
#![cfg_attr(
    all(target_arch = "riscv64", feature = "rvv"),
    feature(riscv_target_feature)
)]
#![cfg_attr(
    all(target_arch = "riscv64", feature = "rvv", feature = "std"),
    feature(stdarch_riscv_feature_detection)
)]
#![deny(missing_docs)]
#![warn(clippy::all)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicPtr, AtomicU8, Ordering};

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod mask;
#[cfg(all(target_arch = "powerpc64", feature = "vsx"))]
mod powerpc64;
#[cfg(all(target_arch = "riscv64", feature = "rvv"))]
mod riscv64;
mod scalar;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

/// Conversion error, with [`core::str::Utf8Error`] semantics plus the number of
/// code points written before the error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Utf32Error {
    /// Number of bytes of `src` that were valid (and converted) before the error.
    pub valid_up_to: usize,
    /// Number of code points written to `dst` (the conversion of `src[..valid_up_to]`).
    pub written: usize,
    /// Length of the maximal invalid subpart (1..=3), or `None` if `src` ends
    /// with a valid but truncated sequence (carry it over to the next read).
    pub error_len: Option<u8>,
}

impl core::fmt::Display for Utf32Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.error_len {
            Some(n) => write!(
                f,
                "invalid utf-8 sequence of {} bytes from index {}",
                n, self.valid_up_to
            ),
            None => write!(
                f,
                "incomplete utf-8 byte sequence from index {}",
                self.valid_up_to
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Utf32Error {}

/// Implementation level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Level {
    /// Portable scalar code.
    Scalar = 1,
    /// SSSE3 + SSE4.1 (+ POPCNT), 128-bit.
    Sse41 = 2,
    /// AVX2 (+ POPCNT), 256-bit.
    Avx2 = 3,
    /// AVX-512 F + BW + VBMI + VBMI2 (+ POPCNT), 512-bit.
    Avx512 = 4,
    /// ARM NEON, 128-bit.
    Neon = 5,
    /// POWER9 VSX, 128-bit (nightly + `vsx` feature).
    Vsx = 6,
    /// RISC-V RVV, 256-bit (nightly + `rvv` feature, requires V).
    Rvv = 7,
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
macro_rules! has {
    ($($f:tt),*) => { true $(&& std::is_x86_feature_detected!($f))* };
}
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(feature = "std")))]
macro_rules! has {
    ($($f:tt),*) => { true $(&& cfg!(target_feature = $f))* };
}
#[cfg(all(target_arch = "powerpc64", feature = "vsx", feature = "std"))]
macro_rules! has_ppc {
    ($detected:tt, $target_feature:tt) => {
        std::arch::is_powerpc_feature_detected!($detected)
    };
}
#[cfg(all(target_arch = "powerpc64", feature = "vsx", not(feature = "std")))]
macro_rules! has_ppc {
    ($detected:tt, $target_feature:tt) => {
        cfg!(target_feature = $target_feature)
    };
}

impl Level {
    /// All levels, lowest first.
    pub const ALL: [Level; 7] = [
        Level::Scalar,
        Level::Sse41,
        Level::Avx2,
        Level::Avx512,
        Level::Neon,
        Level::Vsx,
        Level::Rvv,
    ];

    /// Whether this level can run here (runtime detection with `std`,
    /// compile-time target features without).
    pub fn is_supported(self) -> bool {
        match self {
            Level::Scalar => true,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Sse41 => has!("ssse3", "sse4.1", "popcnt"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Avx2 => has!("avx2", "popcnt"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Avx512 => has!("avx512f", "avx512bw", "avx512vbmi", "avx512vbmi2", "popcnt"),
            #[cfg(target_arch = "aarch64")]
            Level::Neon => true,
            #[cfg(all(target_arch = "powerpc64", feature = "vsx"))]
            Level::Vsx => has_ppc!("power9", "power9-vector"),
            #[cfg(all(target_arch = "riscv64", feature = "rvv"))]
            Level::Rvv => riscv64::supported(),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// Best supported level (ignores `UTF32_LEVEL`).
    pub fn detect() -> Level {
        Level::supported().last().unwrap_or(Level::Scalar)
    }

    /// Supported levels, lowest first.
    pub fn supported() -> impl DoubleEndedIterator<Item = Level> {
        Level::ALL.iter().copied().filter(|l| l.is_supported())
    }

    /// Short name: `scalar`, `sse4.1`, `avx2`, `avx512`, `neon`, `vsx`, `rvv`.
    pub fn name(self) -> &'static str {
        match self {
            Level::Scalar => "scalar",
            Level::Sse41 => "sse4.1",
            Level::Avx2 => "avx2",
            Level::Avx512 => "avx512",
            Level::Neon => "neon",
            Level::Vsx => "vsx",
            Level::Rvv => "rvv",
        }
    }

    /// Parse a [`Level::name`] (also accepts `sse41`, case-insensitive).
    pub fn from_name(s: &str) -> Option<Level> {
        if s.eq_ignore_ascii_case("sse41") {
            return Some(Level::Sse41);
        }
        Level::ALL
            .iter()
            .copied()
            .find(|l| l.name().eq_ignore_ascii_case(s))
    }

    fn kernel(self) -> Kernel {
        assert!(
            self.is_supported(),
            "level {:?} not supported on this CPU",
            self
        );
        match self {
            Level::Scalar => scalar::convert,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Sse41 => x86::convert_sse41,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Avx2 => x86::convert_avx2,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Avx512 => x86::convert_avx512,
            #[cfg(target_arch = "aarch64")]
            Level::Neon => aarch64::convert_neon,
            #[cfg(all(target_arch = "powerpc64", feature = "vsx"))]
            Level::Vsx => powerpc64::convert_vsx,
            #[cfg(all(target_arch = "riscv64", feature = "rvv"))]
            Level::Rvv => riscv64::convert_rvv,
            #[allow(unreachable_patterns)]
            _ => scalar::convert,
        }
    }

    #[cfg(feature = "alloc")]
    fn ascii_len(self) -> AsciiLen {
        match self {
            Level::Scalar => scalar::ascii_len,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Sse41 => x86::ascii_len_sse2,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Avx2 => x86::ascii_len_avx2,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Avx512 => x86::ascii_len_avx512,
            #[cfg(target_arch = "aarch64")]
            Level::Neon => aarch64::ascii_len_neon,
            #[cfg(all(target_arch = "powerpc64", feature = "vsx"))]
            Level::Vsx => powerpc64::ascii_len_vsx,
            #[cfg(all(target_arch = "riscv64", feature = "rvv"))]
            Level::Rvv => riscv64::ascii_len_rvv,
            #[allow(unreachable_patterns)]
            _ => scalar::ascii_len,
        }
    }
}

/// Safety: `dst` must be valid for writes of `src.len()` u32s.
type Kernel = unsafe fn(&[u8], *mut u32, u8) -> Result<usize, Utf32Error>;
pub(crate) const STRICT: u8 = 0;
pub(crate) const LOSSY: u8 = 1;
pub(crate) const LOSSY_C1: u8 = 2;
#[cfg(feature = "alloc")]
type AsciiLen = unsafe fn(&[u8]) -> usize;

static ACTIVE: AtomicU8 = AtomicU8::new(0);
static KERNEL: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());

/// Level used by the functions without an explicit level: with `std`, the
/// `UTF32_LEVEL` environment variable if set and supported, else
/// [`Level::detect`]. Determined once and cached.
pub fn active_level() -> Level {
    let v = ACTIVE.load(Ordering::Relaxed);
    if v != 0 {
        return Level::ALL[v as usize - 1];
    }
    #[cfg(feature = "std")]
    let l = std::env::var("UTF32_LEVEL")
        .ok()
        .and_then(|s| Level::from_name(&s))
        .filter(|l| l.is_supported())
        .unwrap_or_else(Level::detect);
    #[cfg(not(feature = "std"))]
    let l = Level::detect();
    ACTIVE.store(l as u8, Ordering::Relaxed);
    l
}

#[inline]
fn active_kernel() -> Kernel {
    let p = KERNEL.load(Ordering::Relaxed);
    if !p.is_null() {
        // SAFETY: only ever stores a `Kernel` fn pointer.
        return unsafe { core::mem::transmute::<*mut (), Kernel>(p) };
    }
    let k = active_level().kernel();
    KERNEL.store(k as *mut (), Ordering::Relaxed);
    k
}

/// Converts UTF-8 `src` to UTF-32 in `dst`, returning the number of code points written.
///
/// On invalid input, returns the error with `std` semantics; `dst[..err.written]`
/// then holds the conversion of the valid prefix.
///
/// # Panics
///
/// If `dst.len() < src.len()` (the kernels write whole vectors, so the full
/// worst-case length is required even when fewer code points result).
#[inline]
pub fn utf8_to_utf32(src: &[u8], dst: &mut [MaybeUninit<u32>]) -> Result<usize, Utf32Error> {
    assert!(
        dst.len() >= src.len(),
        "dst must be at least src.len() long"
    );
    // SAFETY: dst has room for src.len() u32s.
    unsafe { active_kernel()(src, dst.as_mut_ptr() as *mut u32, STRICT) }
}

/// [`utf8_to_utf32`] at an explicit level.
///
/// # Panics
///
/// If `level` is not supported, or `dst.len() < src.len()`.
pub fn utf8_to_utf32_with(
    level: Level,
    src: &[u8],
    dst: &mut [MaybeUninit<u32>],
) -> Result<usize, Utf32Error> {
    assert!(
        dst.len() >= src.len(),
        "dst must be at least src.len() long"
    );
    // SAFETY: dst has room for src.len() u32s; kernel() checks support.
    unsafe { level.kernel()(src, dst.as_mut_ptr() as *mut u32, STRICT) }
}

/// Lossy conversion into a slice, returning the number of code points written.
///
/// Each maximal invalid subpart becomes one U+FFFD (the same output as
/// `String::from_utf8_lossy(src).chars()`). With `c1_as_controls`, a lone
/// invalid byte in `0x80..=0x9F` is written as its raw value instead
/// (terminal C1 control semantics).
///
/// # Panics
///
/// If `dst.len() < src.len()`.
#[inline]
pub fn utf8_to_utf32_lossy(
    src: &[u8],
    dst: &mut [MaybeUninit<u32>],
    c1_as_controls: bool,
) -> usize {
    lossy_slice(active_kernel(), src, dst, c1_as_controls)
}

/// [`utf8_to_utf32_lossy`] at an explicit level.
///
/// # Panics
///
/// If `level` is not supported, or `dst.len() < src.len()`.
pub fn utf8_to_utf32_lossy_with(
    level: Level,
    src: &[u8],
    dst: &mut [MaybeUninit<u32>],
    c1_as_controls: bool,
) -> usize {
    lossy_slice(level.kernel(), src, dst, c1_as_controls)
}

fn lossy_slice(kernel: Kernel, src: &[u8], dst: &mut [MaybeUninit<u32>], c1: bool) -> usize {
    assert!(
        dst.len() >= src.len(),
        "dst must be at least src.len() long"
    );
    let mode = if c1 { LOSSY_C1 } else { LOSSY };
    // SAFETY: dst has room for src.len() u32s; lossy mode never fails.
    match unsafe { kernel(src, dst.as_mut_ptr() as *mut u32, mode) } {
        Ok(n) => n,
        Err(_) => unreachable!("lossy conversion never fails"),
    }
}

/// Lossy conversion appending to `out`; see [`utf8_to_utf32_lossy`].
#[cfg(feature = "alloc")]
#[inline]
pub fn utf8_to_utf32_lossy_into(src: &[u8], out: &mut Vec<u32>, c1_as_controls: bool) {
    lossy_vec(active_kernel(), src, out, c1_as_controls)
}

/// [`utf8_to_utf32_lossy_into`] at an explicit level.
///
/// # Panics
///
/// If `level` is not supported.
#[cfg(feature = "alloc")]
pub fn utf8_to_utf32_lossy_into_with(
    level: Level,
    src: &[u8],
    out: &mut Vec<u32>,
    c1_as_controls: bool,
) {
    lossy_vec(level.kernel(), src, out, c1_as_controls)
}

#[cfg(feature = "alloc")]
fn lossy_vec(kernel: Kernel, src: &[u8], out: &mut Vec<u32>, c1: bool) {
    out.reserve(src.len());
    let base = out.len();
    // SAFETY: reserve() guarantees src.len() spare elements after `base`.
    let spare = unsafe {
        core::slice::from_raw_parts_mut(
            out.as_mut_ptr().add(base) as *mut MaybeUninit<u32>,
            src.len(),
        )
    };
    let n = lossy_slice(kernel, src, spare, c1);
    // SAFETY: the kernel initialised `n` elements.
    unsafe { out.set_len(base + n) };
}

/// A piece of input produced by [`Pieces::next_piece`].
#[cfg(feature = "alloc")]
#[derive(Debug, PartialEq, Eq)]
pub enum Piece<'a> {
    /// A run of ASCII bytes (>= `Decoder::ASCII_MIN` long, or reaching the end of input).
    Ascii(&'a [u8]),
    /// Decoded code points; may contain short ASCII runs between non-ASCII chars.
    Wide(&'a [u32]),
    /// A maximal invalid subpart (1..=3 bytes).
    Invalid(&'a [u8]),
    /// A valid but truncated sequence at the very end of `src` (carry it to the next read).
    Incomplete(&'a [u8]),
}

/// Chunk decoder with an internal scratch buffer.
///
/// Splits input into ASCII runs, decoded runs, and errors; the decoded
/// pieces borrow the decoder's buffer, so iteration is lending-style via
/// [`Pieces::next_piece`] rather than `Iterator`.
///
/// ```
/// use utf8_to_utf32::{Decoder, Piece};
/// let mut dec = Decoder::new();
/// let mut it = dec.pieces(b"plain ascii text here, then \xe2\x94\x80 and \x9b");
/// while let Some(p) = it.next_piece() {
///     match p {
///         Piece::Ascii(a) => { /* print a run of ASCII */ let _ = a; }
///         Piece::Wide(w) => { /* code points */ let _ = w; }
///         Piece::Invalid(b) => assert_eq!(b, b"\x9b"),
///         Piece::Incomplete(_) => unreachable!(),
///     }
/// }
/// ```
#[cfg(feature = "alloc")]
pub struct Decoder {
    scratch: Vec<u32>,
    kernel: Kernel,
    ascii: AsciiLen,
}

#[cfg(feature = "alloc")]
impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "alloc")]
impl Decoder {
    /// Minimum ASCII run (when followed by non-ASCII) reported as `Piece::Ascii`.
    pub const ASCII_MIN: usize = 16;
    /// Max bytes decoded into one `Piece::Wide`.
    pub const CHUNK: usize = 16 * 1024;

    /// Decoder at [`active_level`].
    pub fn new() -> Self {
        Self::with_level(active_level())
    }

    /// Decoder at an explicit level (panics if unsupported).
    pub fn with_level(level: Level) -> Self {
        Decoder {
            scratch: Vec::with_capacity(Self::CHUNK + 4),
            kernel: level.kernel(),
            ascii: level.ascii_len(),
        }
    }

    /// Start decoding `src`.
    pub fn pieces<'a>(&'a mut self, src: &'a [u8]) -> Pieces<'a> {
        Pieces {
            dec: self,
            src,
            pos: 0,
            pending: None,
        }
    }
}

/// In-progress decode of one input; see [`Decoder`].
#[cfg(feature = "alloc")]
pub struct Pieces<'a> {
    dec: &'a mut Decoder,
    src: &'a [u8],
    pos: usize,
    /// Error found by the previous Wide decode: (position, error_len).
    pending: Option<(usize, u8)>,
}

#[cfg(feature = "alloc")]
impl<'a> Pieces<'a> {
    /// Bytes of `src` consumed so far.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Next piece, or `None` at the end of input.
    #[allow(clippy::should_implement_trait)]
    pub fn next_piece(&mut self) -> Option<Piece<'_>> {
        let src = self.src;
        let len = src.len();
        let pos = self.pos;
        if pos >= len {
            return None;
        }
        if let Some((ep, n)) = self.pending.take() {
            if ep == pos {
                self.pos = pos + n as usize;
                return Some(Piece::Invalid(&src[pos..pos + n as usize]));
            }
        }
        let r = unsafe { (self.dec.ascii)(&src[pos..]) };
        if r >= Decoder::ASCII_MIN || pos + r == len {
            self.pos += r;
            return Some(Piece::Ascii(&src[pos..pos + r]));
        }
        let mut end = (pos + Decoder::CHUNK).min(len);
        let lim = (end + 3).min(len);
        while end < lim && src[end] & 0xC0 == 0x80 {
            end += 1;
        }
        let win = &src[pos..end];
        let sc = &mut self.dec.scratch;
        sc.clear();
        sc.reserve(win.len());
        let res = unsafe { (self.dec.kernel)(win, sc.as_mut_ptr(), STRICT) };
        match res {
            Ok(n) => {
                unsafe { sc.set_len(n) };
                self.pos = end;
                Some(Piece::Wide(&sc[..]))
            }
            Err(e) if e.valid_up_to > 0 => {
                unsafe { sc.set_len(e.written) };
                self.pos = pos + e.valid_up_to;
                self.pending = e.error_len.map(|n| (self.pos, n));
                Some(Piece::Wide(&sc[..]))
            }
            Err(_) => {
                // Resolve against the full input (the window may have cut a sequence).
                match scalar::decode_one(src, pos) {
                    Err(Some(n)) => {
                        self.pos = pos + n as usize;
                        Some(Piece::Invalid(&src[pos..pos + n as usize]))
                    }
                    Err(None) => {
                        self.pos = len;
                        Some(Piece::Incomplete(&src[pos..]))
                    }
                    Ok(_) => unreachable!("window error must be an error on full input"),
                }
            }
        }
    }
}
