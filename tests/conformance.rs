use std::mem::MaybeUninit;
use utf8_to_utf32::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Interesting byte sequences: valid of every width, edge code points, and every class of invalid.
const SPECIALS: &[&[u8]] = &[
    b"a",
    b"\x00",
    b"\x7f",
    b"\x1b[31m",
    "é".as_bytes(),
    "\u{80}".as_bytes(),
    "\u{7ff}".as_bytes(),
    "\u{800}".as_bytes(),
    "─".as_bytes(),
    "中".as_bytes(),
    "\u{d7ff}".as_bytes(),
    "\u{e000}".as_bytes(),
    "\u{fffd}".as_bytes(),
    "\u{ffff}".as_bytes(),
    "\u{10000}".as_bytes(),
    "😀".as_bytes(),
    "\u{10ffff}".as_bytes(),
    // invalid
    b"\x80",
    b"\x9b",
    b"\x9f",
    b"\xa0",
    b"\xbf",
    b"\xc0\x80",
    b"\xc1\xbf",
    b"\xc2",
    b"\xdf",
    b"\xe0\x80\x80",
    b"\xe0\x9f\xbf",
    b"\xe0\xa0",
    b"\xe2\x94",
    b"\xed\xa0\x80",
    b"\xed\xbf\xbf",
    b"\xef\xbf",
    b"\xf0\x80\x80\x80",
    b"\xf0\x8f\xbf\xbf",
    b"\xf0\x9f\x98",
    b"\xf0\x9f",
    b"\xf4\x90\x80\x80",
    b"\xf4\x8f\xbf",
    b"\xf5\x80\x80\x80",
    b"\xf8\x88\x80\x80\x80",
    b"\xfe",
    b"\xff",
    b"\xc2\x41",
    b"\xe2\x41\x41",
    b"\xf0\x9f\x41",
];

fn gen(r: &mut Rng, n: usize) -> Vec<u8> {
    let mut v = Vec::new();
    let mode = r.below(4);
    while v.len() < n {
        match (mode, r.below(10)) {
            (0, _) => v.extend_from_slice(SPECIALS[r.below(SPECIALS.len() as u64) as usize]),
            (1, 0) => v.push(r.next() as u8),
            (1, _) | (2, 0..=4) => {
                let c = char::from_u32(r.below(0x11_0000) as u32).unwrap_or('x');
                v.extend_from_slice(c.to_string().as_bytes());
            }
            (2, 5..=7) => v.extend_from_slice("│ ✓ 中文 é".as_bytes()),
            (2, _) => v.extend_from_slice(SPECIALS[r.below(SPECIALS.len() as u64) as usize]),
            (_, 0) => v.extend_from_slice(SPECIALS[r.below(SPECIALS.len() as u64) as usize]),
            (_, _) => {
                let k = r.below(40) as usize;
                v.extend(std::iter::repeat_n(b'a' + (r.below(26) as u8), k))
            }
        }
    }
    v
}

fn reference_lossy(s: &[u8], c1: bool) -> Vec<u32> {
    let mut out = Vec::new();
    for ch in s.utf8_chunks() {
        out.extend(ch.valid().chars().map(|c| c as u32));
        let inv = ch.invalid();
        if c1 && inv.len() == 1 && (0x80..=0x9f).contains(&inv[0]) {
            out.push(inv[0] as u32);
        } else if !inv.is_empty() {
            out.push(0xFFFD);
        }
    }
    out
}

fn check(level: Level, s: &[u8]) {
    // strict
    let mut dst = vec![MaybeUninit::<u32>::uninit(); s.len()];
    let got = utf8_to_utf32_with(level, s, &mut dst);
    match std::str::from_utf8(s) {
        Ok(st) => {
            let exp: Vec<u32> = st.chars().map(|c| c as u32).collect();
            let n =
                got.unwrap_or_else(|e| panic!("{level:?} rejected valid input {e:?}: {s:02x?}"));
            let out: Vec<u32> = dst[..n]
                .iter()
                .map(|x| unsafe { x.assume_init() })
                .collect();
            assert_eq!(out, exp, "{level:?} mismatch on {s:02x?}");
        }
        Err(e) => {
            let ge = got.expect_err(&format!("{level:?} accepted invalid {s:02x?}"));
            assert_eq!(
                ge.valid_up_to,
                e.valid_up_to(),
                "{level:?} valid_up_to on {s:02x?}"
            );
            assert_eq!(
                ge.error_len,
                e.error_len().map(|x| x as u8),
                "{level:?} error_len on {s:02x?}"
            );
            let exp: Vec<u32> = std::str::from_utf8(&s[..e.valid_up_to()])
                .unwrap()
                .chars()
                .map(|c| c as u32)
                .collect();
            assert_eq!(ge.written, exp.len());
            let out: Vec<u32> = dst[..ge.written]
                .iter()
                .map(|x| unsafe { x.assume_init() })
                .collect();
            assert_eq!(out, exp, "{level:?} prefix mismatch on {s:02x?}");
        }
    }
    for c1 in [false, true] {
        let mut d = vec![MaybeUninit::<u32>::uninit(); s.len()];
        let n = utf8_to_utf32_lossy_with(level, s, &mut d, c1);
        let got: Vec<u32> = d[..n].iter().map(|x| unsafe { x.assume_init() }).collect();
        assert_eq!(got, reference_lossy(s, c1), "{level:?} lossy slice c1={c1}");
    }
    #[cfg(feature = "alloc")]
    check_alloc(level, s);
}

/// The Vec-based APIs: `lossy_into` and the `Decoder`.
#[cfg(feature = "alloc")]
fn check_alloc(level: Level, s: &[u8]) {
    // lossy, both policies; also against String::from_utf8_lossy
    for c1 in [false, true] {
        let mut out = vec![7u32];
        utf8_to_utf32_lossy_into_with(level, s, &mut out, c1);
        assert_eq!(
            &out[1..],
            &reference_lossy(s, c1)[..],
            "{level:?} lossy c1={c1} on {s:02x?}"
        );
    }
    let mut out = Vec::new();
    utf8_to_utf32_lossy_into_with(level, s, &mut out, false);
    let std_lossy: Vec<u32> = String::from_utf8_lossy(s)
        .chars()
        .map(|c| c as u32)
        .collect();
    assert_eq!(out, std_lossy);
    // pieces
    let mut dec = Decoder::with_level(level);
    let mut pcs = dec.pieces(s);
    let mut cps = Vec::new();
    let mut invalid = Vec::new();
    let mut consumed = 0usize;
    let mut saw_incomplete = false;
    while let Some(p) = pcs.next_piece() {
        assert!(!saw_incomplete, "piece after Incomplete");
        match p {
            Piece::Ascii(a) => {
                assert!(!a.is_empty() && a.iter().all(|b| b.is_ascii()));
                cps.extend(a.iter().map(|&b| b as u32));
                consumed += a.len();
            }
            Piece::Wide(w) => {
                assert!(!w.is_empty());
                consumed += w
                    .iter()
                    .map(|&c| char::from_u32(c).unwrap().len_utf8())
                    .sum::<usize>();
                cps.extend_from_slice(w);
            }
            Piece::Invalid(b) => {
                cps.push(0xFFFD);
                invalid.push(b.to_vec());
                consumed += b.len();
            }
            Piece::Incomplete(b) => {
                cps.push(0xFFFD);
                invalid.push(b.to_vec());
                consumed += b.len();
                saw_incomplete = true;
            }
        }
        assert_eq!(consumed, pcs.position());
    }
    assert_eq!(consumed, s.len());
    assert_eq!(cps, std_lossy, "{level:?} pieces on {s:02x?}");
    let exp_inv: Vec<Vec<u8>> = s
        .utf8_chunks()
        .map(|c| c.invalid().to_vec())
        .filter(|v| !v.is_empty())
        .collect();
    assert_eq!(invalid, exp_inv);
}

fn levels() -> Vec<Level> {
    let l: Vec<Level> = Level::supported().collect();
    eprintln!("testing levels: {l:?}");
    l
}

#[test]
fn random_fuzz() {
    let iters: u64 = std::env::var("FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    for level in levels() {
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        for _ in 0..iters {
            let n = match r.below(4) {
                0 => r.below(8),
                1 => r.below(80),
                2 => r.below(300),
                _ => r.below(3000),
            } as usize;
            let s = gen(&mut r, n);
            check(level, &s);
        }
    }
}

/// Every special sequence at every offset 0..=140 inside ASCII / non-ASCII fill,
/// so it lands on every block boundary and lookahead position of every level.
#[test]
fn block_boundaries() {
    let fills: [&[u8]; 3] = [b"abcdefgh", "é".as_bytes(), "─中".as_bytes()];
    for level in levels() {
        for fill in fills {
            for &sp in SPECIALS {
                for off in 0..=140usize {
                    for tail in [0usize, 1, 2, 3, 70] {
                        let mut s: Vec<u8> = b"x".repeat(off);
                        if fill != b"abcdefgh" {
                            // valid non-ASCII prefix of exactly `off` bytes is not always possible; mix.
                            s.clear();
                            while s.len() + fill.len() <= off {
                                s.extend_from_slice(fill);
                            }
                            while s.len() < off {
                                s.push(b'.');
                            }
                        }
                        s.extend_from_slice(sp);
                        s.extend(fill.iter().cycle().take(tail));
                        check(level, &s);
                    }
                }
            }
        }
    }
}

/// Runs of errors at several spacings between clean text, so that the x86 kernels'
/// scalar stretch after a failing block (and the `Decoder`'s after a failing window) grows,
/// ends inside an error run, and resets.
#[test]
fn error_runs() {
    let clean = "é─中😀 ascii ".repeat(40);
    for level in levels() {
        for gap in [0usize, 1, 2, 7, 13, 30, 61, 64, 127, 129, 200, 520] {
            for errs in [1usize, 3, 40] {
                for bad in [&b"\xff"[..], b"\x9b", b"\xe2\x94", b"\xf0\x9f\x98"] {
                    let mut s = clean.as_bytes()[..76].to_vec();
                    for _ in 0..errs {
                        s.extend_from_slice(bad);
                        // A gap that cuts a character only adds an error.
                        s.extend(clean.bytes().take(gap));
                    }
                    s.extend_from_slice(clean.as_bytes());
                    check(level, &s);
                }
            }
        }
    }
}

/// All 2-byte pairs and all (lead, second) pairs of 3/4-byte leads, at several offsets.
#[test]
fn exhaustive_pairs() {
    for level in levels() {
        for off in [0usize, 13, 15, 16, 29, 31, 45, 47, 57, 59, 63] {
            for a in 0x80u32..=0xFF {
                for b in 0u32..=0xFF {
                    for tail in [
                        &b""[..],
                        b"\x80",
                        b"\x80\x80",
                        b"zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
                    ] {
                        let mut s = b"y".repeat(off);
                        s.push(a as u8);
                        s.push(b as u8);
                        s.extend_from_slice(tail);
                        s.extend_from_slice(&b"q".repeat(64));
                        let mut dst = vec![MaybeUninit::<u32>::uninit(); s.len()];
                        let got = utf8_to_utf32_with(level, &s, &mut dst);
                        match std::str::from_utf8(&s) {
                            Ok(_) => assert!(got.is_ok(), "{level:?} {s:02x?}"),
                            Err(e) => {
                                let g = got.expect_err("must reject");
                                assert_eq!(
                                    (g.valid_up_to, g.error_len),
                                    (e.valid_up_to(), e.error_len().map(|x| x as u8)),
                                    "{level:?} {s:02x?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Every Unicode scalar value, encoded, in a long buffer (exercises all SIMD paths).
#[test]
fn all_scalars() {
    let s: String = (0..=0x10FFFFu32).filter_map(char::from_u32).collect();
    let exp: Vec<u32> = s.chars().map(|c| c as u32).collect();
    for level in levels() {
        let mut dst = vec![MaybeUninit::<u32>::uninit(); s.len()];
        let n = utf8_to_utf32_with(level, s.as_bytes(), &mut dst).unwrap();
        assert_eq!(n, exp.len());
        assert!(
            dst[..n]
                .iter()
                .zip(&exp)
                .all(|(a, b)| unsafe { a.assume_init() } == *b),
            "{level:?}"
        );
    }
}

#[cfg(feature = "alloc")]
#[test]
fn dispatch_default() {
    let s = "héllo ─ 中 😀".repeat(50);
    let mut out = Vec::new();
    utf8_to_utf32_lossy_into(s.as_bytes(), &mut out, true);
    assert_eq!(out, s.chars().map(|c| c as u32).collect::<Vec<_>>());
    eprintln!("active level: {:?}", active_level());
}
