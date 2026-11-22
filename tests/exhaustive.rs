//! Exhaustive checks of the SIMD validators against `core::str::from_utf8`.
//! Every 1..=3-byte combination is placed at several offsets of a buffer long
//! enough to take the SIMD path; 4-byte leads get sampled tails.
use std::mem::MaybeUninit;
use utf8_to_utf32::*;

/// NEON and VSX enter their loop only while `i + 80 <= len`. Shorter buffers
/// stay in the scalar tail and never touch the lookup tables.
const BUF: usize = 96;

fn check(level: Level, buf: &[u8], dst: &mut [MaybeUninit<u32>]) {
    let got = utf8_to_utf32_with(level, buf, dst);
    match (std::str::from_utf8(buf), got) {
        (Ok(s), Ok(n)) => {
            assert_eq!(n, s.chars().count(), "{buf:02x?}");
            for (k, c) in s.chars().enumerate() {
                assert_eq!(unsafe { dst[k].assume_init() }, c as u32, "{buf:02x?}");
            }
        }
        (Err(e), Err(g)) => {
            assert_eq!(g.valid_up_to, e.valid_up_to(), "{buf:02x?}");
            assert_eq!(g.error_len, e.error_len().map(|n| n as u8), "{buf:02x?}");
        }
        (a, b) => panic!("mismatch {a:?} vs {b:?} on {buf:02x?}"),
    }
}

fn run(extra: &[u8]) {
    for level in Level::supported() {
        let mut dst = vec![MaybeUninit::<u32>::uninit(); BUF];
        // 13..=16 and 29..=30 straddle 16-byte quarters. 57..=59 are the last
        // positions of the 60-position x86 blocks. 61..=63 are the bytes the
        // 64-byte block advance inspects.
        for off in [0usize, 1, 13, 14, 15, 16, 29, 30, 57, 58, 59, 61, 62, 63] {
            let mut buf = vec![b'a'; BUF];
            for b0 in 0..=255u8 {
                for b1 in 0..=255u8 {
                    for &b2 in extra {
                        buf[off] = b0;
                        buf[off + 1] = b1;
                        buf[off + 2] = b2;
                        check(level, &buf, &mut dst);
                        // also with a multibyte filler so the u16/u32 decode paths run
                        let mut buf2 = "é中".repeat(20).into_bytes();
                        buf2.truncate(BUF);
                        let o2 = off.min(buf2.len() - 4);
                        buf2[o2] = b0;
                        buf2[o2 + 1] = b1;
                        buf2[o2 + 2] = b2;
                        check(level, &buf2, &mut dst);
                    }
                }
            }
        }
    }
}

#[test]
fn all_three_byte_combos() {
    let all: Vec<u8> = (0..=255).collect();
    run(&all);
}

#[test]
fn four_byte_leads_sampled() {
    for level in Level::supported() {
        let mut dst = vec![MaybeUninit::<u32>::uninit(); BUF];
        let tails = [0x00u8, 0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0, 0xFF];
        for off in [0usize, 5, 13, 14, 15, 16, 28, 29, 30, 61, 62, 63] {
            for b0 in 0xF0..=0xF8u8 {
                for &b1 in &tails {
                    for &b2 in &tails {
                        for &b3 in &tails {
                            let mut buf = vec![b'a'; BUF];
                            buf[off..off + 4].copy_from_slice(&[b0, b1, b2, b3]);
                            check(level, &buf, &mut dst);
                            let mut buf = "😀é".repeat(20).into_bytes();
                            buf.truncate(BUF);
                            buf[off..off + 4].copy_from_slice(&[b0, b1, b2, b3]);
                            check(level, &buf, &mut dst);
                        }
                    }
                }
            }
        }
    }
}
