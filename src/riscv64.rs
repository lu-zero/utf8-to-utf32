//! riscv64 RVV kernel, vector-length agnostic.
//!
//! Rust's `core::arch` has no RVV intrinsics yet, so the vector work is inline
//! `asm!`. Vector registers can only be clobbers, so each helper sets its own
//! `vsetvli` and reads what it needs from memory.
//!
//! No block width is hardcoded: `vl` comes from `vsetvli(min(remaining - 3,
//! vlmax))`, so every block uses the whole register. Validation runs at SEW=8
//! with one byte per lane; the decode loops run at SEW=32 with `vzext.vf4` and
//! `vcompress.vm`, advancing by the runtime `vl` each iteration.

#![cfg_attr(not(feature = "alloc"), allow(dead_code))]

use crate::scalar;
use crate::Utf32Error;
use core::arch::asm;

const ERR_STRETCH: usize = 32;
const ERR_BACKOFF_MAX: u32 = 5;

#[target_feature(enable = "v")]
unsafe fn vlen_bytes() -> usize {
    let v: usize;
    unsafe {
        asm!("vsetvli {v}, zero, e8, m1, ta, ma", v = out(reg) v, options(nostack));
    }
    v
}

/// The kernel needs at least 32 byte lanes; wider registers are used in full.
pub(crate) fn supported() -> bool {
    #[cfg(feature = "std")]
    if !std::arch::is_riscv_feature_detected!("v") {
        return false;
    }
    unsafe { vlen_bytes() >= 32 }
}

/// Offset of the first non-ASCII byte in the first `n` bytes at `p`, or -1.
#[target_feature(enable = "v")]
unsafe fn non_ascii_at(p: *const u8, n: usize) -> isize {
    let first: isize;
    unsafe {
        asm!(
            "vsetvli {vl}, {n}, e8, m1, ta, ma",
            "vle8.v v1, ({p})",
            "li t0, 0x80",
            "vmsgeu.vx v0, v1, t0",
            "vfirst.m {first}, v0",
            vl = out(reg) _,
            n = in(reg) n,
            p = in(reg) p,
            first = out(reg) first,
            out("t0") _,
            out("v0") _, out("v1") _,
            options(nostack),
        );
    }
    first
}

static B1_HIGH: [u8; 32] = [
    2, 2, 2, 2, 2, 2, 2, 2, 128, 128, 128, 128, 33, 1, 21, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
];
static B1_LOW: [u8; 32] = [
    231, 163, 131, 131, 139, 203, 203, 203, 203, 203, 203, 203, 203, 219, 203, 203, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
static B2_HIGH: [u8; 32] = [
    1, 1, 1, 1, 1, 1, 1, 1, 230, 174, 186, 186, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0,
];

/// First invalid position below `adv`, or -1. Keiser-Lemire lookup validator:
/// three `vrgather.vv` nibble lookups, then all boolean logic in mask registers.
#[target_feature(enable = "v")]
unsafe fn validate_block(p: *const u8, adv: usize, n: usize) -> isize {
    let first: isize;
    unsafe {
        asm!(
            "vsetvli {vl}, {n}, e8, m1, ta, ma",
            "vle8.v v8, ({p})", "vle8.v v9, ({p1})",
            "vle8.v v20, ({t1h})", "vle8.v v21, ({t1l})", "vle8.v v22, ({t2h})",

            "vmv.v.i v10, 0", "vslideup.vi v10, v8, 1",
            "vmv.v.i v11, 0", "vslideup.vi v11, v8, 2",
            "vmv.v.i v12, 0", "vslideup.vi v12, v8, 3",

            "vsrl.vi v13, v10, 4", "vand.vi v14, v10, 15", "vsrl.vi v15, v8, 4",
            "vrgather.vv v16, v20, v13",
            "vrgather.vv v17, v21, v14",
            "vrgather.vv v18, v22, v15",
            "vand.vv v16, v16, v17", "vand.vv v16, v16, v18",
            "vmsne.vi v0, v16, 0",

            // must: a 3-byte lead two back or 4-byte lead three back requires
            // this lane to be a continuation. sc already carries TWO_CONTS.
            "li {t0}, 0xe0", "vmsgeu.vx v2, v11, {t0}",
            "li {t0}, 0xf0", "vmsgeu.vx v3, v12, {t0}",
            "vmor.mm v1, v2, v3",
            "vmxor.mm v0, v0, v1",

            // range checks after E0 / ED / F0 / F4
            "li {t0}, 0xe0", "vmseq.vx v2, v8, {t0}",
            "li {t0}, 0xa0", "vmsltu.vx v3, v9, {t0}",
            "vmand.mm v2, v2, v3",
            "li {t0}, 0xed", "vmseq.vx v3, v8, {t0}",
            "li {t0}, 0xa0", "vmsgeu.vx v4, v9, {t0}",
            "vmand.mm v3, v3, v4", "vmor.mm v2, v2, v3",
            "li {t0}, 0xf0", "vmseq.vx v3, v8, {t0}",
            "li {t0}, 0x90", "vmsltu.vx v4, v9, {t0}",
            "vmand.mm v3, v3, v4", "vmor.mm v2, v2, v3",
            "li {t0}, 0xf4", "vmseq.vx v3, v8, {t0}",
            "li {t0}, 0x90", "vmsgeu.vx v4, v9, {t0}",
            "vmand.mm v3, v3, v4", "vmor.mm v2, v2, v3",
            "vmor.mm v0, v0, v2",

            // keep only positions below adv; add claims past the cut
            "vid.v v23", "vmsltu.vx v5, v23, {adv}",
            "vmand.mm v0, v0, v5",
            "li {t0}, 0xc0", "vmsgeu.vx v7, v10, {t0}",
            "vmor.mm v7, v7, v1",
            "vmnot.m v6, v5",
            "vmand.mm v6, v6, v7",
            "li {t0}, 0x80", "vmsgeu.vx v7, v8, {t0}",
            "li {t0}, 0xc0", "vmsltu.vx v4, v8, {t0}",
            "vmand.mm v7, v7, v4", "vmnot.m v4, v7",
            "vmand.mm v6, v6, v4", "vmor.mm v0, v0, v6",

            "vfirst.m {first}, v0",

            vl = out(reg) _,
            n = in(reg) n,
            p = in(reg) p, p1 = in(reg) p.add(1),
            t1h = in(reg) B1_HIGH.as_ptr(),
            t1l = in(reg) B1_LOW.as_ptr(),
            t2h = in(reg) B2_HIGH.as_ptr(),
            adv = in(reg) adv,
            t0 = out(reg) _,
            first = out(reg) first,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
            out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _,
            out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _,
            out("v18") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            options(nostack),
        );
    }
    first
}

/// Widen `adv` ASCII bytes to u32. The loop runs at SEW=32, so one iteration
/// covers `vl` bytes and `vl` u32 lanes, whatever the vector width.
#[target_feature(enable = "v")]
unsafe fn ascii_widen(p: *const u8, dst: *mut u32, adv: usize) {
    unsafe {
        asm!(
            "vsetvli {vl}, {adv}, e32, m4, ta, ma",
            "vle8.v v4, ({p})",
            "vzext.vf4 v8, v4",
            "vse32.v v8, ({d})",
            vl = out(reg) _,
            adv = in(reg) adv,
            p = in(reg) p,
            d = in(reg) dst,
            out("v4") _, out("v5") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            options(nostack),
        );
    }
}

/// Decode `adv` bytes. Phase 1 computes lo/hi/top bytes of every candidate
/// code point at SEW=8 in one pass; phase 2 widens them to u32 at SEW=32 and
/// compacts with vcompress. Scratch buffers live on the stack.
#[target_feature(enable = "v")]
unsafe fn decode_block(p: *const u8, dst: *mut u32, o: usize, adv: usize) -> usize {
    let mut lo = [0u8; 32];
    let mut hi = [0u8; 32];
    let mut top = [0u8; 32];
    let mut b0buf = [0u8; 32];
    let mut lo_ptr = lo.as_mut_ptr();
    let mut hi_ptr = hi.as_mut_ptr();
    let mut top_ptr = top.as_mut_ptr();
    let mut b0_ptr = b0buf.as_mut_ptr();
    let mut rem = adv;
    let mut pp = p;
    let mut dd = dst.add(o);
    let mut cnt = 0usize;
    let vl: usize;
    let c: usize;
    unsafe {
        asm!(
            // ---- phase 1: SEW=8, all adv byte positions at once ----
            "mv {rem}, {adv}",
            "vsetvli {vl0}, {rem}, e8, m1, ta, ma",
            "vle8.v v1, ({p})", "vle8.v v2, ({p1})",
            "vle8.v v3, ({p2})", "vle8.v v4, ({p3})",

            "vmsgeu.vx v5, v1, {cc0}",
            "vmsgeu.vx v6, v1, {ce0}",
            "vmsgeu.vx v7, v1, {cf0}",

            // low byte candidates
            "vand.vx v8, v4, {c3f}", "vsll.vi v9, v3, 6", "vor.vv v8, v9, v8",
            "vand.vx v9, v3, {c3f}", "vsll.vi v10, v2, 6", "vor.vv v9, v10, v9",
            "vand.vx v10, v2, {c3f}", "vsll.vi v11, v1, 6", "vor.vv v10, v11, v10",
            "vmv.v.v v12, v1",
            "vmv.v.v v0, v5", "vmerge.vvm v12, v12, v10, v0",
            "vmv.v.v v0, v6", "vmerge.vvm v12, v12, v9, v0",
            "vmv.v.v v0, v7", "vmerge.vvm v12, v12, v8, v0",
            "vse8.v v12, ({dlo})",

            // high byte candidates
            "vsrl.vi v8, v1, 2", "vand.vi v8, v8, 7",
            "vand.vx v9, v2, {c3c}", "vsrl.vi v9, v9, 2",
            "vand.vi v10, v1, 15", "vsll.vi v10, v10, 4", "vor.vv v9, v9, v10",
            "vand.vx v10, v3, {c3c}", "vsrl.vi v10, v10, 2",
            "vand.vi v11, v2, 15", "vsll.vi v11, v11, 4", "vor.vv v10, v10, v11",
            "vmv.v.i v12, 0",
            "vmv.v.v v0, v5", "vmerge.vvm v12, v12, v8, v0",
            "vmv.v.v v0, v6", "vmerge.vvm v12, v12, v9, v0",
            "vmv.v.v v0, v7", "vmerge.vvm v12, v12, v10, v0",
            "vse8.v v12, ({dhi})",

            // top byte (4-byte characters only)
            "vand.vi v8, v1, 7", "vsll.vi v8, v8, 2",
            "vand.vx v9, v2, {c30}", "vsrl.vi v9, v9, 4", "vor.vv v8, v8, v9",
            "vmv.v.i v12, 0",
            "vmv.v.v v0, v7", "vmerge.vvm v12, v12, v8, v0",
            "vse8.v v12, ({dtop})",
            "vse8.v v1, ({db0})",

            // ---- phase 2: SEW=32, widen and compact ----
            "mv {dd}, {d}",
            "mv {cnt}, zero",
            "2:",
            "beq {rem}, zero, 3f",
            "vsetvli {vl}, {rem}, e32, m1, ta, ma",
            "vle8.v v13, ({dlo})", "vzext.vf4 v14, v13",
            "vle8.v v15, ({dhi})", "vzext.vf4 v16, v15",
            "vle8.v v17, ({dtop})", "vzext.vf4 v18, v17",
            "vsll.vi v16, v16, 8", "vsll.vi v18, v18, 16",
            "vor.vv v14, v14, v16", "vor.vv v14, v14, v18",
            "vle8.v v19, ({db0})", "vzext.vf4 v20, v19",
            "vmsgeu.vx v0, v20, {c80}",
            "vmsltu.vx v1, v20, {cc0}",
            "vmand.mm v0, v0, v1", "vmnot.m v0, v0",
            "vid.v v21", "vmsltu.vx v1, v21, {rem}", "vmand.mm v0, v0, v1",
            "vpopc.m {c}, v0",
            "vcompress.vm v22, v14, v0",
            "vse32.v v22, ({dd})",
            "add {cnt}, {cnt}, {c}",
            "add {dlo}, {dlo}, {vl}", "add {dhi}, {dhi}, {vl}",
            "add {dtop}, {dtop}, {vl}", "add {db0}, {db0}, {vl}",
            "slli {t1}, {c}, 2", "add {dd}, {dd}, {t1}",
            "sub {rem}, {rem}, {vl}",
            "j 2b",
            "3:",

            adv = in(reg) adv,
            c3f = in(reg) 0x3fusize, c30 = in(reg) 0x30usize, c3c = in(reg) 0x3cusize,
            c80 = in(reg) 0x80usize, cc0 = in(reg) 0xc0usize,
            ce0 = in(reg) 0xe0usize, cf0 = in(reg) 0xf0usize,
            p = in(reg) p, p1 = in(reg) p.add(1),
            p2 = in(reg) p.add(2), p3 = in(reg) p.add(3),
            d = in(reg) dst.add(o),
            dlo = inout(reg) lo_ptr,
            dhi = inout(reg) hi_ptr,
            dtop = inout(reg) top_ptr,
            db0 = inout(reg) b0_ptr,
            rem = out(reg) rem,
            dd = out(reg) dd,
            cnt = out(reg) cnt,
            vl = out(reg) vl, vl0 = out(reg) _,
            c = out(reg) c,
            t1 = out(reg) _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
            out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
            out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
            out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _,
            options(nostack),
        );
    }
    o + cnt
}

#[target_feature(enable = "v")]
unsafe fn rvv_impl(src: &[u8], dst: *mut u32, mode: u8) -> Result<usize, Utf32Error> {
    let len = src.len();
    let p = src.as_ptr();
    let vlmax = vlen_bytes();
    let (mut i, mut o) = (0usize, 0usize);
    let mut streak = 0u32;
    while len - i >= vlmax + 3 {
        let n = vlmax.min(len - i - 3);
        let pi = p.add(i);
        let adv = if *pi.add(n - 1) >= 0xC0 {
            n - 1
        } else if *pi.add(n - 2) >= 0xE0 {
            n - 2
        } else if *pi.add(n - 3) >= 0xF0 {
            n - 3
        } else {
            n
        };
        if non_ascii_at(pi, adv) < 0 {
            ascii_widen(pi, dst.add(o), adv);
            i += adv;
            o += adv;
            streak = 0;
            continue;
        }
        let first_bad = validate_block(pi, adv, n);
        if first_bad >= 0 {
            let mut cut = 0;
            for j in (0..first_bad as usize).rev() {
                if *pi.add(j) >= 0xC0 {
                    cut = j;
                    break;
                }
            }
            o = decode_block(pi, dst, o, cut);
            i += cut;
            let stop = (i + (ERR_STRETCH << streak)).min(len);
            streak = (streak + 1).min(ERR_BACKOFF_MAX);
            (i, o) = scalar::step(scalar::run(src, i, i + 1, dst, o), src, dst, mode)?;
            while i < stop {
                (i, o) = scalar::step(scalar::run(src, i, stop, dst, o), src, dst, mode)?;
            }
            continue;
        }
        o = decode_block(pi, dst, o, adv);
        i += adv;
        streak = 0;
    }
    scalar::finish(src, i, dst, o, mode)
}

pub(crate) unsafe fn convert_rvv(src: &[u8], dst: *mut u32, mode: u8) -> Result<usize, Utf32Error> {
    rvv_impl(src, dst, mode)
}

#[cfg(feature = "alloc")]
#[target_feature(enable = "v")]
unsafe fn ascii_len_rvv_impl(src: &[u8]) -> usize {
    let vlmax = vlen_bytes();
    let mut i = 0;
    while src.len() - i >= vlmax {
        let first = unsafe { non_ascii_at(src.as_ptr().add(i), vlmax) };
        if first >= 0 {
            return i + first as usize;
        }
        i += vlmax;
    }
    i + scalar::ascii_len(&src[i..])
}
#[cfg(feature = "alloc")]
pub(crate) unsafe fn ascii_len_rvv(src: &[u8]) -> usize {
    ascii_len_rvv_impl(src)
}
