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

/// First invalid position below `adv` in the block at `p`, or -1.
///
/// `n = adv + up to 3` is the number of loaded lanes, so a lead just before
/// `adv` can still claim a continuation beyond it. Such a claim is reported as
/// bad, which makes the caller cut the block before that lead.
#[target_feature(enable = "v")]
unsafe fn validate_block(p: *const u8, adv: usize, n: usize) -> isize {
    let first: isize;
    unsafe {
        asm!(
            "vsetvli {vl}, {n}, e8, m1, ta, ma",
            "vle8.v v1, ({p})",
            "vle8.v v2, ({p1})",
            "vle8.v v3, ({p2})",
            "vle8.v v4, ({p3})",

            // hi, c0, e0, f0, c2, f5 for b0
            "li t0, 0x80",
            "vmsgeu.vx v0, v1, t0", "vmv.v.v v20, v0",
            "li t0, 0xc0",
            "vmsgeu.vx v0, v1, t0", "vmv.v.v v21, v0",
            "li t0, 0xe0",
            "vmsgeu.vx v0, v1, t0", "vmv.v.v v22, v0",
            "li t0, 0xf0",
            "vmsgeu.vx v0, v1, t0", "vmv.v.v v23, v0",
            "li t0, 0xc2",
            "vmsgeu.vx v0, v1, t0", "vmv.v.v v24, v0",
            "li t0, 0xf5",
            "vmsgeu.vx v0, v1, t0", "vmv.v.v v25, v0",

            // previous byte, two back, three back
            "vmv.v.i v5, 0", "vslideup.vi v5, v1, 1",
            "vmv.v.i v6, 0", "vslideup.vi v6, v1, 2",
            "vmv.v.i v7, 0", "vslideup.vi v7, v1, 3",
            "li t0, 0xc0",
            "vmsgeu.vx v0, v5, t0", "vmv.v.v v8, v0",
            "li t0, 0xe0",
            "vmsgeu.vx v0, v6, t0", "vmv.v.v v9, v0",
            "li t0, 0xf0",
            "vmsgeu.vx v0, v7, t0", "vmv.v.v v10, v0",
            "vor.vv v8, v8, v9", "vor.vv v8, v8, v10",

            // cont ^ claimed
            "vmv.v.v v0, v21", "vmnot.m v0, v0", "vmv.v.v v26, v0",
            "vand.vv v26, v20, v26",
            "vxor.vv v11, v26, v8",

            // invalid lead = (c0 && !c2) || f5
            "vmv.v.v v0, v24", "vmnot.m v0, v0", "vmv.v.v v12, v0",
            "vand.vv v12, v21, v12", "vor.vv v12, v12, v25",
            "vor.vv v11, v11, v12",

            // range checks after E0 / ED / F0 / F4
            "li t0, 0xe0",
            "vmseq.vx v0, v1, t0", "vmv.v.v v13, v0",
            "li t0, 0xa0",
            "vmsgeu.vx v0, v2, t0", "vmv.v.v v14, v0",
            "vmv.v.v v0, v14", "vmnot.m v0, v0", "vmv.v.v v15, v0",
            "vand.vv v13, v13, v15",
            "li t0, 0xed",
            "vmseq.vx v0, v1, t0", "vmv.v.v v15, v0",
            "vand.vv v15, v15, v14", "vor.vv v13, v13, v15",
            "li t0, 0xf0",
            "vmseq.vx v0, v1, t0", "vmv.v.v v15, v0",
            "li t0, 0x90",
            "vmsgeu.vx v0, v2, t0", "vmv.v.v v16, v0",
            "vmv.v.v v0, v16", "vmnot.m v0, v0", "vmv.v.v v17, v0",
            "vand.vv v15, v15, v17", "vor.vv v13, v13, v15",
            "li t0, 0xf4",
            "vmseq.vx v0, v1, t0", "vmv.v.v v15, v0",
            "vand.vv v15, v15, v16", "vor.vv v13, v13, v15",
            "vor.vv v11, v11, v13",

            // base checks only below adv; claims beyond adv also fail
            "vid.v v18", "vmsltu.vx v0, v18, {adv}", "vmv.v.v v19, v0",
            "vand.vv v11, v11, v19",
            "vmv.v.v v0, v19", "vmnot.m v0, v0", "vmv.v.v v23, v0",
            "vand.vv v23, v8, v23",
            "vmv.v.v v0, v26", "vmnot.m v0, v0", "vmv.v.v v24, v0",
            "vand.vv v23, v23, v24",
            "vor.vv v11, v11, v23",

            "vfirst.m {first}, v11",

            vl = out(reg) _,
            n = in(reg) n,
            p = in(reg) p,
            p1 = in(reg) p.add(1),
            p2 = in(reg) p.add(2),
            p3 = in(reg) p.add(3),
            adv = in(reg) adv,
            first = out(reg) first,
            out("t0") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
            out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
            out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
            out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _,
            out("v25") _, out("v26") _,
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

/// Decode `adv` bytes into consecutive u32 lanes. Each SEW=32 iteration covers
/// `vl` byte positions; all arithmetic and the compaction stay inside asm.
#[target_feature(enable = "v")]
unsafe fn decode_block(p: *const u8, dst: *mut u32, o: usize, adv: usize) -> usize {
    let cnt: usize;
    unsafe {
        asm!(
            "mv {rem}, {adv}",
            "mv {pp}, {p}",
            "mv {dd}, {d}",
            "mv {cnt}, zero",
            "3:",
            "beq {rem}, zero, 4f",
            "vsetvli {vl}, {rem}, e32, m1, ta, ma",
            "addi t0, {pp}, 1",
            "addi t1, {pp}, 2",
            "addi t2, {pp}, 3",

            "vle8.v v1, ({pp})",  "vzext.vf4 v2, v1",
            "vle8.v v3, (t0)",    "vzext.vf4 v4, v3",
            "vle8.v v5, (t1)",    "vzext.vf4 v6, v5",
            "vle8.v v7, (t2)",    "vzext.vf4 v8, v7",

            "li t4, 0xc0",
            "vmsgeu.vx v0, v2, t4", "vmv.v.v v13, v0",
            "li t4, 0xe0",
            "vmsgeu.vx v0, v2, t4", "vmv.v.v v14, v0",
            "li t4, 0xf0",
            "vmsgeu.vx v0, v2, t4", "vmv.v.v v15, v0",

            "li t4, 0x7f",
            "vand.vx v9, v2, t4",
            "li t4, 0x1f",
            "vand.vx v10, v2, t4",
            "vsll.vi v10, v10, 6",
            "li t4, 0x3f",
            "vand.vx v11, v4, t4",
            "vor.vv v10, v10, v11",

            "li t4, 0x1f",
            "vand.vx v11, v2, t4",
            "vand.vi v11, v11, 15",
            "vsll.vi v11, v11, 12",
            "li t4, 0x3f",
            "vand.vx v16, v4, t4",
            "vsll.vi v16, v16, 6",
            "vor.vv v11, v11, v16",
            "vand.vx v16, v6, t4",
            "vor.vv v11, v11, v16",

            "vand.vi v12, v2, 7",
            "vsll.vi v12, v12, 18",
            "vand.vx v16, v4, t4",
            "vsll.vi v16, v16, 12",
            "vor.vv v12, v12, v16",
            "vand.vx v16, v6, t4",
            "vsll.vi v16, v16, 6",
            "vor.vv v12, v12, v16",
            "vand.vx v16, v8, t4",
            "vor.vv v12, v12, v16",

            "vmv.v.v v17, v9",
            "vmv.v.v v0, v13",
            "vmerge.vvm v17, v17, v10, v0",
            "vmv.v.v v0, v14",
            "vmerge.vvm v17, v17, v11, v0",
            "vmv.v.v v0, v15",
            "vmerge.vvm v17, v17, v12, v0",

            // lead = !(b0 in 0x80..=0xbf), masked to this chunk's rem
            "li t4, 0x80",
            "vmsgeu.vx v0, v2, t4",
            "vmv.v.v v18, v0",
            "li t4, 0xc0",
            "vmsltu.vx v0, v2, t4",
            "vand.vv v18, v18, v0",
            "vmv.v.v v0, v18",
            "vmnot.m v0, v0",
            "vmv.v.v v18, v0",
            "vid.v v19", "vmsltu.vx v0, v19, {rem}", "vand.vv v18, v18, v0",
            "vmv.v.v v0, v18",

            "vpopc.m {c}, v18",
            "vcompress.vm v20, v17, v0",
            "vse32.v v20, ({dd})",

            "add {cnt}, {cnt}, {c}",
            "add {pp}, {pp}, {vl}",
            "slli t3, {c}, 2",
            "add {dd}, {dd}, t3",
            "sub {rem}, {rem}, {vl}",
            "j 3b",
            "4:",

            adv = in(reg) adv,
            p = in(reg) p,
            d = in(reg) dst.add(o),
            rem = out(reg) _,
            pp = out(reg) _,
            dd = out(reg) _,
            cnt = out(reg) cnt,
            vl = out(reg) _,
            c = out(reg) _,
            out("t0") _, out("t1") _, out("t2") _, out("t3") _, out("t4") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
            out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
            out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
            out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _,
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
