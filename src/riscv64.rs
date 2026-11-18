//! riscv64 RVV kernel for 256-bit vector registers.
//!
//! Rust's `core::arch` has no RVV intrinsics yet, so the vector work is inline
//! assembly. The hot loop is one `global_asm!` function, `rvv_blocks`, which
//! owns the block loop and returns to Rust on an error or at the end of the
//! input. That lets the three lookup tables load once at entry instead of
//! once per block, and keeps constants in registers across the loop. Error
//! semantics stay in `scalar`: Rust computes the cut, decodes the valid prefix
//! with `decode_block`, then resolves the error and re-enters.
//!
//! `vl` comes from `vsetvli(min(remaining - 3, vlmax))`. Validation loads `n`
//! lanes rather than the shortened `adv`, so a lead just before the cut can
//! claim a byte past it. Decode computes candidate code points at SEW=8, then
//! widens and compacts them in two unrolled SEW=32 groups of 16 lanes each.
//! The 32-byte stack buffers and the two fixed groups require VLEN=256.

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

/// The decode path uses two fixed 16-lane groups, so it requires VLEN=256.
pub(crate) fn supported() -> bool {
    #[cfg(not(feature = "std"))]
    if !cfg!(target_feature = "v") {
        return false;
    }
    #[cfg(feature = "std")]
    if !std::arch::is_riscv_feature_detected!("v") {
        return false;
    }
    unsafe { vlen_bytes() == 32 }
}

/// Offset of the first non-ASCII byte in the first `n` bytes at `p`, or -1.
#[target_feature(enable = "v")]
unsafe fn non_ascii_at(p: *const u8, n: usize) -> isize {
    let first: isize;
    unsafe {
        asm!(
            "vsetvli {vl}, {n}, e8, m1, ta, ma",
            "vle8.v v1, ({p})",
            "li t0, 0x7f",
            "vmsgtu.vx v0, v1, t0",
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

static GB1_HIGH: [u8; 32] = [
    2, 2, 2, 2, 2, 2, 2, 2, 128, 128, 128, 128, 33, 1, 21, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
];
static GB1_LOW: [u8; 32] = [
    231, 163, 131, 131, 139, 203, 203, 203, 203, 203, 203, 203, 203, 219, 203, 203, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
static GB2_HIGH: [u8; 32] = [
    1, 1, 1, 1, 1, 1, 1, 1, 230, 174, 186, 186, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0,
];

/// First invalid position below `adv`, or -1. Keiser-Lemire lookup validator:
/// three `vrgather.vv` nibble lookups, then all boolean logic in mask registers.
#[target_feature(enable = "v")]
unsafe fn decode_block(p: *const u8, dst: *mut u32, o: usize, adv: usize) -> usize {
    let mut lo = [0u8; 32];
    let mut hi = [0u8; 32];
    let mut top = [0u8; 32];
    let mut b0buf = [0u8; 32];
    let lo_ptr = lo.as_mut_ptr();
    let hi_ptr = hi.as_mut_ptr();
    let top_ptr = top.as_mut_ptr();
    let b0_ptr = b0buf.as_mut_ptr();
    let mut cnt = 0usize;
    unsafe {
        asm!(
            // ---- phase 1: SEW=8, all adv byte positions at once ----
            "mv {rem}, {adv}",
            "vsetvli {vl0}, {rem}, e8, m1, ta, ma",
            "vle8.v v1, ({p})", "vle8.v v2, ({p1})",
            "vle8.v v3, ({p2})", "vle8.v v4, ({p3})",

            "vmsgtu.vx v5, v1, {cbf}",
            "vmsgtu.vx v6, v1, {cdf}",
            "vmsgtu.vx v7, v1, {cef}",

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

            // ---- phase 2: SEW=32, m2 (16 lanes), widen and compact ----
            "mv {dd}, {d}",
            "mv {cnt}, zero",
            "2:",
            "beq {rem}, zero, 3f",
            "vsetvli {vl}, {rem}, e32, m2, ta, ma",
            "vle8.v v2, ({dlo})", "vzext.vf4 v6, v2",
            "vle8.v v3, ({dhi})", "vzext.vf4 v8, v3",
            "vle8.v v4, ({dtop})", "vzext.vf4 v10, v4",
            "vsll.vi v8, v8, 8", "vsll.vi v10, v10, 16",
            "vor.vv v6, v6, v8", "vor.vv v6, v6, v10",
            "vle8.v v5, ({db0})", "vzext.vf4 v12, v5",
            "vmsgtu.vx v0, v12, {c7f}",
            "vmsltu.vx v1, v12, {cc0}",
            "vmand.mm v0, v0, v1", "vmnot.m v0, v0",
            "vid.v v14", "vmsltu.vx v1, v14, {rem}", "vmand.mm v0, v0, v1",
            "vpopc.m {c}, v0",
            "vcompress.vm v18, v6, v0",
            "vse32.v v18, ({dd})",
            "add {cnt}, {cnt}, {c}",
            "add {dlo}, {dlo}, {vl}", "add {dhi}, {dhi}, {vl}",
            "add {dtop}, {dtop}, {vl}", "add {db0}, {db0}, {vl}",
            "slli {t1}, {c}, 2", "add {dd}, {dd}, {t1}",
            "sub {rem}, {rem}, {vl}",
            "j 2b",
            "3:",

            adv = in(reg) adv,
            c3f = in(reg) 0x3fusize, c30 = in(reg) 0x30usize, c3c = in(reg) 0x3cusize,
            c7f = in(reg) 0x7fusize, cc0 = in(reg) 0xc0usize,
            cbf = in(reg) 0xbfusize, cdf = in(reg) 0xdfusize, cef = in(reg) 0xefusize,
            p = in(reg) p, p1 = in(reg) p.add(1),
            p2 = in(reg) p.add(2), p3 = in(reg) p.add(3),
            d = in(reg) dst.add(o),
            dlo = inout(reg) lo_ptr => _,
            dhi = inout(reg) hi_ptr => _,
            dtop = inout(reg) top_ptr => _,
            db0 = inout(reg) b0_ptr => _,
            rem = out(reg) _,
            dd = out(reg) _,
            cnt = out(reg) cnt,
            vl = out(reg) _, vl0 = out(reg) _,
            c = out(reg) _,
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

core::arch::global_asm!(
    r#"
    .option arch, +v
    .globl rvv_blocks
    .type rvv_blocks, @function
rvv_blocks:
    addi sp, sp, -128
    ld t0, 0(a4)
    ld t1, 0(a5)
    li a6, 32
    vsetvli x0, a6, e8, m1, ta, ma
    la a6, {t1h}
    vle8.v v28, (a6)
    la a6, {t1l}
    vle8.v v29, (a6)
    la a6, {t2h}
    vle8.v v30, (a6)
    vid.v v24
Lloop:
    sub t3, a2, t0
    addi a6, a3, 3
    blt t3, a6, Ldone
    addi t3, t3, -3
    blt t3, a3, Ln_ok
    mv t3, a3
Ln_ok:
    add t2, a0, t0
    vsetvli zero, t3, e8, m1, ta, ma
    vle8.v v8, (t2)
    addi t6, t2, -1
    add t6, t6, t3
    lbu t6, 0(t6)
    li a6, 0xc0
    blt t6, a6, Ladv2
    addi t4, t3, -1
    j Ladv_done
Ladv2:
    addi t6, t2, -2
    add t6, t6, t3
    lbu t6, 0(t6)
    li a6, 0xe0
    blt t6, a6, Ladv3
    addi t4, t3, -2
    j Ladv_done
Ladv3:
    addi t6, t2, -3
    add t6, t6, t3
    lbu t6, 0(t6)
    li a6, 0xf0
    blt t6, a6, Ladv32
    addi t4, t3, -3
    j Ladv_done
Ladv32:
    mv t4, t3
Ladv_done:
    li a6, 0x7f
    vmsgtu.vx v0, v8, a6
    vfirst.m t6, v0
    bltz t6, Lascii
    vmv.v.i v10, 0
    vslideup.vi v10, v8, 1
    vmv.v.i v11, 0
    vslideup.vi v11, v8, 2
    vmv.v.i v12, 0
    vslideup.vi v12, v8, 3
    vsrl.vi v13, v10, 4
    vand.vi v14, v10, 15
    vsrl.vi v15, v8, 4
    vrgather.vv v16, v28, v13
    vrgather.vv v17, v29, v14
    vrgather.vv v18, v30, v15
    vand.vv v16, v16, v17
    vand.vv v16, v16, v18
    vmsne.vi v0, v16, 0
    li a6, 0xdf
    vmsgtu.vx v2, v11, a6
    li a6, 0xef
    vmsgtu.vx v3, v12, a6
    vmor.mm v1, v2, v3
    vmxor.mm v0, v0, v1
    vmsltu.vx v5, v24, t4
    vmand.mm v0, v0, v5
    li a6, 0xbf
    vmsgtu.vx v7, v10, a6
    vmor.mm v7, v7, v1
    vmnot.m v6, v5
    vmand.mm v6, v6, v7
    li a6, 0x7f
    vmsgtu.vx v7, v8, a6
    li a6, 0xc0
    vmsltu.vx v4, v8, a6
    vmand.mm v7, v7, v4
    vmnot.m v4, v7
    vmand.mm v6, v6, v4
    vmor.mm v0, v0, v6
    vfirst.m t5, v0
    bgez t5, Lerror
    li a6, 0xDF
    vmsgtu.vx v1, v8, a6
    vmand.mm v1, v1, v5
    vfirst.m t6, v1
    bgez t6, Ldecode_full
    addi t3, t2, 1
    vle8.v v9, (t3)
    li a6, -65
    vmsgt.vx v0, v8, a6
    vmand.mm v0, v0, v5
    vcpop.m t5, v0
    vcompress.vm v16, v8, v0
    vcompress.vm v17, v9, v0
    vsetvli zero, t5, e8, m1, ta, ma
    li a6, 0x3F
    vand.vx v17, v17, a6
    li a6, 0xBF
    vmsgtu.vx v1, v16, a6
    li a6, 0x1F
    vand.vx v18, v16, a6
    vsetvli zero, t5, e16, m2, ta, ma
    vzext.vf2 v20, v16
    vzext.vf2 v12, v18
    vsll.vi v12, v12, 6
    vzext.vf2 v14, v17
    vor.vv v12, v12, v14
    vmv1r.v v0, v1
    vmerge.vvm v20, v20, v12, v0
    vsetvli zero, t5, e32, m4, ta, ma
    vzext.vf2 v8, v20
    slli a7, t1, 2
    add a7, a1, a7
    vse32.v v8, (a7)
    add t1, t1, t5
    add t0, t0, t4
    j Lloop
Ldecode_full:
    addi t3, t2, 1
    vle8.v v9, (t3)
    addi t3, t2, 2
    vle8.v v10, (t3)
    addi t3, t2, 3
    vle8.v v11, (t3)
    li a6, 0x3f
    vand.vx v16, v11, a6
    vsll.vi v17, v10, 6
    vor.vv v16, v17, v16
    vand.vx v17, v10, a6
    vsll.vi v18, v9, 6
    vor.vv v17, v18, v17
    vand.vx v18, v9, a6
    vsll.vi v19, v8, 6
    vor.vv v18, v19, v18
    vmv.v.v v20, v8
    li a6, 0xbf
    vmsgtu.vx v5, v8, a6
    li a6, 0xdf
    vmsgtu.vx v6, v8, a6
    li a6, 0xef
    vmsgtu.vx v7, v8, a6
    vmv.v.v v0, v5
    vmerge.vvm v20, v20, v18, v0
    vmv.v.v v0, v6
    vmerge.vvm v20, v20, v17, v0
    vmv.v.v v0, v7
    vmerge.vvm v20, v20, v16, v0
    vse8.v v20, (sp)
    vsrl.vi v16, v8, 2
    vand.vi v16, v16, 7
    li a6, 0x3c
    vand.vx v17, v9, a6
    vsrl.vi v17, v17, 2
    li a6, 0x0f
    vand.vx v18, v8, a6
    vsll.vi v18, v18, 4
    vor.vv v17, v17, v18
    li a6, 0x3c
    vand.vx v18, v10, a6
    vsrl.vi v18, v18, 2
    li a6, 0x0f
    vand.vx v19, v9, a6
    vsll.vi v19, v19, 4
    vor.vv v18, v18, v19
    vmv.v.i v20, 0
    vmv.v.v v0, v5
    vmerge.vvm v20, v20, v16, v0
    vmv.v.v v0, v6
    vmerge.vvm v20, v20, v17, v0
    vmv.v.v v0, v7
    vmerge.vvm v20, v20, v18, v0
    addi t3, sp, 32
    vse8.v v20, (t3)
    vand.vi v16, v8, 7
    vsll.vi v16, v16, 2
    li a6, 0x30
    vand.vx v17, v9, a6
    vsrl.vi v17, v17, 4
    vor.vv v16, v16, v17
    vmv.v.i v20, 0
    vmv.v.v v0, v7
    vmerge.vvm v20, v20, v16, v0
    addi t3, sp, 64
    vse8.v v20, (t3)
    addi t3, sp, 96
    vse8.v v8, (t3)
    li t5, 0
    li t2, 16
    slli a7, t1, 2
    add a7, a1, a7

    vsetvli x0, t2, e8, mf2, ta, ma
    addi t3, sp, 96
    vle8.v v8, (t3)
    li a6, -65
    vmsgt.vx v0, v8, a6
    vsetvli x0, t2, e32, m2, ta, ma
    vle8.v v2, (sp)
    vzext.vf4 v10, v2
    addi t3, sp, 32
    vle8.v v4, (t3)
    vzext.vf4 v12, v4
    addi t3, sp, 64
    vle8.v v6, (t3)
    vzext.vf4 v14, v6
    vsll.vi v12, v12, 8
    vsll.vi v14, v14, 16
    vor.vv v10, v10, v12
    vor.vv v10, v10, v14
    vpopc.m t6, v0
    vcompress.vm v20, v10, v0
    vse32.v v20, (a7)
    slli t3, t6, 2
    add a7, a7, t3

    add t5, t5, t6
    addi t2, t4, -16
    vsetvli x0, t2, e8, mf2, ta, ma
    addi t3, sp, 112
    vle8.v v8, (t3)
    li a6, -65
    vmsgt.vx v0, v8, a6
    vsetvli x0, t2, e32, m2, ta, ma
    addi t3, sp, 16
    vle8.v v2, (t3)
    vzext.vf4 v10, v2
    addi t3, sp, 48
    vle8.v v4, (t3)
    vzext.vf4 v12, v4
    addi t3, sp, 80
    vle8.v v6, (t3)
    vzext.vf4 v14, v6
    vsll.vi v12, v12, 8
    vsll.vi v14, v14, 16
    vor.vv v10, v10, v12
    vor.vv v10, v10, v14
    vpopc.m t6, v0
    vcompress.vm v20, v10, v0
    vse32.v v20, (a7)

    add t5, t5, t6
    add t1, t1, t5
    add t0, t0, t4
    j Lloop
Lascii:
    vsetvli zero, t4, e32, m4, ta, ma
    vle8.v v16, (t2)
    vzext.vf4 v20, v16
    slli a7, t1, 2
    add a7, a1, a7
    vse32.v v20, (a7)
    add t0, t0, t4
    add t1, t1, t4
    j Lloop
Lerror:
    sd t0, 0(a4)
    sd t1, 0(a5)
    mv a0, t5
    addi sp, sp, 128
    ret
Ldone:
    sd t0, 0(a4)
    sd t1, 0(a5)
    li a0, -1
    addi sp, sp, 128
    ret
    .size rvv_blocks, .-rvv_blocks
"#,
    t1h = sym GB1_HIGH,
    t1l = sym GB1_LOW,
    t2h = sym GB2_HIGH,
);

extern "C" {
    fn rvv_blocks(
        src: *const u8,
        dst: *mut u32,
        len: usize,
        vlmax: usize,
        ip: *mut usize,
        op: *mut usize,
    ) -> isize;
}

#[target_feature(enable = "v")]
unsafe fn rvv_impl(src: &[u8], dst: *mut u32, mode: u8) -> Result<usize, Utf32Error> {
    let p = src.as_ptr();
    let vlmax = vlen_bytes();
    let (mut i, mut o) = (0usize, 0usize);
    let mut streak = 0u32;
    loop {
        let first_bad = unsafe { rvv_blocks(p, dst, src.len(), vlmax, &mut i, &mut o) };
        if first_bad < 0 {
            return scalar::finish(src, i, dst, o, mode);
        }
        let pi = p.add(i);
        let mut cut = 0;
        for j in (0..first_bad as usize).rev() {
            if *pi.add(j) >= 0xC0 {
                cut = j;
                break;
            }
        }
        o = decode_block(pi, dst, o, cut);
        i += cut;
        let stop = (i + (ERR_STRETCH << streak)).min(src.len());
        streak = (streak + 1).min(ERR_BACKOFF_MAX);
        (i, o) = scalar::step(scalar::run(src, i, i + 1, dst, o), src, dst, mode)?;
        while i < stop {
            (i, o) = scalar::step(scalar::run(src, i, stop, dst, o), src, dst, mode)?;
        }
    }
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
