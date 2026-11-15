//! Shared bitmask validation for SIMD UTF-8 decoders.

/// Ok((lead mask, bytes to advance incl. lookahead)) for a valid block.
/// Err(bad bits). `plan_err` turns bad bits into (lead mask, cut): only the
/// characters wholly before the one containing the first bad bit (positions < cut)
/// are known valid.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan(
    hi: u64,
    c0: u64,
    c2: u64,
    e0: u64,
    f0: u64,
    f5: u64,
    p: u32,
) -> Result<(u64, usize), u64> {
    let pm = (1u64 << p) - 1;
    let cont = hi & !c0;
    let claimed = ((c0 & pm) << 1) | ((e0 & pm) << 2) | ((f0 & pm) << 3);
    let look = 7u64 << p;
    let bad = ((cont ^ claimed) & pm) | (claimed & look & !cont) | (((c0 & !c2) | f5) & pm);
    if bad == 0 {
        return Ok((
            !cont & pm,
            p as usize + (claimed & look).count_ones() as usize,
        ));
    }
    Err(bad)
}

#[cold]
#[inline(never)]
pub(crate) fn plan_err(bad: u64, cont: u64, pm: u64) -> Result<(u64, usize), (u64, usize)> {
    let q = bad.trailing_zeros();
    let below = !cont & pm & ((1u64 << q) - 1);
    let cut = if below == 0 {
        0
    } else {
        63 - below.leading_zeros()
    };
    Err((below & ((1u64 << cut) - 1), cut as usize))
}

/// Bad second bytes after E0 / ED / F0 / F4 (overlong, surrogate, > U+10FFFF).
#[inline(always)]
pub(crate) fn ranges_bad(
    eq_e0: u64,
    eq_ed: u64,
    eq_f0: u64,
    eq_f4: u64,
    ge_a0: u64,
    ge_90: u64,
) -> u64 {
    let na = ge_a0 >> 1;
    let n9 = ge_90 >> 1;
    (eq_e0 & !na) | (eq_ed & na) | (eq_f0 & !n9) | (eq_f4 & n9)
}

/// Decode the valid prefix of a failing block with SIMD (else: whole block scalar).
pub(crate) const PREFIX: bool = false;
