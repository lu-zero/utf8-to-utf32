fn main() {
    let s = std::env::args().nth(1).unwrap_or_default();
    let mut out: Vec<u32> = Vec::with_capacity(s.len());
    unsafe {
        let r = simdutf::convert_utf8_to_utf32_with_errors(s.as_ptr(), s.len(), out.as_mut_ptr());
        out.set_len(r.count);
    }
    println!("{}", out.len());
}
