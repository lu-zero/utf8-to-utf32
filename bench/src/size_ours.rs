fn main() {
    let s = std::env::args().nth(1).unwrap_or_default();
    let mut out = Vec::new();
    utf8_to_utf32::utf8_to_utf32_lossy_into(s.as_bytes(), &mut out, true);
    println!("{}", out.len());
}
