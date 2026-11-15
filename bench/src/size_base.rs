fn main() {
    let s = std::env::args().nth(1).unwrap_or_default();
    let out: Vec<u32> = String::from_utf8_lossy(s.as_bytes())
        .chars()
        .map(|c| c as u32)
        .collect();
    println!("{}", out.len());
}
