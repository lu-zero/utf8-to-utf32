use std::hint::black_box;
use std::mem::MaybeUninit;
use std::time::Instant;
use utf8_to_utf32 as utf32;
use utf8_to_utf32::{Decoder, Level, Piece};

type F = Box<dyn FnMut(&[u8]) -> usize>;

fn sd(s: &[u8], out: &mut Vec<u32>) -> usize {
    out.clear();
    out.reserve(s.len());
    unsafe {
        let r = simdutf::convert_utf8_to_utf32_with_errors(s.as_ptr(), s.len(), out.as_mut_ptr());
        out.set_len(r.count);
    }
    out.len()
}
fn stdc(s: &[u8], out: &mut Vec<u32>) -> usize {
    out.clear();
    out.extend(std::str::from_utf8(s).unwrap().chars().map(|c| c as u32));
    out.len()
}
fn s8(s: &[u8], out: &mut Vec<u32>) -> usize {
    out.clear();
    out.extend(
        simdutf8::basic::from_utf8(s)
            .unwrap()
            .chars()
            .map(|c| c as u32),
    );
    out.len()
}
fn chk(s: &[u8], out: &mut Vec<u32>) -> usize {
    out.clear();
    out.reserve(s.len());
    for c in s.utf8_chunks() {
        out.extend(c.valid().chars().map(|c| c as u32));
        let inv = c.invalid();
        if inv.len() == 1 && (0x80..=0x9f).contains(&inv[0]) {
            out.push(inv[0] as u32);
        } else if !inv.is_empty() {
            out.push(0xFFFD);
        }
    }
    out.len()
}

fn median_gbs(data: &[u8], f: &mut F) -> f64 {
    for _ in 0..10 {
        black_box(f(data));
    }
    let mut v: Vec<f64> = (0..7)
        .map(|_| {
            let n = 60;
            let t = Instant::now();
            for _ in 0..n {
                black_box(f(black_box(data)));
            }
            (data.len() * n) as f64 / t.elapsed().as_secs_f64() / 1e9
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn rows(clean: bool) -> Vec<(String, F)> {
    let mut r: Vec<(String, F)> = Vec::new();
    macro_rules! vecrow {
        ($name:expr, $f:expr) => {{
            let mut out = Vec::new();
            r.push(($name.to_string(), Box::new(move |s: &[u8]| $f(s, &mut out))));
        }};
    }
    if clean {
        vecrow!("simdutf C++ (strict)", sd);
        vecrow!("simdutf8 + chars", s8);
        vecrow!("std from_utf8 + chars", stdc);
    }
    vecrow!("utf8_chunks + chars (C1)", chk);
    for l in Level::supported() {
        if clean {
            let mut dst: Vec<MaybeUninit<u32>> = vec![MaybeUninit::uninit(); 4 << 20];
            r.push((
                format!("ours {} slice fn", l.name()),
                Box::new(move |s: &[u8]| {
                    utf32::utf8_to_utf32_with(l, s, &mut dst[..s.len()]).unwrap()
                }),
            ));
        }
        let mut out = Vec::new();
        r.push((
            format!("ours {} lossy_into (C1)", l.name()),
            Box::new(move |s: &[u8]| {
                out.clear();
                utf32::utf8_to_utf32_lossy_into_with(l, s, &mut out, true);
                out.len()
            }),
        ));
        let mut dec = Decoder::with_level(l);
        r.push((
            format!("ours {} Decoder pieces", l.name()),
            Box::new(move |s: &[u8]| {
                let mut n = 0;
                let mut p = dec.pieces(s);
                while let Some(x) = p.next_piece() {
                    n += match x {
                        Piece::Ascii(a) => a.len(),
                        Piece::Wide(w) => w.len(),
                        Piece::Invalid(_) | Piece::Incomplete(_) => 1,
                    };
                }
                n
            }),
        ));
    }
    r
}

fn main() {
    let ascii = "drwxr-xr-x  2 user user 4096 Nov 14 18:00 src  ".repeat(25000);
    let latin = "café naïve résumé über straße ".repeat(30000);
    let cjk = "中文字符测试日本語のテキスト한국어 ".repeat(20000);
    let mixed = "│ build ✓ ok ── 42 ms → done ".repeat(30000);
    let emoji = "😀 ok 🎉 done 🚀 ".repeat(30000);
    let mut dirty = Vec::new();
    for _ in 0..30000 {
        dirty.extend_from_slice("│ build ✓ ok ".as_bytes());
        dirty.push(0x9b);
        dirty.extend_from_slice(b"31m done ");
        dirty.push(0xff);
    }
    let inputs: [(&str, &[u8]); 6] = [
        ("ascii", ascii.as_bytes()),
        ("latin", latin.as_bytes()),
        ("cjk", cjk.as_bytes()),
        ("box/mixed", mixed.as_bytes()),
        ("emoji", emoji.as_bytes()),
        ("dirty", &dirty),
    ];
    println!("active level (dispatch): {:?}", utf32::active_level());
    let names: Vec<String> = rows(true).into_iter().map(|(n, _)| n).collect();
    let mut table: Vec<Vec<Option<f64>>> = vec![vec![None; inputs.len()]; names.len()];
    for (ci, (n, d)) in inputs.iter().enumerate() {
        eprintln!("{n}: {} KB", d.len() / 1024);
        for (name, mut f) in rows(*n != "dirty") {
            let ri = names.iter().position(|x| *x == name).unwrap();
            table[ri][ci] = Some(median_gbs(d, &mut f));
        }
    }
    print!("| {:<28} |", "GB/s (median of 7)");
    for (n, _) in &inputs {
        print!(" {n:>9} |");
    }
    println!();
    for (ri, name) in names.iter().enumerate() {
        print!("| {name:<28} |");
        for c in &table[ri] {
            match c {
                Some(v) => print!(" {v:>9.2} |"),
                None => print!(" {:>9} |", "-"),
            }
        }
        println!();
    }
}
