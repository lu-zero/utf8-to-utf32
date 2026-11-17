# Scripts

`performance.py` writes the throughput graphs of the README and of
`docs/architecture/`. It reads the table that `utf32-bench` prints, so a new
measurement only needs the result file to be replaced:

```sh
cargo run --release -p utf32-bench --bin utf32-bench | grep '^|' > bench/results/neon.md
scripts/performance.py bench/results/neon.md docs/performance-aarch64.svg "aarch64, one machine, median of 7."
scripts/performance.py bench/results/default.md docs/performance-x86.svg "Intel Xeon with AVX-512 VBMI2, median of 7."
```

The VSX level needs nightly and the `vsx` feature; run the benchmark on a
powerpc64 machine:

```sh
cargo +nightly run --release -p utf32-bench --features utf8-to-utf32/vsx | grep '^|' > bench/results/ppc64.md
scripts/performance.py bench/results/ppc64.md docs/performance-ppc64.svg "POWER9, one machine, median of 7."
```

`performance_summary.py` builds a cross-architecture graph from the same result
files and prints its markdown table:

```sh
scripts/performance_summary.py > /tmp/opencode/performance-summary.md
```

It compares each SIMD level to scalar on the same machine, so absolute GB/s from
different machines are never compared.

The caption names the machine. It needs only Python 3. The colours are those
of the winnow-args figures: teal titles, grey axes, green for this crate
(darker for wider vectors), red for simdutf, blue-grey for `std`.
