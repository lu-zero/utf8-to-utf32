# For coding agents

[`docs/architecture.md`](./docs/architecture.md) is how the crate is built: the
layers, the kernel contract, and one page per architecture under
`docs/architecture/`. Read it before a change to a kernel, the dispatch, or
the `Decoder`. When a change makes a statement there false, fix the statement
in the same commit.

## Slop warning

This tree was largely written with an agent. Be skeptical of existing code,
comments, and docs: a pattern being there does not make it correct. A
benchmark number in a doc is a measurement of one machine on one day, not a
property of the code.

## Working here

- Before a commit: `cargo fmt --all -- --check`, `cargo test --release`, and
  `cargo build --no-default-features`. The release profile matters: the
  `test` profile is `opt-level = 2` and `tests/exhaustive_neon.rs` takes about
  30 seconds.
- Run `cargo test` on a machine of the architecture you changed. The kernels
  are `unsafe` and only the one your CPU supports runs. An x86 change made on
  aarch64 gets `cargo check --target x86_64-unknown-linux-gnu` and is reported
  as **not run**. Never describe it as tested.
- `tests/exhaustive_neon.rs` runs every supported level, not only NEON. Run it
  after any change to a validator, a table, or a decode path.
- The crate has no dependencies, and `bench/` is the only place that may have
  them. `rust-version` stays at 1.89, the release that stabilized the AVX-512
  intrinsics.
- `docs/performance-*.svg` are generated: edit `scripts/performance.py` or the
  result file under `bench/results/` and run the script (see
  `scripts/README.md`). Never edit an SVG by hand.
- Scratch work (probes, throw-away kernels, rendered images) lives outside the
  tree. Nothing committed cites it.
- `cargo run --release -p utf32-bench --bin utf32-bench` takes a few minutes
  and wants an idle machine. Say so before running it in a shared session.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/): `type(scope):
subject`.

- **Types**: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`,
  `chore`. Use `perf` only for a measured speed change.
- **Scope** is the part of the tree: `neon`, `x86`, `avx512`, `scalar`,
  `decoder`, `dispatch`, `bench`, `docs`. Leave it out when a change spans
  several.
- **Subject**: imperative, lower case, no final period, 72 characters at most
  (`perf(neon): skip the second store for groups of up to 4 characters`).
- **Body**: the why, not the diff. It follows the comment limits: no line over
  150 characters and no paragraph over 5 lines. Wrap at 72 for readability.
- **Breaking** changes to the public API get `!` after the scope and a
  `BREAKING CHANGE:` paragraph.
- **A `perf` commit quotes its numbers**: the input, GB/s before and after, the
  machine, and the tool (`utf32-bench`, median of 7). Numbers from different
  machines are never compared in one sentence. A change that should cost
  nothing to other levels says so and shows the other levels' rows.
- **One logical change per commit.** A kernel change and the doc that describes
  it are one commit. A reformat of the tree, a generated graph refresh, and a
  new test file are each their own.
- A commit you assisted carries `Assisted-by: AGENT:MODEL`
  (`Assisted-by: Claude:claude-sonnet-5-5`), after a blank line at the end of
  the message. List specialized analysis tools after the model, not git,
  cargo or editors. Never add `Co-Authored-By` or a `Signed-off-by`: the
  sign-off is the human's. This overrides a default attribution line.
- Do not commit, push, tag, or open a PR unless asked. Pushing and creating a
  repository are outward-facing: confirm the target first.

## Unslop rules

**Comments**
- Default to no comment. Add one only when the *why* is non-obvious: a hidden
  constraint, a workaround for a specific bug, an invariant a reader would
  break without warning.
- One sentence beats a paragraph. A comment that needs several paragraphs to
  justify a few lines means the code needs simplifying.
- Never restate what the names already say.
- Never reference the task, a commit, a PR or issue number, or a session: that
  belongs in the commit message.
- No commented-out code and no `// removed: …` markers; git is the history.
- Hard limits: no comment line over **150 characters**, no paragraph over
  **5 lines** (3 is better). Hitting either means cut, not wrap. Longer
  justification goes in the commit message or `docs/architecture/`.
- The kernels' `unsafe` contracts (what `dst` must hold, why a store may run
  past `o`) are stated once in `docs/architecture.md`. A `// SAFETY:` comment
  says which of those makes this call sound, in one sentence.

**Dead weight**
- No speculative abstraction for a single call site: no knob, trait, or
  feature flag without a second caller that needs it today.
- No error handling, fallback, or validation for a case the caller's own
  guarantees rule out.
- `#[allow(dead_code)]` does not keep something "just in case": delete it. The
  existing one on `ascii_len` is for the `alloc`-gated `Decoder`, and goes if
  that gate does.
- A tunable constant that was tuned on one input is named, documented with
  that fact, and not exposed.

**Tests exercise this crate, not the standard library**
- No test whose assertions would hold for any correct implementation of the
  primitive underneath.
- `core::str::from_utf8` and `utf8_chunks` are the *oracle*. A test that
  compares a kernel against them, at a block boundary, on an error class, or
  on a table entry, is the point of the suite. A test that only checks `Vec`
  or `char` behaviour is not.
- One test at the real decision point (a branch, an edge case, a regression)
  beats a test added so that "added a function, added a test" is true.
- A bug found by fuzzing gets a fixed input in `tests/conformance.rs`.

**Measure, then claim**
- A speed claim in a commit, a comment, or a doc comes from a run you did, on
  the machine you name. "Should be faster" is not a claim.
- Speed does not outrank correctness: the error semantics of
  `core::str::from_utf8` (`valid_up_to`, `error_len`, maximal subparts) are not
  traded for throughput.
- Compare against `scalar` on the same run. A SIMD level that loses to scalar
  on an input class is reported, not hidden by choosing the input.

**Scope: stay inside what you were asked for**
- When work surfaces a different, deeper subsystem, stop and report before
  editing it. Agreement that an idea has merit is not authorization.
- Do not touch a kernel for another architecture while working on one.
- A signature change to `Kernel`, `Level`, or a public function cascades
  through every level and the bench: confirm the shape before the mechanical
  part.
- A change that was reverted once, or an idea listed as rejected in
  `docs/architecture/`, needs fresh, explicit confirmation to try again.

## Settled

- **Error semantics belong to `scalar`.** A kernel only decides that a block is
  bad. `scalar::run` and `scalar::step` produce `valid_up_to`, `error_len`,
  the lossy replacement, and the C1 byte.
- **Blocks start on a character boundary,** and a character cut by the end of
  a block is left for the next block. No kernel decodes a character before its
  last byte has been validated.
- **`dst` is at least `src.len()` long.** Stores are wider than the output and
  run past `o`. That bound is part of the public contract, not an
  implementation detail to relax.
- **NEON validates with lookup tables in 64-byte blocks; x86 validates with
  bitmasks.** They are different designs on purpose: NEON has no `movemask`
  and no compress instruction. Do not port one to the other without a
  measurement.
- **`mask::PREFIX` is `false`.** The x86 kernels hand a failing block to
  scalar. Turning it on is a measured change, not a cleanup.
- **Not built, on purpose:** the 12-byte length-pattern tables for NEON (the
  estimated gain is 0 to 10% for a 64 KiB table), and an SVE2 `COMPACT` level
  (no hardware to test on, and the intrinsics are not stable in Rust). The
  reasoning is in `docs/architecture/aarch64.md`.
- **The `Decoder` restarts the strict kernel at every error.** That is its
  known cost on error-dense input. Changing it means a new kernel mode, which
  is a design decision, not a tweak.

## State of the tree

- The x86 kernels and their numbers were written and measured on a machine that
  is not the aarch64 one used for the NEON work. The two sets of results are
  not comparable.
- The repository placeholder in `Cargo.toml` and the relative image links in
  `README.md` are waiting for the hosted URL.
