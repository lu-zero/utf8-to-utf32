# For coding agents

The project's rules for code, commits and working style are in
[`CONTRIBUTING.md`](./CONTRIBUTING.md). Follow them, including the linear
history rule: no merge commits on `main`. What follows is agent-specific.

[`docs/architecture.md`](./docs/architecture.md) is how the crate is built.
Read it before a change to a kernel, the dispatch, or the `Decoder`; a change
that makes a statement there false updates the doc in the same commit.

## Slop warning

This tree was largely written with an agent. Be skeptical of existing code,
comments, and docs: a pattern being there does not make it correct. A
benchmark number in a doc is a measurement of one machine on one day, not a
property of the code.

## Working here

- When integrating someone else's patches, keep the original author and
  `Assisted-by` trailer, check the patch applies to the current tree, and
  verify locally what this machine can run before reporting what it cannot.
- Never describe a test or measurement as yours if you did not run it. Say who
  ran it and on which machine.

## Decide before acting

- When work surfaces a different, deeper subsystem, stop and report before
  editing it. Agreement that an idea has merit is not authorization.
- Confirm the shape before a mechanical change that cascades: a signature
  change to `Kernel`, `Level`, or a public function touches every level and
  the bench.
- A change that was reverted once, or an idea listed as rejected in
  `docs/architecture/`, needs fresh, explicit confirmation to try again.
- A negative measurement is a result. Keep the tree clean and report it; do
  not keep a change that lost the A/B.
