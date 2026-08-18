---
title: Proof Cache
weight: 11
---

Verifying a file you have just edited re-proves every property in it, including
the ones your edit could not have affected. The proof cache stores the claims
ESBMC has already discharged and skips the solver when it meets one again, so a
re-run only pays for what actually changed.

It is off by default and enabled by pointing `--vcc-cache` at a directory:

```bash
# First run: solves every claim and records the ones it proves
esbmc example.c --multi-property --vcc-cache .esbmc-cache

# After editing one function: only the affected claims reach the solver
esbmc example.c --multi-property --vcc-cache .esbmc-cache
```

Each run reports what it reused:

```
VCC cache: 44 claim(s) reused, 8 solved
```

The cache requires `--multi-property`, which is what splits a run into
per-claim proofs.

## What a claim is keyed on

A claim is identified by the *sliced SSA cone* that ESBMC solves for it — the
minimal set of constraints that claim depends on, after slicing — combined with
everything else its verdict is contingent on: the ESBMC build, every option in
effect, and the data model. Two claims share an entry only when all of that
agrees.

The cone is compared in full before an entry is used, so a hash collision costs
a re-solve rather than producing a wrong answer.

Because the key is the claim's own dependencies rather than its position, an
edit invalidates only the claims that genuinely read what you changed. Editing
one function typically leaves the rest of the file cached: measured across
`regression/esbmc`, inserting a statement into one function left **90.9%** of
the file's claims (350 of 385) reusable, with every verdict unchanged.

The source path is deliberately *not* part of the key, so a cache carries over
between two checkouts of the same tree — the usual CI arrangement, where each
job runs in a different workspace directory.

## Only proofs are stored

A claim ESBMC *disproves* is never cached. Reporting a counterexample requires
the model the solver produced, and a stored verdict cannot reconstruct one, so
violated claims are re-solved on every run and their traces are always freshly
generated. A vacuous discharge is not stored either, since it reports `Unknown`
rather than a proof.

The practical consequence is that a failing file gets less benefit than a
passing one, and that the cache can never turn a failure into a success.

## Checking the cache against the solver

`--vcc-cache-verify` reads the cache but solves every claim anyway, reporting an
error if a stored proof disagrees with the solver:

```bash
esbmc example.c --multi-property --vcc-cache .esbmc-cache --vcc-cache-verify
```

This gives up the speed-up, so it is meant for validating the cache — for
example on a nightly job — rather than for routine runs.

## When the cache is inactive

Naming a cache directory is ignored, and every claim solved normally, under:

- `--k-induction`, `--forward-condition` and `--inductive-step`, where one claim
  is re-used across phases that mean different things
- `--incremental-bmc`, where the unwinding bound changes within the run
- `--ltl` and `--smt-during-symex`
- the coverage modes and `--dead-code-check`, whose probes are reachability
  questions rather than properties
- thread interleavings after the first, where a claim carries a schedule its
  cone does not name

## Invalidation and housekeeping

Every option is folded into the key, including ones that only affect
scheduling, so changing any flag starts a fresh set of entries rather than
reusing existing ones. This is deliberately conservative: an option wrongly
judged irrelevant would silently reuse a proof that no longer holds.

Entries are plain files named by their digest and are never evicted. The
directory is safe to delete at any time — the next run simply repopulates it.

## Diagnosing a cache that does not hit

`--vcc-fingerprint-dump` writes one line per solved claim giving the digest of
its cone, its size, the verdict and its location:

```bash
esbmc example.c --multi-property --vcc-fingerprint-dump fp.tsv
esbmc example.c --multi-property --vcc-fingerprint-dump -   # to the log
```

Comparing the dumps from two runs shows which claims changed identity. For the
full text behind a digest, raise the `fingerprint` log module:

```bash
esbmc example.c --multi-property --vcc-fingerprint-dump - --verbosity fingerprint:9
```
