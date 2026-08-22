---
title: Proof Cache
weight: 11
---

Verifying a file you have just edited re-proves every property in it, including
the ones your edit could not have affected. The proof cache stores the claims
ESBMC has already discharged and skips the solver when it meets one again, so a
re-run only pays for what actually changed.

## Getting started

Point `--vcc-cache` at a directory. It is created if it does not exist, and the
run needs `--multi-property`, which is what splits verification into per-claim
proofs:

```bash
esbmc example.c --multi-property --vcc-cache .esbmc-cache
```

Nothing else changes: the same properties are checked and the same verdict is
reported. The only new output is a line saying what the run reused.

### A worked example

```c
#include <assert.h>

int scale(int v)
{
  int r = v * 2;
  assert(r >= v || v < 0);
  return r;
}

int clamp(int v, int hi)
{
  if (v > hi)
    v = hi;
  assert(v <= hi);
  return v;
}

int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x > 0 && x < 100);

  int a[8];
  a[x % 8] = scale(x % 10);
  assert(a[x % 8] >= 0);

  return clamp(x, 50);
}
```

The first run has an empty cache and solves everything:

```
$ esbmc example.c --multi-property --vcc-cache .esbmc-cache
VCC cache: 2 claim(s) reused, 5 solved
** 0 of 7 properties failed, 7 passed
VERIFICATION SUCCESSFUL
```

Seven properties, five distinct proofs — the two reused on this first run are
claims whose constraints are identical to another claim in the same file, which
the cache also collapses.

Now change `clamp`'s assertion to `assert(v < hi + 1);` and run again. Only
that claim is re-proved; the array bounds checks and `scale`'s assertion come
straight from the cache:

```
$ esbmc example.c --multi-property --vcc-cache .esbmc-cache
Solving claim 'assertion v < hi + 1 at file example.c line 14 column 3 function clamp'
VCC cache: 6 claim(s) reused, 1 solved
VERIFICATION SUCCESSFUL
```

### Using it in CI

The cache pays off most across CI runs, where the same file is re-verified after
a small change. Persist the directory between jobs — with GitHub Actions:

```yaml
- uses: actions/cache@v4
  with:
    path: .esbmc-cache
    key: esbmc-cache-${{ github.sha }}
    restore-keys: esbmc-cache-

- run: esbmc src/example.c --multi-property --vcc-cache .esbmc-cache
```

`restore-keys` matters: it lets a job start from the previous commit's cache
rather than an empty one. The source path is deliberately not part of a claim's
identity, so a cache restored into a different workspace directory still hits.

Add the directory to `.gitignore` — it is build output, not source.

### What invalidates what

An edit invalidates a claim when it changes the constraints that claim depends
on, so the blast radius follows the code, not the line numbers:

- Changing an expression inside one assertion re-proves that claim alone.
- Changing a branch condition reaches further, because the guard it produces is
  part of what other claims downstream depend on.
- Adding or removing a function, or anything that changes the program's set of
  objects, can invalidate claims that touch memory even in untouched functions.

Measured over 671 regression files, inserting a statement into one function
left **84.9%** of a file's claims reusable, and a re-run with no edit at all
reused **93.6%** — the shortfall being claims the solver refuted, which are
never stored. Every verdict was unchanged.

## What a claim is keyed on

A claim is identified by the *sliced SSA cone* that ESBMC solves for it — the
minimal set of constraints that claim depends on, after slicing — combined with
everything else its verdict is contingent on: the ESBMC build, every option in
effect, and the data model. Two claims share an entry only when all of that
agrees.

An entry is an empty file whose name *is* the key: 32 hex digits of cone digest
prefixed by 16 of context digest. Nothing else is stored, so the digests are all
that a hit is decided on — the cone itself is never re-compared. The cone half
is 128 bits wide for that reason; the context half is 64, which is the narrower
of the two and the one to widen first if this ever needs strengthening.

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

## Unwinding strategies

`--k-induction` and `--incremental-bmc` are cached like any other run. Both
raise the unwinding bound as they go and, for k-induction, move between the
base case, the forward condition and the inductive step. Each of those is set
on the option set before the run, and the option set is part of a claim's key,
so a proof from one phase or bound is never reused for another.

## When the cache is inactive

Naming a cache directory is ignored, and every claim solved normally, under:

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
its cone under each normalisation, its size, the verdict and its location:

```bash
esbmc example.c --multi-property --vcc-fingerprint-dump fp.tsv
esbmc example.c --multi-property --vcc-fingerprint-dump -   # to the log
```

Comparing the dumps from two runs shows which claims changed identity.

## Cost

Keying a claim means hashing its sliced cone, which is work a run without the
cache does not do, and it is paid per claim on every run — a miss costs it, and
so does a hit. The median claim takes roughly 0.2 ms to key, with a long tail
on claims whose cones are large.

What that buys back is capped by the solver's share of the run, so the quantity
that decides whether the cache pays is **solver time per claim**, not program
size. Few expensive claims win: on a module spending 97% of its time in the
solver, a warm run drops from 6.7s to 0.3s. Many cheap claims lose — every one
is keyed, and none was costing anything to solve.

ESBMC's own regression suite is the second kind. Its median file spends under
1% of its run in the solver, and only 42 of 670 spend more than half, so the
suite is a poor advertisement for the cache and a good illustration of its
limit.

That makes it the wrong tool for one-shot verification -- a competition run, or
a CI job that verifies each file exactly once with no persistent directory. It
is the right tool for the edit-and-recheck loop, where the same file is
verified repeatedly.
