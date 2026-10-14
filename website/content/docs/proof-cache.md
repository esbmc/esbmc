---
title: Proof Cache
weight: 11
---

Verifying a file you have just edited re-proves every property in it, including
the ones your edit could not have affected. The proof cache stores the claims
ESBMC has already discharged and skips the solver when it meets one again, so a
re-run only pays for what actually changed.

## Getting started

Point `--proof-cache` at a directory. It is created if it does not exist, and
the run needs `--multi-property`, which is what splits verification into
per-claim proofs. ESBMC refuses the flag without it rather than accepting it
and reusing nothing:

```bash
esbmc example.c --multi-property --proof-cache .esbmc-cache
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
$ esbmc example.c --multi-property --proof-cache .esbmc-cache
Proof cache: 2 claim(s) reused, 5 solved
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
$ esbmc example.c --multi-property --proof-cache .esbmc-cache
Solving claim 'assertion v < hi + 1 at file example.c line 14 column 3 function clamp'
Proof cache: 6 claim(s) reused, 1 solved
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

- run: esbmc src/example.c --multi-property --proof-cache .esbmc-cache
```

`restore-keys` matters: it lets a job start from the previous commit's cache
rather than an empty one. The source path is deliberately not part of a claim's
identity, so a cache restored into a different workspace directory still hits.

Sharing one directory between runners of different kinds is safe but pointless:
the target triple and the data model are part of the key, and with no `--32` or
`--i386-linux` style flag ESBMC takes both from the machine it is running on. A
Linux runner and a macOS runner therefore populate disjoint sets of entries
rather than reusing each other's.

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
effect, and the data model — the target triple, every type width, signedness,
endianness and the C or C++ dialect in force. Two claims share an entry only
when all of that agrees.

An entry is an empty file whose name *is* the key: 32 hex digits of cone digest
prefixed by 16 of context digest. Nothing else is stored, so the digests are all
that a hit is decided on — the cone itself is never re-compared. The cone half
is 128 bits wide for that reason; the context half is 64, which is the narrower
of the two and the one to widen first if this ever needs strengthening.

### Which ESBMC proved it

A proof must not outlive the build that produced it: change how ESBMC encodes
or checks something and the old verdict may no longer be the right one. The
build is named by the ID stamped into the binary on every build — the commit it
was built from, and whether that tree was dirty — so a rebuild from a different
commit starts a fresh set of entries.

A dirty tree, or a build from no git checkout at all, names a *class* of builds
rather than one, and two of them would otherwise share keys. ESBMC hashes its
own executable in that case, which identifies the build exactly at the price of
one read of the binary per run. So if you build ESBMC itself, expect that read;
a release build does not pay it.

The set of SMT backends the build enabled is in the key too, because which
solver a run uses is only in the option set when it was named on the command
line — otherwise it is whatever this build compiled in. What remains outside
the key is a second build of the same commit, with the same backends, from a
different toolchain: run it once under `--proof-cache-verify` if that is your
situation.

## Only proofs are stored

A claim ESBMC *disproves* is never cached. Reporting a counterexample requires
the model the solver produced, and a stored verdict cannot reconstruct one, so
violated claims are re-solved on every run and their traces are always freshly
generated. A vacuous discharge is not stored either, since it reports `Unknown`
rather than a proof.

The practical consequence is that a failing file gets less benefit than a
passing one, and that the cache can never turn a failure into a success.

## Checking the cache against the solver

`--proof-cache-verify` reads the cache but solves every claim anyway,
reporting an error if a stored proof disagrees with the solver:

```bash
esbmc example.c --multi-property --proof-cache .esbmc-cache --proof-cache-verify
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

Each of these prints one line saying so, because a run that reuses nothing
otherwise looks exactly like a run whose cache is working:

```
WARNING: Proof cache: inactive (--dead-code-check); every claim will be solved and none stored
```

The two ways of naming the feature and getting nothing at all are refused
outright instead: `--proof-cache` without `--multi-property`, and
`--proof-cache-verify` with no cache to check. Both are judged on the flags
alone, so a run that would not have solved anything anyway — `--skip-bmc`, or
one of the print-and-stop modes — is refused too rather than accepting a
combination that could never have worked.

A run that reports no cache line at all solved no claims: everything the
program asserts was discharged by the simplifier before the solver was reached.
That is not a cache failure.

## Invalidation and housekeeping

Every option is folded into the key, including ones that only affect
scheduling and every value of a repeatable one, so changing any flag starts a
fresh set of entries rather than reusing existing ones. This is deliberately
conservative: an option wrongly judged irrelevant would silently reuse a proof
that no longer holds.

The exceptions are the options that reach nothing but the report —
`--verbosity`, `--quiet`, `--log-message`, `--color`, `--ascii-report`,
`--file-output` and `--cex-output`, alongside the cache's own flags and the
path of the file being verified. `--color` and `--ascii-report` are the reason
the list exists at all: both are auto-detected, from the terminal and the
locale, so leaving them in made an interactive run and the same run in a
pipeline disagree on every key while agreeing on every verdict.

Entries are plain files named by their digest and are never evicted. The
directory is safe to delete at any time — the next run simply repopulates it.

## Diagnosing a cache that does not hit

`--claim-fingerprint-dump` writes one line per solved claim giving the digest
of its cone under each normalisation, its size, the verdict and its location:

```bash
esbmc example.c --multi-property --claim-fingerprint-dump fp.tsv
esbmc example.c --multi-property --claim-fingerprint-dump -   # to the log
```

Comparing the dumps from two runs shows which claims changed identity.

## Cost

Keying a claim means hashing its sliced cone, which is work a run without the
cache does not do, and it is paid per claim on every run — a miss costs it, and
so does a hit. On a typical file that is about 0.07 ms per claim, but the spread
is wide: 0.5 ms at the third quartile and 1.6 ms at the 95th, tracking how large
the cones are.

A run also pays once for naming the ESBMC that is proving, and that is free
unless the binary was built from a tree git could not name, in which case it is
one read of the executable — a tenth of a second or so on a 100 MB binary. See
*Which ESBMC proved it*.

What that buys back is capped by the solver's share of the run, so the quantity
that decides whether the cache pays is **solver time per claim**, not program
size. Few expensive claims win: on a 43-property module spending 97% of its time
in the solver, a warm run drops from 6.0s to 0.27s. Many cheap claims lose —
every one is keyed, and none was costing anything to solve.

ESBMC's own regression suite is the second kind. Its median file spends under
1% of its run in the solver, and only 43 of 671 spend more than half, so the
suite is a poor advertisement for the cache and a good illustration of its
limit.

Between those poles sits the case the cache is for. Re-verifying a three-module
project across six commits, each editing one function, took 71.9s without the
cache and 23.3s with it — 3.1x overall, with the first pass paying 1% for a
cache it could not yet hit and later passes running up to 15x faster.

That makes it the wrong tool for one-shot verification -- a competition run, or
a CI job that verifies each file exactly once with no persistent directory. It
is the right tool for the edit-and-recheck loop, where the same file is
verified repeatedly.
