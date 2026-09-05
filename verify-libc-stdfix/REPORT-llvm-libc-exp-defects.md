# LLVM libc `stdfix` exp: one wrong-answer defect, and a documented bound that cannot be met

**Status: DRAFT, not reported upstream.** Held for review.

Files: `libc/src/stdfix/exphk.cpp`, `libc/src/stdfix/expk.cpp`.
Tree: llvm-project a074f5ba20c7.

Two entry points exist — `exphk` (`short accum`, s8.7) and `expk` (`accum`,
s16.15) — and both carry their computation inline rather than calling a shared
`fixed_point::exp`. Each was verified over its **entire input domain** with
ESBMC against an exact, correctly-rounded fixed-point exponential (camada's
`mkFXPExp`), using 32-way input partitioning. Every number below was reproduced
by executing libc's own compiled code.

## Summary

| # | finding | strength |
|---|---|---|
| 1 | `exphk` selects a `SACCUM_MAX` placeholder table entry for inputs it accepts, returning ~36% low | **firm** — a wrong answer, not rounding |
| 2 | `exphk` flushes to zero a value that rounds to a representable non-zero | **firm** — narrow but definite |
| 3 | the stated relative error bound is exceeded, worst measured 101.9× | **the claim looks wrong, not the code** |

Finding 3 is deliberately *not* presented as an accuracy defect. See below: the
absolute error never exceeds ~1.1 ulp anywhere in either domain, and the bound
as written is unachievable by any implementation returning a representable
value. The recommendation there is to restate the bound.

## Method, and why it covers everything

`mkFXPExp` computes exp correctly rounded to nearest with ties to even,
saturating at the format maximum and flushing below half an ulp. It is exact
arithmetic, not an approximation, so it can serve as the reference a documented
bound is checked against.

The oracle was validated before use, against eight natively-measured anchors
spanning both formats' ranges including the saturation and flush boundaries; a
negated anchor fails, so the validation is not vacuous.

Coverage is by input partitioning: the domain is split into 32 disjoint raw
ranges, verified programmatically to tile it exactly with no gap and no overlap,
and each shard verified separately. All shards passing is a complete case split,
not sampling.

| entry point | domain | shards | pass | fail | slowest shard |
|---|---|---|---|---|---|
| `exphk` (s8.7) | 2^16 | 32 × 2048 | 30 | 2 | 529 s |
| `expk` (s16.15) | 2^32 | 32 × 134217728 | 30 | 2 | 1399 s |

The unsharded `expk` query did not finish in 90 minutes; sharded, 31 of 32
returned in under 30 seconds.

## Finding 1: a reachable placeholder table entry

`exphk` reduces `x = hi + mid + lo` and looks `exp(hi)` up in a 12-entry table.
The library's own comment says the top entries do not fit the format:

```cpp
// Notice that when i = 88 and 89, e_hi will overflow short accum range.
static constexpr short accum EXP_HI[12] = {
    ..., 0x1.28d4p7hk, SACCUM_MAX,
};
```

`EXP_HI[11]` is therefore a placeholder, not `exp(6) = 403.4287935`. Nothing
excludes it. The guard rejects `x >= 0x1.64p2 = 5.5625`, but index 88 requires
only `x_rounded = 5.5`, which every `x` in `[5.4375, 5.5625)` rounds to. The
placeholder is then multiplied by `exp_mid * (1 + lo)` with `exp_mid = 0.609` at
that index, and the result collapses:

```
x = 5.4375   exp(x) = 229.8668   exphk = 145.9922
x = 5.5000   exp(x) = 244.6919   exphk = 155.9922      (36% low)
x = 5.5469   exp(x) = 256.4349   exphk = 161.9922
```

`exp(5.5) = 244.69` is comfortably representable in s8.7 (max 255.996), so this
is not saturation — the answer exists and is not returned.

Traced through the real body, `x = 5.5` gives `idx = 88`, `idx >> 3 = 11`,
`EXP_HI[11] = 255.9922`, `exp_mid = 0.609`, `lo = 0`.

### Confirmed end to end against a binary built from libc's own source

Not a transcription: `libc/src/stdfix/exphk.cpp` was compiled directly
(`clang++ -ffixed-point -O2 -DLIBC_NAMESPACE=libc_test -c`), linked into a test
program, and called. The prebuilt `libc.a` in this tree contains no stdfix
symbols -- the whole family is gated behind `LIBC_COMPILER_HAS_FIXED_POINT` --
so compiling the source file is the way to exercise the shipped entry point.

```
exphk(5.5000) = 155.992188   true exp = 244.691932   36.250% low
exphk(5.4375) = 145.992188   true exp = 229.866798   36.488% low
```

Sweeping every raw value around the window locates it exactly:

```
raw      x         exphk        true exp     rel err   note
 695   5.42969   227.257812   228.077960    0.360%
 696   5.43750   145.992188   229.866798   36.488%   <-- window starts
 704   5.50000   155.992188   244.691932   36.250%
 709   5.53906   161.992188   254.439351   36.334%   <-- window ends
 710   5.54688   161.992188   256.434943      --     true value > format max
 712   5.56250   255.992188   260.473206      --     guard fires, saturates
```

**14 inputs** are affected: raw 696-709, i.e. `x` in `[5.4375, 5.5391]`. Every
one has a representable correct answer, and every one is 36-37% low. At raw 695
the error is 0.36% -- ordinary approximation -- so the failure is a cliff, not a
gradual drift.

Raw 710-711 also return low values, but there the true result exceeds the format
maximum so clamping low is defensible; from raw 712 the guard fires and saturates
correctly. The defect is bounded by those two edges.

**`expk` is not affected.** Its `EXP_HI[23]` is likewise an `ACCUM_MAX`
placeholder (`exp(12) = 162754.79` does not fit s16.15), but it is unreachable:
the maximum index reached is 22, consistent with the source's own
"indices <= 355" comment (355 >> 4 = 22). So this is specific to `exphk`'s
narrower table rather than a shared design flaw — worth stating, because
assuming the pattern generalises would be wrong.

## Finding 2: a representable value flushed to zero

At index 0 the table entry is `EXP_HI[0] = 0x1.0p-7hk` — exactly one ulp — and
`lo` can be negative, so `exp_hi * (exp_mid * (1 + lo))` underflows to zero.
The guard only flushes `x <= -0x1.63p2 = -5.5625`, so these inputs are inside
the supported domain:

```
exp(-5.5390625) = 0.003930210     half an ulp = 0.003906250
```

which is above half an ulp, so the correctly-rounded result is raw 1, not 0.
`exphk` returns 0. The window runs from about `x = -5.5547` to `-5.3906`.

This needed checking rather than asserting: the first reading was that flushing
a "representable" value must be wrong, but what matters is whether it *rounds*
to zero. It does not — by a margin of 0.000024.

`expk` shows the same shape from the same cause (`EXP_HI[0] = 0x1p-15k`, one
ulp): `exp(-11.0898132) = 0.0000152671` against a half-ulp of `0.0000152588`.

## Finding 3: the relative bound, and why the claim is the problem

Both files state a relative bound on the final range-reduction step:

```
exphk.cpp:  exp(x) ~ exp(hi) * exp(mid) * (1 + lo)
            "with relative errors < |lo|^2 <= 2^-8"

expk.cpp:   exp(x) ~ exp(hi) * exp(mid) * (1 + lo + lo^2/2)
            "with relative errors < |lo|^3/2 <= 2^-16"
```

Sharded verification found four inputs where the **end-to-end** relative error
exceeds those figures. All confirmed natively:

| entry point | x | exp(x) | libc | absolute | relative | vs bound |
|---|---|---|---|---|---|---|
| `exphk` | −1.3984375 | 0.246982573 | raw 31, want 32 | 0.61 ulp | 1.94e−2 | 4.97× |
| `exphk` | +0.5 | 1.648721271 | raw 212, want 211 | 0.96 ulp | 4.57e−3 | 1.17× |
| `expk` | +0.3828125 | 1.466403054 | raw 48050, want 48051 | 1.10 ulp | 2.28e−5 | 1.49× |
| `expk` | −3.8750305 | 0.020753705 | raw 679, want 680 | 1.06 ulp | 1.55e−3 | **101.90×** |

`x = 0.5` and `x = 0.383` are unremarkable inputs — no saturation, no flush, no
table edge.

**But the absolute error never exceeds ~1.1 ulp anywhere**, and the errors go in
both directions (raw 31 where 32 is correct, raw 212 where 211 is correct). That
is ordinary approximation error, correct to within a rounding step.

The 101.9× case shows what is actually wrong. `exp(-3.875) = 0.0207`, so a
single ulp of absolute error is 1.55e−3 in relative terms. A *relative* bound
cannot be satisfied near the bottom of a range spanning five orders of
magnitude: one ulp of unavoidable rounding already exceeds 2^-16 there, so **no
fixed-point implementation returning a representable value could meet the stated
bound**, and no code change would fix it.

Two further points on the wording:

* the bound is stated for one approximation step, not end to end, so strictly it
  is not a claim about the function's result at all. A reader would reasonably
  take it as one.
* an *absolute* bound of about 2 ulp would be both meetable and true of the
  measured behaviour.

Suggested resolution: restate as an absolute bound, or scope the relative claim
to the range where it holds. This is a documentation change; findings 1 and 2
are the ones needing code.

## Reproduction

```sh
# per-shard, 32 shards each; all shards must pass for whole-domain coverage
esbmc shards_exp/hk_NN.cpp -I$LLVM/libc -I$LLVM/libc/include -I$LLVM --std=c++17
esbmc shards_exp/k_NN.cpp  -I$LLVM/libc -I$LLVM/libc/include -I$LLVM --std=c++17

# the two wrong-answer findings, directly
esbmc harness_exphk_saturated_entry.cpp ...   # FAILED
esbmc harness_exphk_flush_zero.cpp ...        # FAILED
esbmc harness_expk_flush_symbolic.cpp ...     # FAILED (needs --z3)
```

bitwuzla for the shards. `harness_expk_flush_symbolic.cpp` is the one query where
z3 is required: bitwuzla produced no verdict in 90 minutes on the full window,
z3 discharged it in 41 s.

## What was checked and found correct

* `abs`, `countls`, `round`, `bitsfx`/`fxbits` — pass on all formats.
* `divifx`, which all eight `divi*` entry points call — agrees with an exact
  reference on all eight, over a restricted domain.
* `expk`'s top table entry — a placeholder, but proved unreachable.
* Both entry points' saturation and flush guards behave correctly outside the
  two windows above.
* Absolute accuracy: within ~1.1 ulp across both entire domains.
