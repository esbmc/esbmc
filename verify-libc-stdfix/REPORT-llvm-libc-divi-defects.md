# Three defects in LLVM libc `fixed_point::divi`

**Status: DRAFT, not reported upstream.** Held for review.

File: `libc/src/__support/fixed_point/fx_bits.h`, `divi` (line 240 onward).
Tree: llvm-project a074f5ba20c7.

Found with ESBMC bounded model checking against libc's own template, included
from the checkout rather than transcribed. Every claim below is proved over all
inputs of the stated format, and every number was reproduced by running the
real function natively under `clang -ffixed-point`.

## Scope: one shipped entry point, not eight

**`fixed_point::divi` is reached only through `rdivi`.**

`fx_bits.h` declares two unrelated functions whose names invite confusion:

```
line 241  divi(int n, int d) -> XType              integer / integer -> fixed-point
line 342  divifx(IntType n, FXType d) -> IntType   integer / fixed-point -> integer
```

All eight `divi*` entry points -- `divir`, `diviur`, `divilr`, `diviulr`,
`divik`, `diviuk`, `divilk`, `diviulk` -- call **`divifx`**:

```cpp
LLVM_LIBC_FUNCTION(int, divir, (int n, fract d)) {
  return fixed_point::divifx<int, fract>(n, d);   // NOT divi
}
```

`stdfix.yaml` confirms the signature: `divir(int, fract) -> int`.

Only `rdivi` calls `divi`:

```cpp
LLVM_LIBC_FUNCTION(fract, rdivi, (int a, int b)) {
  return fixed_point::divi<fract>(a, b);
}
```

So the three defects below are real, but the exposed surface is **`rdivi`
alone**. An earlier draft of this report named the `divi*` family as affected;
that was wrong, and the format labels used throughout the tables below
(`divihr`, `divir`, `divilr`, ...) are **shorthand for `divi<T>` template
instantiations at those formats**, not references to those entry points.

`divifx` was verified separately and **agrees with an exact reference** on the
`_Accum` formats over a restricted domain -- see RESULTS.md. Nothing in this
report applies to it.

## Summary

| # | defect | affected `divi<T>` instantiations | strength of claim |
|---|---|---|---|
| 1 | sign applied twice on the power-of-two-divisor path | s0.7, s0.15, s0.31 (and `_Accum` off-rail) | **firm** -- violates the sign law of division |
| 2 | `1 << F` is UB for `F >= 31` | the four 31/32-fraction-bit formats | **firm** -- C11 6.5.7p3/p4, and clang warns |
| 3 | intermediate format equals result format | s32.31, u32.32 | **weaker** -- libc depends on a conversion TR 18037 leaves undefined |

`rdivi` instantiates `divi<fract>` (s0.15), so **defect 1 is the one that
reaches shipped code**. Defects 2 and 3 need `long`/`_Accum` instantiations that
no current entry point creates -- they are latent until someone instantiates
`divi` at those formats, which the template permits.

Verified across the whole tree, source and tests:

```
$ grep -rn "divi<" libc/src libc/test | grep -v divifx
libc/src/stdfix/rdivi.cpp:18:  return fixed_point::divi<fract>(a, b);
```

`divi<fract>` is the only instantiation that exists. The per-format tables later
in this report therefore describe **potential** behaviour at formats nothing
currently instantiates; only the s0.15 rows correspond to shipped code.

### Defect 1 through rdivi: measured

```
rdivi( -2, -1) = -32768   exact +2.000   should be MAX 32767   WRONG RAIL
rdivi( -3, -1) = -32768   exact +3.000   should be MAX 32767   WRONG RAIL
rdivi(-64,-32) = -32768   exact +2.000   should be MAX 32767   WRONG RAIL
rdivi( -3, -2) = -32768   exact +1.500   should be MAX 32767   WRONG RAIL
rdivi( -5, -4) = -32768   exact +1.250   should be MAX 32767   WRONG RAIL
rdivi(  1, -1) =  32767   exact -1.000   should be MIN -32768  WRONG RAIL
```

Six of nine boundary cases return the opposite saturation rail. Exact fractions
are unaffected: `rdivi(-1,-2) = 16384` (+0.5) is correct.

`rdivi(1, -1)` is also the input **avr-libc's own test suite** expects to yield
`FRACT_MIN` (`tests/simulate/stdfix/rdivi-1.c`), so an independent
implementation's tests disagree with LLVM libc here.

Only the *sign* of the result is claimed as wrong. `divi` documents **no
end-to-end accuracy bound** -- it is Newton-Raphson division with per-iteration
error comments (`E0 = 1/17`, `E1 = 0.0034`, ...) and no overall claim -- so
magnitude deviations of a few ulp are reported as data, never as defects.

## Defect 1: the sign is applied twice

For a power-of-two divisor `divi` takes a fast path:

```c
int64_t scaled_n = static_cast<int64_t>(n) << F;   // line 256: keeps n's sign
int64_t res64 = scaled_n >> k;                     // line 257: still signed
...
long accum res_accum = static_cast<long accum>(res64)
                     / static_cast<long accum>(1 << F);          // line 266-267
res_accum = (d < 0) ? static_cast<long accum>(-1) * res_accum : res_accum;  // 268
```

`res64` already carries the numerator's sign. Line 268 then negates again
whenever `d < 0`. When both operands are negative the quotient is positive, but
the sign is applied twice and the result is negative.

```
divihr(-64, -32) = -128    true quotient +2.0   -> should saturate to MAX 127
divihr( -3,  -1) = -128    true quotient +3.0   -> should saturate to MAX 127
divilr(-64, -32) = -2147483648   true +2.0      -> should be MAX
```

The general branch, taken for non-power-of-two divisors, is correct:

```c
bool result_is_negative = ((n < 0) != (d < 0));    // line 274
```

which is why the same operands with a non-power-of-two divisor work:

```
divihr(-64, -3) = 127     correct
```

### Why the `_Accum` formats appear unaffected

Sign-law violations over the `(n, d)` grid in `[-64, 64]`:

| format | violations |
|---|---|
| `divihr` s0.7 | 649 / 16512 |
| `divir` s0.15 | 649 / 16512 |
| `divilr` s0.31 | 1143 / 16512 |
| `divihk` s8.7 | 0 |
| `divik` s16.15 | 0 |
| `divilk` s32.31 | 1088 / 16512 (defect 2 dominates) |

The `_Accum` zeroes are **not** evidence that those formats are correct. The
double negation still executes; `_Accum` has integer bits, so these quotients
stay in range and never reach the saturation rail, where a flipped sign shows up
as MIN-instead-of-MAX. There it appears as a wrong value instead.

### Proof and control

* `harness_divi_bug1.cpp` -- VERIFICATION FAILED at s0.7.
* `harness_divi_widths.cpp` -- FAILED at s0.15 and s0.31, so it is not
  width-specific.
* `harness_divi_control.cpp` -- **VERIFICATION SUCCESSFUL**. Same signs, same
  magnitudes, non-power-of-two divisor. This is what makes the finding a defect
  in the branch rather than a wrong property: if the control also failed, the
  property or the harness would be at fault.

The control was checked for vacuity, since it is the one that passes: negating
its assertion FAILS, and a reachability probe inside its assumed region FAILS.
Its success is therefore a real proof, not an artefact of unsatisfiable
assumptions.

## Defect 2: `1 << F` is undefined behaviour for `F >= 31`

Line 267 forms the rescale divisor as `static_cast<long accum>(1 << F)`. `1` is
an `int`:

| type | `F` | `1 << F` | standard |
|---|---|---|---|
| `long _Fract` | 31 | `-2147483648` | C11 6.5.7p4 -- signed overflow |
| `unsigned long _Fract` | 32 | `1` | C11 6.5.7p3 -- count >= width |
| `long _Accum` | 31 | `-2147483648` | 6.5.7p4 |
| `unsigned long _Accum` | 32 | `1` | 6.5.7p3 |

Clang already warns at the instantiation:

```
fx_bits.h:267:68: warning: shift count >= width of type [-Wshift-count-overflow]
        static_cast<long accum>(res64) / static_cast<long accum>(1 << F);
```

This is not a benign warning, because the value is a **divisor**. At `F == 31`
the scale is negative, which flips the sign of every result on this path
independently of defect 1. At `F == 32` the scale is `1`, so the rescale is
dropped entirely.

```
divilk(-64, 32) = +4294967296     true quotient -2.0
```

Note also that `-Werror` builds instantiating `divi<unsigned long _Fract>` or
`divi<unsigned long _Accum>` will not compile.

Fix: `1LL << F`, or `static_cast<int64_t>(1) << F`.

### Caught by ESBMC's own UB checker on the real template

The original proof here used `harness_divi_shift.c`, which asserts `scale > 0` --
a hand-written *consequence* of the UB rather than the UB itself, on a reduced
shift rather than on libc's code. That was weaker than necessary.
`--overflow-check` flags the shifts directly in `fx_bits.h`, with no assertion
written by the harness at all (`harness_divi_shift_real.cpp`).

Doing that exposes **three** distinct shift-UB sites on this path, not one. Each
appears once the preceding one is excluded by assumption:

| line | expression | ESBMC diagnostic | condition | standard |
|---|---|---|---|---|
| **256** | `static_cast<int64_t>(n) << F` | undefined behavior on shift operation `shl` | `F >= 0 && F < 64 && (int64_t)n >= 0` | C11 6.5.7p4 -- left operand must be non-negative |
| **257** | `scaled_n >> k` | undefined behavior on shift operation `ashr` | `k >= 0 && k < 64` | C11 6.5.7p3 |
| **266** | `static_cast<long accum>(1 << F)` | arithmetic overflow on `shl` | `!overflow("shl", 1, F)` | 6.5.7p3/p4 -- the site reported above |

Only line 266 was in the original finding. Two are new:

* **Line 256 is UB for any negative `n`**, at every format -- not just `F >= 31`.
  The cast to `int64_t` fixes the *width* but a left-shift of a negative value is
  undefined regardless. `divi(-3, -1)` reaches it, and `rdivi` passes signed
  operands straight through, so this is on the shipped path.
* **Line 257** right-shifts by `k = countr_zero(|d|)`; ESBMC requires
  `k < 64`, which holds for the `uint32_t` argument in practice, so this one is
  a checker completeness condition rather than a reachable defect. Recorded for
  accuracy, not claimed as a bug.

Reproduction, each with the preceding site assumed away:

```sh
# line 256: negative n
esbmc harness_divi_shift_real.cpp --overflow-check ...        # n in [-8, 8]
# line 266: 1 << F at F == 32
esbmc harness_divi_shift_real.cpp --overflow-check ...        # unsigned long _Fract, n >= 0
```

That ESBMC reports these without any user-written property is the stronger form
of the claim: the tool's UB checker, not a hand-derived consequence, is the
oracle.

## Defect 3: no intermediate headroom at s32.31 / u32.32

`divi` computes through `long accum` regardless of `XType`. For narrow targets
that is fine, because `long accum` (s32.31, range +-2^32) has headroom over the
result. When `XType` **is** `long _Accum`, intermediate and target are the same
format, and line 266's `static_cast<long accum>(res64)` is a *value* cast on a
raw 64-bit quantity:

```
res64 = 2^31 -> (long accum)  2147483648.0   exact
res64 = 2^32 -> (long accum) -4294967296.0   wrapped
res64 = 2^33 and above -> 0.0                lost
```

so:

```
divilk(-64, -1) = 0      true quotient +64.0
```

**This claim is weaker than the first two, deliberately.** TR 18037 4.1.3 makes
conversion of a value not representable in an *unsaturated* fixed-point type
undefined, so clang wrapping to zero is not itself a compiler bug. The defect is
that `divi` relies on that conversion for arguments it accepts, and documents no
precondition excluding them. A caller cannot tell from the interface that
`divilk(-64, -1)` is outside the supported domain.

Proof: `harness_divi_widecast.cpp` -- FAILED, asserting only that an
exactly-representable nonzero quotient is not returned as zero.

## A candidate fix, and the evidence it gives

Repairing defects 1 and 2 in the fast path -- shift the magnitude, apply the
sign once from both operands, widen the shift literal:

```c
bool neg = ((n < 0) != (d < 0));
int64_t an = n < 0 ? -static_cast<int64_t>(n) : static_cast<int64_t>(n);
int64_t res64 = (an << F) >> k;
if (neg) res64 = -res64;
...
/ static_cast<long accum>(1LL << F);
```

Exhaustive over the power-of-two-divisor cases in the same grid:

| format | pow2 cases | upstream wrong | patched wrong |
|---|---|---|---|
| `divihr` s0.7 | 1806 | 649 | **0** |
| `divir` s0.15 | 1806 | 649 | **0** |
| `divilr` s0.31 | 1806 | 1143 | **0** |
| `divihk` s8.7 | 1806 | 0 | 0 |
| `divik` s16.15 | 1806 | 0 | 0 |
| `divilk` s32.31 | 1806 | 1780 | 1044 |

Five formats going to exactly zero is the strongest evidence available that
lines 257/268 and 267 are the right lines. `divilk`'s remainder is defect 3,
which these two lines cannot address -- it needs an intermediate wider than
`long accum`.

This is offered as evidence for the diagnosis, not as a proposed patch; the
maintainers may prefer a different intermediate strategy, especially for
defect 3.

## Corroboration from another project

`rdivi(1, -1)` returning `FRACT_MAX` rather than `FRACT_MIN` contradicts
**avr-libc's own published test expectation** for the same function
(`tests/simulate/stdfix/rdivi-1.c`). An independent implementation's test suite
disagrees with LLVM libc on exactly this input, which is a stronger argument
than a single reading of the TR.

## What was checked and found correct

Not everything suspected turned out to be a defect, and two of my own earlier
claims were retracted:

* **The saturation-rail framing was wrong for `_Accum`.** An earlier sweep
  assumed any quotient with `|q| >= 1.0` must saturate. That holds only for the
  *fract* formats; `_Accum` has integer bits, so `divik(16, 1) = 16.0` is
  correct. The sweep reported "272/272 wrong" for `divik` and `divilk` purely
  as an artefact of that assumption. Hand-checking `divik(4,2)`, `(16,1)`,
  `(-16,1)`, `(5,-1)`, `(1,-16)`, `(-16,-1)`, `(3,2)` showed all exact.
* **`divir(-3, 3) = -32767` against MIN is one ulp**, on a function with no
  documented accuracy bound. Retracted as a defect.
* Exact fractions are correct throughout: `rdivi(1,2) = 16384`,
  `rdivi(-1,2) = -16384`.
* The general (non-power-of-two) branch derives its sign correctly at every
  format tested.

## Reproduction

```sh
esbmc harness_divi_bug1.cpp     -I$LLVM/libc -I$LLVM/libc/include -I$LLVM --std=c++17  # FAILED
esbmc harness_divi_control.cpp  -I$LLVM/libc -I$LLVM/libc/include -I$LLVM --std=c++17  # SUCCESSFUL
esbmc harness_divi_widths.cpp   -I$LLVM/libc -I$LLVM/libc/include -I$LLVM --std=c++17  # FAILED
esbmc harness_divi_widecast.cpp -I$LLVM/libc -I$LLVM/libc/include -I$LLVM --std=c++17  # FAILED
esbmc harness_divi_shift.c --overflow-check                                            # FAILED
```

All five give identical verdicts under `--no-simplify`, so none of this depends
on ESBMC's simplifier folding the arithmetic.
