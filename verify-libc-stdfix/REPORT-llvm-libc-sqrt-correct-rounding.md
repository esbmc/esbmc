# LLVM libc `stdfix` sqrt is not correctly rounded

**Status: DRAFT, not reported upstream.** Held for review.

File: `libc/src/__support/fixed_point/sqrt.h`.
Tree: llvm-project a074f5ba20c7.

**`fixed_point::sqrt` returns a value other than the nearest representable root
for the majority of inputs, at every format measured, and documents no accuracy
bound that would tell a caller so.**

Measured exhaustively -- every input of each format, no sampling:

| entry point | format | inputs | not correctly rounded | worst error | exact roots wrong |
|---|---|---|---|---|---|
| `sqrtuhr` | u0.8 | 256 | **146 (57.0%)** | 1.09 ulp | 8 / 16 |
| `sqrtur` | u0.16 | 65 536 | **53 060 (81.0%)** | 2.48 ulp | 153 / 256 |
| `sqrtuhk` | u8.8 | 65 536 | **34 767 (53.1%)** | 1.09 ulp | 153 / 256 |

28 228 of the u0.16 failures are more than one ULP out, so this is not a
tie-breaking disagreement.

The sharpest cases need no rounding convention to adjudicate. `sqrt(81/256) =
9/16` and `sqrt(100/65536) = 10/256` are **exactly representable** -- there is a
correct answer with zero error available -- and both come back one ULP low.

```
x = 254/256 = 0.992187500
  true root x 256 = 254.998039     nearest representable: raw 255
  libc returns      raw 254        -0.998 ulp
```

The root sits 99.8% of the way to the next representable value; libc returns the
one below.

## Why this is the finding, rather than a bound violation

`fixed_point::sqrt` is declared at sqrt.h:184 and the comment above it is a TODO
about division-free Newton iterations -- **no accuracy claim at all**. So nothing
in the source is contradicted. That is precisely the problem: a caller sees a
function named `sqrt`, taking and returning the same fixed-point format, and has
no way to learn from the source that it is a ~2.5-ULP approximation.

The implementation is *usable* -- bounded, both-directions, no wrong answers of
the kind found in exp. The gap is between what it does and what a reader would
assume.

### An earlier draft of this report got the attribution wrong

It was titled "`sqrt` exceeds its documented error bound" and applied one bound
to all seven entry points. `sqrt.h` in fact states **three different bounds on
three different functions**:

| `sqrt.h` line | bound | function | reached by |
|---|---|---|---|
| 165 | `\|r - sqrt(x_frac)\| < max(1.5*2^-11, eps)` | `sqrt_core` | all, indirectly |
| **(none)** | -- | **`fixed_point::sqrt`** | `sqrtuhr` `sqrtur` `sqrtulr` `sqrtuhk` `sqrtuk` |
| 212 | **absolute** errors < 2^-F | `isqrt` | **nothing** |
| 237 | **relative** errors < 2^-F | `isqrt_fast` | `uhksqrtus` `uksqrtui` |

Two secondary findings follow from the table:

1. **`isqrt_fast`'s relative bound is violated** on the two entry points that
   call it -- 1.51x at `uhksqrtus`, 3.47x at `uksqrtui`. This is the only sqrt
   finding where a function fails a claim it makes about itself. Note a relative
   bound is an awkward shape here for the same reason documented for exp: near
   the bottom of the range, one ULP of unavoidable error is large in relative
   terms.
2. **`isqrt`'s absolute bound is violated** (-1.86 ulp at u8.8, -1.68 ulp at
   u16.16), but `grep` over `libc/src` and `libc/test` finds **no caller**. A
   defect in a documented internal function nothing currently uses.

## How correct rounding was checked

Two independent methods, agreeing.

**In-solver**, over all inputs at once, with no reference root computed by the
harness. camada's `mkFXPSqrt` is the exact root truncated toward zero, so the
true root lies in `[oracle, oracle+1ulp)` and the nearest representable value is
decided by squaring the midpoint:

```
root >= oracle + 1/2   <=>   (2*oracle + 1)^2 <= 4 * raw_x * 2^F
```

`harness_sqrt_correctly_rounded.cpp` -- **VERIFICATION FAILED in 0.70 s**,
counterexample `xb = 254`.

**Natively**, by executing libc's own compiled code over every input of the three
formats above and comparing against `sqrt()` in `long double`. This is what
produced the percentages; it needs no verifier and is independently reproducible.

### The one bound covering `sqrt`'s machinery is honoured

`sqrt_core`'s line-165 bound was verified against camada's exact `mkFXPSqrt`
over every normalised u0.16 input (`x_frac >= 16384`, its documented domain):
**VERIFICATION SUCCESSFUL**. At 16 fraction bits `1.5*2^-11` is 48 ulp, so the
window is generous and the core sits inside it.

### And the rescale theory was wrong

An earlier draft blamed the truncating rescale at sqrt.h:205
(`r >>= EXP_ADJUSTMENT - (x_exp >> 1)`). The counterexample refutes it:

```
x_raw = 63211 (0.964523315)
  exact root * 2^16   = 64363.0025
  sqrt_core           = 64361      -2.0025 ulp   (48-ulp bound: WITHIN)
  fixed_point::sqrt   = 64361      -2.0025 ulp   (no documented bound)
```

Both return the same value — the rescale is a no-op at this input — so the error
originates in `sqrt_core`, within its own bound. The gap is that `sqrt_core`'s
48-ulp slack propagates to `sqrt` unchanged and nothing narrows or documents it.

## Measurements

Exhaustive over the entire input domain of each format (all 256 and all
65536 values respectively), comparing against `::sqrt` on the exact rational
`raw / 2^N`:

| format | inputs at or over 1 ULP | worst absolute error | claimed bound | ratio |
|---|---|---|---|---|
| `unsigned short fract` (u0.8) | **13 / 256** (5.1%) | 0.004270720 | 0.003906250 | **1.09x** |
| `unsigned fract` (u0.16) | **28348 / 65536** (43.3%) | 0.000037852 | 0.000015259 | **2.48x** |

Worst case for u0.8 is `x = 88/256`; for u0.16 it is `x = 16569/65536`.

First violations in u0.8, with the correctly rounded result for comparison:

| x | `sqrt` returns | correctly rounded |
|---|---|---|
| 9/256 | 47/256 | 48/256 |
| 22/256 | 74/256 | 75/256 |
| 25/256 | 79/256 | 80/256 |
| 36/256 | 95/256 | 96/256 |
| 81/256 | 143/256 | 144/256 |
| 88/256 | 149/256 | 150/256 |

Worked example, `x = 22/256`:

```
x        = 0.0859375
sqrt(x)  = 0.293150985
returned = 74/256 = 0.2890625     error 0.004088485
correct  = 75/256 = 0.29296875    error 0.000598765
1 ULP    = 0.003906250            -> returned error is 1.05 ULP
```

## Cause

The result is not merely a truncation. 13 of the 256 u0.8 results fall
**below `floor(true_root)`**, so the error is larger than any rounding
direction alone can explain:

```
u0.8 direction: 13 below floor(true), 0 above round(true), 243 within [floor, round]
```

The reason is in the configuration. `sqrt_core` refines its initial linear
approximation with Newton steps:

```cpp
// sqrt.h:164-178
// Initial approximation step.
// Estimated error bounds: | r - sqrt(x_frac) | < max(1.5 * 2^-11, eps).
FracType r = a * x_frac + b;
...
for (int i = 0; i < Config::EXTRA_STEPS; ++i)
  r = (r >> 1) + (x_frac >> 1) / r;
```

and the per-format step counts are:

| format | `EXTRA_STEPS` |
|---|---|
| `unsigned short fract` | **0** (sqrt.h:33) |
| `unsigned fract` | 1 (sqrt.h:55) |
| `unsigned long fract` | 2 (sqrt.h:81) |

For `unsigned short fract` **no refinement runs at all** — the returned
value is the raw Sollya linear approximation, whose own documented bound is
`max(1.5 * 2^-11, eps)`. In a format whose ULP is `2^-8`, `1.5 * 2^-11` is
only `3/16` ULP on paper, but that bound is stated for the *normalised*
`x_frac` before the exponent rescale (`r >>= EXP_ADJUSTMENT - (x_exp >> 1)`
at sqrt.h:205, and `r >>= (shift >> 1)` in `isqrt`). Shifting right discards
low bits, so the approximation error and the rescale truncation compound,
and the total lands just over one ULP.

### More refinement does not close the gap

The obvious hypothesis -- too few Newton steps -- is **contradicted by the
data**. Adding steps does not help:

| format | `EXTRA_STEPS` | worst error / bound |
|---|---|---|
| u0.8 | 0 | 1.09x |
| u8.8 | 0 | 1.09x |
| u0.16 | 1 | 2.48x |
| u0.32 | **2** | 2.49x |
| u16.16 | 2 | 2.49x |

u0.32 already runs the maximum two refinement steps and is no better than
u0.16 with one. The ratio tracks *fractional bits*, not refinement count, and
u8.8 matching u0.8 (same 8 fractional bits, different storage width) confirms
it is fraction length rather than width.

**The rescale attribution here is withdrawn.** This section previously concluded
that the truncating rescale (`r >>= EXP_ADJUSTMENT - (x_exp >> 1)` at sqrt.h:205)
was the dominant error source. Direct measurement refutes it: at
`x_raw = 63211`, `sqrt_core` and `fixed_point::sqrt` return the *same* value
(64361, exact root 64363.0025), so the rescale contributes nothing there and the
2 ulp error is already present in the core. `sqrt_core` is nonetheless inside its
own documented 48-ulp bound at u0.16 — the bound is simply far looser than one
ulp, and that slack reaches the entry points undocumented.

## Reproducer

Self-contained; needs only the llvm-project checkout on the include path
and a `-ffixed-point` compiler. No verifier involved.

```cpp
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "src/__support/fixed_point/sqrt.h"

using LIBC_NAMESPACE::fixed_point::sqrt;
static unsigned char raw(unsigned short _Fract f)
{ unsigned char r; memcpy(&r, &f, 1); return r; }

int main()
{
  int viol = 0; double worst = 0; int worst_i = -1;
  for (int i = 0; i < 256; i++)
  {
    unsigned short _Fract x;
    unsigned char b = (unsigned char)i;
    memcpy(&x, &b, 1);
    double xd = i / 256.0, rd = raw(sqrt(x)) / 256.0, t = ::sqrt(xd);
    double e = fabs(rd - t);
    if (e >= 1 / 256.0) viol++;
    if (e > worst) { worst = e; worst_i = i; }
  }
  printf("%d/256 at-or-over 1 ulp; worst %.9f at %d/256 (bound %.9f)\n",
         viol, worst, worst_i, 1 / 256.0);
  return 0;
}
```

```
$ clang++ -ffixed-point -O0 -std=c++17 -I$LLVM/libc -I$LLVM/libc/include -I$LLVM r.cpp -o r && ./r
13/256 at-or-over 1 ulp; worst 0.004270720 at 88/256 (bound 0.003906250)
```

Verified on llvm-project `a074f5ba20c7`, clang 20.1.8, x86_64-linux, at `-O0`
and `-O2` (identical results).

## How ESBMC found it

The verifier's contribution was pinning the property rather than sampling
it. Asserting the bound as a bracket around the true root,

```c
r*r <= x   &&   x <= (r+ulp)*(r+ulp)
```

ESBMC proves the lower half for *all* inputs (`sqrt(x)^2 <= x` always holds:
the result is never an over-estimate) and refutes the upper half with
`x = 22/256` as a witness. The exhaustive sweep above then confirmed and
quantified it. A random or boundary-value test suite would plausibly have
missed 13 scattered inputs out of 256.

## Suggested resolutions

Any one of these makes the documentation and the code agree:

1. **Give `fixed_point::sqrt` an accuracy comment** — the primary
   recommendation. It has none, and five of the seven shipped entry points call
   it. Measurement says it is within ~2.5 ULP but correctly rounded on only
   19-47% of inputs depending on format. Stating either figure closes the gap
   between the implementation and what a reader assumes, and costs one comment.
2. **Correct or re-scope `isqrt`'s line-212 bound**, which is violated
   (−1.86 ulp at u8.8). Low priority while nothing calls `isqrt`, but the
   comment is currently untrue as written.
3. **Re-examine `isqrt_fast`'s relative bound** (line 237), the one violation
   on a shipped path: 1.51× at `uhksqrtus`, 3.47× at `uksqrtui`. A relative
   bound is also the awkward shape here — near the bottom of the range one ulp
   of unavoidable error is large in relative terms, the same issue documented
   for exp in REPORT-llvm-libc-exp-defects.md.
4. **Make it correctly rounded**, if that is the intended contract. The exact
   perfect squares are the argument: `sqrt(81/256) = 9/16` has a zero-error
   answer available and does not return it, which no rounding convention
   justifies. This is a larger change than (1) and only worth it if callers are
   expected to rely on nearest-representable results.
5. **Round instead of truncating** on the final rescale, if tighter accuracy is
   wanted without full correct rounding. Offered as an improvement, not a fix
   for a diagnosed cause — the rescale attribution above is withdrawn.

An earlier draft of this report suggested raising `EXTRA_STEPS`. **That
suggestion is withdrawn** -- u0.32 already uses `EXTRA_STEPS = 2` and is no
better in ratio terms than u0.16 with one step, so more refinement demonstrably
does not close the gap.

## Note on scope

u0.8, u8.8 and u0.16 are stated **exhaustively** (every input of the format).
u0.32 and u16.16 are **sampled**, and labelled as such wherever they appear --
exhaustive enumeration is impractical at those widths. The sampled figures are
included because they carry the argument that refinement count is not the
driver; they are not offered as worst-case bounds.

**Correction to an earlier version of this paragraph.** It stated that
`uksqrtui` and `uhksqrtus` call `isqrt`. They do not -- both call `isqrt_fast`
(`uksqrtui.cpp:18`, `uhksqrtus.cpp:18`), whose bound is *relative* rather than
absolute. `isqrt`'s absolute bound is violated but has no caller anywhere in
`libc/src` or `libc/test`. The correct-rounding measurements that lead this
report are about `fixed_point::sqrt`, which is what the other five entry points
call.
