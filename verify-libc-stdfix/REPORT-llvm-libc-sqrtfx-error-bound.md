# LLVM libc `stdfix` sqrt: which bound applies where

**Status: DRAFT, not reported upstream.** Held for review.

**Verdict, corrected.** An earlier draft of this report was titled
"`sqrt` exceeds its documented error bound" and applied one bound to all seven
entry points. That was wrong: `sqrt.h` states **three different bounds on three
different functions**, and the function five of the seven entry points call
states none at all.

| `sqrt.h` line | bound | function | reached by |
|---|---|---|---|
| 165 | `\|r - sqrt(x_frac)\| < max(1.5*2^-11, eps)` | `sqrt_core` | all, indirectly |
| **(none)** | — | **`fixed_point::sqrt`** | `sqrtuhr` `sqrtur` `sqrtulr` `sqrtuhk` `sqrtuk` |
| 212 | **absolute** errors < 2^-F | `isqrt` | **nothing** |
| 237 | **relative** errors < 2^-F | `isqrt_fast` | `uhksqrtus` `uksqrtui` |

So the findings split three ways:

1. **`uhksqrtus` / `uksqrtui` violate `isqrt_fast`'s relative bound** — the claim
   their own function makes. Measured 1.51× and 3.47×. **Firm.**
2. **`isqrt`'s absolute bound is violated** (−1.86 ulp at u8.8, −1.68 ulp at
   u16.16) — but `grep` over `libc/src` and `libc/test` shows **no caller**. A
   defect in a documented internal function nothing currently uses.
3. **The five `sqrt` entry points** are 1–2 ulp from the exact root, and
   153 of 256 exact perfect squares come back one ulp low. But
   `fixed_point::sqrt` documents **no end-to-end bound**, and the one bound in
   its call chain is honoured (see below). This is a **documentation gap**, not a
   proven bound violation.

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

## Evidence for finding 3: exact perfect squares

The measurements below stand as *behaviour*; what changed is the claim they are
evidence for. A caller of a function named `sqrt` returning the same fixed-point
format would reasonably expect near-1-ulp accuracy, and does not get it:

```
sqrt(25)  = 5  exactly -> libc gives  4.99609
sqrt(100) = 10 exactly -> libc gives  9.99609
```

No unsoundness in the arithmetic — the results are usable approximations — but
code relying on the stated bound is relying on something untrue.

## How this was established

Found while verifying LLVM libc's `stdfix` implementation with ESBMC
(esbmc/esbmc PR #4179). The comparison is a **proof over every input of the
format**, not a sampled differential test: the reference is an SMT term
(camada's exact fixed-point square root), so the harness never computes an
expected value.

The reference was itself validated first, against the bracket that
characterises it uniquely — `raw_r^2 <= raw_x * 2^F < (raw_r+1)^2` — so a
mis-wired oracle could not silently produce these findings.

Proved separately per direction, so neither masks the other:

| property | verdict |
|---|---|
| libc is never *above* the true root | **SUCCESSFUL** |
| libc is within 1 ULP *below* the true root | **FAILED** |

The error is one-sided (always downward) and reaches a full ULP, which points
at the truncating rescale rather than at approximation error as such.

Every number below is also reproduced by executing the library's own compiled
code, independently of the verifier.

## The claim — and whose claim it is

```cpp
// libc/src/__support/fixed_point/sqrt.h:211-212
// Integer square root - Accurate version:
// Absolute errors < 2^(-fraction length).
```

`2^(-fraction length)` is one ULP of the result format: `2^-8` for
`unsigned short fract`, `2^-16` for `unsigned fract`.

**This comment sits on `isqrt`, and nothing calls `isqrt`.** The measurements in
the next section were taken against it, which is why they are now reported under
finding 2 (a defect in an uncalled internal function) rather than as a violation
by the shipped entry points. `uhksqrtus` and `uksqrtui` call `isqrt_fast`, whose
bound is *relative* (line 237); the other five call `fixed_point::sqrt`, which
states nothing.

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

1. **Give `fixed_point::sqrt` an accuracy comment.** It has none, and five of the
   seven shipped entry points call it. Whatever bound it actually meets —
   measurement says roughly 2 ulp — stating it would close the gap that makes up
   finding 3 entirely. This is the cheapest and most useful change.
2. **Correct or re-scope `isqrt`'s line-212 bound**, which is violated
   (−1.86 ulp at u8.8). Low priority while nothing calls `isqrt`, but the
   comment is currently untrue as written.
3. **Re-examine `isqrt_fast`'s relative bound** (line 237), the one violation
   on a shipped path: 1.51× at `uhksqrtus`, 3.47× at `uksqrtui`. A relative
   bound is also the awkward shape here — near the bottom of the range one ulp
   of unavoidable error is large in relative terms, the same issue documented
   for exp in REPORT-llvm-libc-exp-defects.md.
4. **Round instead of truncating** on the final rescale, if tighter accuracy is
   wanted. Note this is offered as an improvement, not a fix for a diagnosed
   cause — the rescale attribution above is withdrawn.

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

The violated claim sits on **`isqrt`** (sqrt.h:211-212), which is what the
`uksqrtui` and `uhksqrtus` entry points call -- so the bound is violated on its
own entry points, not merely inherited from the neighbouring
`fixed_point::sqrt`.
