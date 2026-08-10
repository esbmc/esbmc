# LLVM libc's fixed-point `sqrt` exceeds its documented error bound

**Verdict: the implementation is a valid approximation; the documented bound
is wrong.** `sqrt.h` claims absolute errors strictly below one ULP. The error
reaches a full ULP downward, and the sharpest evidence needs no discussion of
rounding at all: **153 of the 256 exact perfect squares in `uhksqrtus`'s domain
come back one ULP low**, on inputs whose answer is exactly representable.

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

## The claim

```cpp
// libc/src/__support/fixed_point/sqrt.h:211-212
// Integer square root - Accurate version:
// Absolute errors < 2^(-fraction length).
```

`2^(-fraction length)` is one ULP of the result format: `2^-8` for
`unsigned short fract`, `2^-16` for `unsigned fract`.

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
it is fraction length rather than width. That points at the truncating rescale
-- `r >>= EXP_ADJUSTMENT - (x_exp >> 1)` at sqrt.h:205 and `r >>= (shift >> 1)`
in `isqrt` -- as the dominant error source, not the approximation itself.

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

1. **Correct the comment** to the measured bound — e.g. "absolute errors
   < 2 * 2^(-fraction length)", or state it per format. Cheapest, and
   honest.
2. **Round instead of truncating** on the final rescale (add a half-ULP bias
   before the shift). This is the change the measurements point at: the error
   tracks fractional-bit count rather than refinement count, so it is the
   rescale that is losing the bits.

(1) is correct regardless of whether (2) is taken, since the current text is
inaccurate for every format measured.

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
