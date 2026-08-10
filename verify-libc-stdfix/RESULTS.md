# LLVM libc stdfix verification results

ESBMC against LLVM libc's own sources (llvm-project a074f5ba20c7), included
from the checkout -- not transcriptions. Properties come from ISO/IEC TR
18037 7.18a.6 or from the library's own documented bounds, never from
re-implementing the code under test. 8/16-bit formats are exhaustive.

| family | entry points | verdict |
|---|---|---|
| `abs` | 6 | **PASS** -- non-negative, magnitude-preserving with MIN saturating, idempotent |
| `countls` | 12 | **PASS** -- TR contract: largest k with `v << k` not overflowing; both admissibility and maximality; signed and unsigned paths |
| `round` | 12 | **PASS** -- multiple-of-step, within-one-step, nearest, tie-to-+Inf, saturating; all 256 inputs x all 7 positions of n (n symbolic) |
| `sqrt` | 5 | **BOUND VIOLATED** -- see REPORT-llvm-libc-sqrtfx-error-bound.md |
| `divi` | 8 | **3 BUGS** -- double sign negation in the power-of-two branch; `1 << F` UB at F>=31; no intermediate headroom at s32.31/u32.32. See below. |
| `bitsfx`/`fxbits` | 24 | **PASS** -- mutual inverses in both directions, exhaustive over 8-bit formats; raw pattern is the scaled value |
| `idiv` | 8 | **PASS** on the whole domain except one input; see the overflow question below |
| `exp` | 2 | not yet tested |

## roundfx tie direction: settled

TR 18037 leaves the halfway direction unspecified, which is why camada's
`mkFXPToFXPRound` takes it as a parameter and ESBMC deferred picking one.
Measured exhaustively on LLVM libc (s0.7, n=3, every tie enumerated):
**every tie rounds toward +Inf**, negatives included (`-8 -> 0`, not `-16`),
and the topmost tie saturates to MAX rather than wrapping. Their
"Round-to-nearest, tie-to-(+Inf)" comment is accurate.

So when ESBMC wires `roundfx` in Phase 4 it should pass the +Inf tie
direction to match this implementation. Note this is LLVM libc's choice, not
a standard requirement -- avr-libc documents `bit = -1` rounding to even as
an extension, so a consumer targeting that toolchain may need the other
argument.

## ESBMC defects found while building these harnesses

Each blocked a harness and is fixed on this branch:

1. `abs`/`sqrt`/`remainder` rewritten by name regardless of who defined the
   function (esbmc/esbmc#6904) -- fixed by a single user-definition guard.
2. Compound shift-assign on fixed-point left unlowered, then the shift count
   cast into the computation type (esbmc/esbmc#6924, C11 6.5.7p3).
3. `__builtin_clzg` unmodelled, returning nondet -- LLVM libc's
   `cpp::countl_zero` prefers it, so leading-zero code silently computed
   garbage (esbmc/esbmc#6925). The `countls` result above is an independent
   check on the model that replaced it.

## idiv: one input, and a question rather than a bug

`idiv` agrees with truncating raw division on **65279 of 65280** signed
(x, y) pairs with y != 0. The exception:

```
idiv<short fract,int>(-128/128, -1/128) = -128     exact quotient = 128
```

x is the s0.7 minimum and y is -1 ulp, so the exact quotient is 2^(N-1),
which does not fit the **signed N-bit `CompType`** the implementation divides
in -- the classic INT_MIN/-1 wrap. The `int` return type holds 128 without
difficulty.

LLVM libc cites TR 18037's "if an integer result of one of these functions
overflows, the behavior is undefined" (fx_bits.h:222-223) directly above the
division, so the behaviour may well be intended. But the overflow here is in
the *intermediate*, not in the result type, and whether the TR's sentence
covers that is a reading question. Two defensible positions:

* the result does not overflow (`int` holds 128), so the TR clause does not
  apply and this is a defect;
* the implementation's chosen intermediate overflows, and any overflow along
  the way is UB, so the value is unspecified and callers must not rely on it.

Recorded, not asserted; the harness excludes the input and proves the rest.
Worth asking the library authors which they intend, since a caller cannot
tell from the documentation.

## 8-bit tier: complete

The eight stdfix entry points whose argument is an 8-bit fixed-point type
(`hr` / `uhr` suffix) are all covered. Both 8-bit domains are 256 values, so
each result below is a proof over every input of that type.

| entry point | algorithm | verdict |
|---|---|---|
| `abshr` | `abs` | PASS (3 properties) |
| `bitshr` | `bitsfx` | PASS (round-trip both directions) |
| `bitsuhr` | `bitsfx` | PASS |
| `countlshr` | `countls` | PASS (TR contract, admissible + maximal) |
| `countlsuhr` | `countls` | PASS |
| `roundhr` | `round` | PASS (5 properties, n symbolic over all positions) |
| `rounduhr` | `round` | PASS (same, unsigned rail) |
| `sqrtuhr` | `sqrt` | **documented bound violated** (1.09x at u0.8) |

Also verified at 8-bit though the library ships no 8-bit instantiation:
`abs` on the unsigned format (identity, SIGN_LEN == 0 arm) and `idiv` on
`short _Fract` (the template at a width the shipped entry points do not
use -- `divi`/`idiv` start at `r`/16-bit).

### Correction to earlier coverage claims

An earlier version of this file implied ~75 of 80 entry points were verified.
That was wrong: the harnesses had each exercised **one or two type instances**,
and the rest was inferred from the wrappers sharing a template. The table
above is the honest 8-bit statement. The 16-, 32- and 64-bit instantiations
are NOT yet verified, and width matters demonstrably:

* `sqrt`'s `EXTRA_STEPS` differs per width (0 / 1 / 2), and the measured
  error was 1.09x the bound at u0.8 but **2.48x at u0.16**;
* `divi`'s bugs involve the power-of-two fast path and, separately, a
  `1 << F` shift that is UB once `F >= 31`;
* `idiv`'s single failing input is an N-bit intermediate overflow, which
  recurs at every width;
* `countls` subtracts `SIGN_LEN`, `round` shifts by `FRACTION_LEN`.

Note on tractability for the wider tiers: BMC does not enumerate, it proves,
so a 32-bit property is one symbolic query rather than 4 billion cases. Domain
size is not the limiting factor; formula difficulty is.

### `_Sat` types: nothing to test here

`stdfix.h` declares no function taking a `_Sat` argument, and LLVM libc's
`FXRep` specialisations for them are empty inheritance
(`FXRep<short sat fract> : FXRep<short fract> {}`) -- same width, scale, MIN,
MAX, EPS. Saturation is a property of the arithmetic the *compiler* emits for
those types, not of anything in the library. Verifying it would be testing
clang's codegen against ESBMC's `mkFXP*Sat` encoding -- ESBMC
self-validation, which belongs in `regression/fixedbv/`, not in this
directory.

## 16-bit tier

Four types: `_Fract` (s0.15), `unsigned _Fract` (u0.16), `short _Accum`
(s8.7), `unsigned short _Accum` (u8.8) -- 26 entry points. The `_Accum` types
are the first with integer bits, and this is the first tier where `divi` and
`idiv` have shipped instantiations at all (the smallest are `divir`/`idivr`).

ESBMC proof over all four types took 12.6s (8-bit tier: 0.4s).

| family | types | verdict |
|---|---|---|
| `abs` | s0.15, s8.7 | **PASS** |
| `countls` | u0.16, s8.7, u8.8 | **PASS** (TR contract, admissible + maximal) |
| `round` | s0.15, s8.7 | **PASS** (n symbolic over all positions) |
| `bits`/`fxbits` | all four | **PASS** (round-trip) |
| `idiv` | s0.15 | **PASS** sampled; same intermediate-overflow input as 8-bit |
| `sqrt` | u0.16, u8.8 | **BOUND VIOLATED** at both |
| `divi` | s0.15 | **BUG** -- sign applied twice on the power-of-two path; see the root-cause section |

### sqrt error grows with width

| format | inputs over 1 ulp | worst error | bound | ratio |
|---|---|---|---|---|
| u0.8 | 13 / 256 (5.1%) | 0.004270720 | 0.003906250 | 1.09x |
| u8.8 | 1904 / 65536 (2.9%) | 0.004252542 | 0.003906250 | 1.09x |
| u0.16 | 28348 / 65536 (43.3%) | 0.000037852 | 0.000015259 | **2.48x** |

u8.8 tracks u0.8 (same 8 fractional bits, same `EXTRA_STEPS`), while u0.16
is far worse -- consistent with the error being driven by fractional-bit
count and refinement steps rather than storage width.

### divi at s0.15 -- superseded by the root-cause analysis below

An earlier pass here reported "60/272 negative-rail and 55/272 positive-rail
wrong" and listed `divir(-3,3) = -32767` as a defect. **Two corrections**, both
found by widening the sweep to every format and then hand-checking:

1. The rail framing was mine, not the library's. That sweep assumed any
   quotient with `|q| >= 1.0` must saturate, which is only true for the
   *fract* formats. `_Accum` has integer bits, so `divik(16,1) = 16.0` is
   correct and my sweep called it wrong -- it produced a spurious "272/272
   wrong" for `divik`/`divilk` until corrected. Verified by hand:
   `divik(4,2)`, `(16,1)`, `(-16,1)`, `(5,-1)`, `(1,-16)`, `(-16,-1)`, `(3,2)`
   are all exact.
2. `divir(-3,3) = -32767` against MIN `-32768` is **one ulp of magnitude**, and
   `divi` has *no documented accuracy bound* -- it is Newton-Raphson with
   per-iteration error comments and no end-to-end claim. A 1-ulp magnitude
   error is therefore not a defect. It should not have been listed as one.

What survives is sharper than the original claim, and is stated by root cause
in the next section rather than by width.

### idiv: the same single input

`idivr(MIN, -1ulp) = -32768`, exact quotient 32768 -- the N-bit intermediate
overflow described for the 8-bit tier, at the shipped width. Sampled
elsewhere with no mismatches; the symbolic proof for this width is in
harness_idiv.cpp's shape but was run at s0.7.

## 32-bit tier

Four types: `long _Fract` (s0.31), `unsigned long _Fract` (u0.32), `_Accum`
(s16.15), `unsigned _Accum` (u16.16) -- 30 entry points. ESBMC proof: 10.7s,
comparable to the 16-bit tier despite a domain 65536x larger, which is the
point of proving rather than enumerating.

| family | types | verdict |
|---|---|---|
| `abs` | s0.31, s16.15 | **PASS** |
| `countls` | u0.32, s16.15, u16.16 | **PASS** |
| `round` | s0.31, s16.15 | **PASS** (n symbolic over all 31 / 15 positions) |
| `bits`/`fxbits` | all four | **PASS** |
| `sqrt` | u0.32 | **BOUND VIOLATED, 2.48x** |

### More Newton refinement does not close the sqrt gap

`unsigned long fract` is the only format with `EXTRA_STEPS = 2`; u0.16 has 1
and u0.8 has 0. If the bound miss were simply "not enough refinement", u0.32
should be the best of the three. It is not:

| format | EXTRA_STEPS | fractional bits | worst error / bound |
|---|---|---|---|
| u0.8 | 0 | 8 | 1.09x |
| u8.8 | 0 | 8 | 1.09x |
| u0.16 | 1 | 16 | 2.48x |
| u0.32 | **2** | 32 | **2.48x** (17579 of 41011 sampled inputs over bound) |

So the ratio tracks *fractional bits* (8 -> 1.09x, 16+ -> 2.48x) and is
insensitive to the refinement count. That points at the final rescale
(`r >>= EXP_ADJUSTMENT - (x_exp >> 1)`) rather than at the approximation
being under-refined -- a truncating shift costs up to 1 ulp regardless of how
accurate the value was beforehand. Suggested fix (3) in the sqrt report
(round instead of truncate on the rescale) is therefore the one most likely
to help; adding a Newton step, suggestion (2), is contradicted by this data
and should be dropped from the report.

### A harness bug worth recording

The first 32-bit run reported a `round` failure that was mine, not libc's.
The bracket computation

```c
const int down = (xr >= 0 ? (xr / step) * step : ((xr - step + 1) / step) * step);
```

is fine at 8 and 16 bits but overflows `int` at 32: with `xr = -1332312064`
and `n = 1`, `step` is 2^30 and `xr - step + 1` underflows, yielding
`down = -3221225472` and a negative "distance". Hand-checking the
counterexample showed libc's answer (-1073741824) was the correct nearest
multiple and the harness was accusing it wrongly. Fixed by computing the
bracket in `int64_t`, then re-validated natively before re-running.

The general lesson for the remaining tier: a property harness needs its own
arithmetic checked at each width, or it manufactures findings. Every
"failure" gets hand-verified against the definition before it is called a
library defect.

**Do the 8- and 16-bit results survive this bug?** Yes, on two independent
grounds. Statically, the faulty expression overflows only if an intermediate
exceeds 2^31-1; the maximum over every rounding position is 256 at s0.7,
65536 at s0.15 and 32896 at s8.7, four to five orders of magnitude below the
limit, so the `int` arithmetic was exact at those widths. Empirically, all
seven 8- and 16-bit harnesses were re-run after the fix and all still verify.

The distinction matters because a corrupted expectation can fail spuriously
(what happened at 32-bit) *or* pass spuriously, if it coincides with a
genuinely wrong library result -- the more dangerous direction. Neither is
reachable at 8/16-bit. The 8- and 16-bit `round` properties were also
native-swept exhaustively with the same formula (2048 and 458752 cases, zero
mismatches), so those two tiers have both a proof and an independent
enumeration behind them.

## 64-bit tier -- and all widths now use __int128 brackets

Two types: `long _Accum` (s32.31), `unsigned long _Accum` (u32.32) -- 13
entry points. No `sqrt` or `exp` exists at this width. ESBMC proof: 11.3s.

| family | types | verdict |
|---|---|---|
| `abs` | s32.31 | **PASS** (3 properties) |
| `countls` | s32.31, u32.32 | **PASS** (u32.32 includes maximality) |
| `round` | s32.31 | **PASS** (n symbolic over all 31 positions) |
| `bits`/`fxbits` | both | **PASS** (round-trip) |

Every harness -- all four tiers -- now computes its bracket arithmetic in
`__int128` rather than `int` or `int64_t`. That removes the overflow class
entirely instead of width-by-width: a raw value is at most 64 bits and step
at most 2^32, so `__int128` has ~30 bits of headroom everywhere. All eight
harnesses were re-verified after the change and all still pass.

## Final coverage

| tier | types | ESBMC time | result |
|---|---|---|---|
| 8-bit | s0.7, u0.8 | 0.4s | PASS (sqrt bound violated) |
| 16-bit | s0.15, u0.16, s8.7, u8.8 | 12.6s | PASS (sqrt violated, divi buggy) |
| 32-bit | s0.31, u0.32, s16.15, u16.16 | 10.7s | PASS (sqrt violated) |
| 64-bit | s32.31, u32.32 | 11.3s | PASS |

All 12 stdfix types covered for `abs`, `countls`, `round` and
`bits`/`fxbits`. Note the times are flat across tiers -- a 64-bit proof costs
about the same as a 16-bit one, because BMC solves rather than enumerates.

Outstanding: `exp` (2 entry points, 16- and 32-bit only) is deferred pending
bit-precise support in camada. `divi`/`idiv` are verified at s0.7 and s0.15
with findings recorded; the wider instantiations are not separately swept.

## Are the results an artefact of ESBMC's simplifier?

Reasonable worry, and it has teeth: earlier in this branch the constant
folder and the solver *did* disagree about fixed-point narrowing, and the
mismatch was verdict-flipping (see the `[util] Align constant folding...`
commit). So the harnesses were re-run with the folder disabled.

**All nine harnesses pass with `--no-simplify`.** VCC counts show how much
work the simplifier normally does:

| harness | default | `--no-simplify` |
|---|---|---|
| 16-bit | 126 generated, **27** remaining | 126 generated, **117** remaining |
| 32-bit | 126 generated, **27** remaining | 126 generated, **117** remaining |
| 64-bit | 73 generated, **16** remaining | 73 generated, **67** remaining |

So roughly 90 of 126 conditions are normally discharged by the folder rather
than the solver. With it off, the solver decides all of them, and the verdicts
are unchanged -- the two mechanisms agree here.

**Negated controls, both modes.** A harness that passes because its
assertions were discarded is indistinguishable from one that passes because
the code is correct, unless you break something on purpose. Flipping the
first assertion of each tier (`r >= ZERO` to `r < ZERO`) gives:

| harness | default | `--no-simplify` |
|---|---|---|
| absr (8-bit) | FAILED | FAILED |
| 16-bit | FAILED | FAILED |
| 32-bit | FAILED | FAILED |
| 64-bit | FAILED | FAILED |

Eight for eight, so the assertions are genuinely evaluated in both modes and
the inputs are not over-constrained into vacuity.

Also confirmed structurally: `nondet_*()` has no body, so ESBMC havocs it --
the GOTO shows `NONDET(fixedbv width:16 integer_bits:1)`, an unconstrained
value, not a sample. And the libc templates appear in the GOTO program as
real instantiated bodies (`abs<Fract>`, `countls<UFract>`, ...), so the code
under test is not itself being havocked by the same no-body rule.

## Three entry points earlier inventories missed

`uksqrtui`, `uhksqrtus` and `rdivi` do not follow the family naming
(`<fam><suffix>`), so the family sweeps skipped them. All three are now
measured, and two of them matter.

### uhksqrtus / uksqrtui: the bound claim is attached to THESE

The "Absolute errors < 2^(-fraction length)" comment sits on `isqrt`
(sqrt.h:211-212), and `isqrt` is what these two call -- `fixed_point::sqrt`,
which the earlier sqrt work tested, is the neighbouring fixed->fixed
function. So the violated claim is violated on its own entry points, not
merely inherited:

| entry point | signature | domain | over 1 ulp | worst / bound |
|---|---|---|---|---|
| `uhksqrtus` | `unsigned short -> u8.8` | exhaustive, 65536 | **28348 (43%)** | **2.48x** |
| `uksqrtui` | `unsigned int -> u16.16` | sampled 65624 | **28599 (44%)** | **2.49x** |

Concrete case: `uhksqrtus(65535)` returns 255.992188 against a true root of
255.998047 -- an error of 0.005859 against a 1-ulp tolerance of 0.003906.

This strengthens the sqrt report: it now covers both the fixed->fixed and
the int->fixed paths, and the int->fixed one is where the documentation
actually lives.

### rdivi: inherits the divi bug

`rdivi` is `divi<fract>` verbatim, so the s0.15 defect reproduces
identically -- 3 of 8 boundary cases wrong:

```
rdivi( 1, -1) =  32767   want MIN (-32768)   WRONG   (wrong rail, bug 1)
rdivi( 2, -1) =  32767   want MIN (-32768)   WRONG   (wrong rail, bug 1)
rdivi(-3,  3) = -32767   want MIN (-32768)   1 ulp -- NOT a defect, see above
```

The first two are bug 1: a negative power-of-two divisor on the fast path, so
the sign is applied twice and the result lands on the opposite rail. The third
is one ulp of magnitude on a function with no documented accuracy bound, and is
retracted as a defect.

Worth noting for the divi report: `rdivi(1,-1) != FRACT_MIN` is one of
**avr-libc's own published test expectations**
(`tests/simulate/stdfix/rdivi-1.c`), so an independent project's test suite
disagrees with LLVM libc on exactly this input. That is a stronger argument
than a lone reading of the TR.

Exact fractions are unaffected: `rdivi(1,2) = 16384` (0.5) and
`rdivi(-1,2) = -16384` are both correct, so the defect is confined to the
saturating rails.

## Vacuity: every PASS verdict is now controlled

A harness whose assumptions are unsatisfiable passes trivially and proves
nothing. Each passing harness was re-run with its first assertion negated; all
of them **must** fail, and all of them do:

| harness | negated assertion |
|---|---|
| `harness_countls.cpp` | FAILED (good) |
| `harness_round.cpp` | FAILED |
| `harness_bitsfx.cpp` | FAILED |
| `harness_8bit_complete.cpp` | FAILED |
| `harness_16bit.cpp` | FAILED |
| `harness_64bit.cpp` | FAILED |
| `harness_divi_control.cpp` | FAILED, plus a reachability probe that also FAILED |

So no PASS in this file rests on an empty assumption region.

## divi, by root cause: three separate defects

Sweeping every one of the ten signed/unsigned fract and accum formats, then
separating the *sign law* (`sgn(n/d) = sgn(n)*sgn(d)`, which needs no accuracy
contract) from *magnitude* (which has no documented bound and so is reported as
data only), the failures collapse to three independent causes. Each is proved
over all inputs by ESBMC against libc's real template, and each has a control
that isolates it.

### Bug 1 -- the sign is applied twice on the power-of-two path

```
fx_bits.h:257   int64_t res64 = scaled_n >> k;              // scaled_n = n << F, keeps n's sign
fx_bits.h:268   res_accum = (d < 0) ? -1 * res_accum : res_accum;   // negates AGAIN on d's sign
```

For `n < 0` and `d < 0` the quotient is positive, but the sign is applied
twice, so it comes back negative. The general (non-power-of-two) branch is
correct -- it derives `result_is_negative = ((n < 0) != (d < 0))` at line 274.

```
divihr(-64,-32) = -128   true quotient +2.0  -> should be MAX 127
divihr( -3, -1) = -128   true quotient +3.0  -> should be MAX 127
divihr(-64, -3) =  127   correct: general branch
```

Sign-law violations per format, over the (n,d) grid in [-64,64]:

| format | sign bugs | note |
|---|---|---|
| `divihr` s0.7 | 649 / 16512 | |
| `divir` s0.15 | 649 / 16512 | |
| `divilr` s0.31 | 1143 / 16512 | also hit by bug 2 |
| `divihk` s8.7 | **0** | integer bits keep results off the rail |
| `divik` s16.15 | **0** | |
| `divilk` s32.31 | 1088 / 16512 | bug 2 dominates here |

Proof: `harness_divi_bug1.cpp` FAILS (s0.7), `harness_divi_widths.cpp` FAILS
(s0.15 and s0.31). The **control** `harness_divi_control.cpp` -- identical
signs and magnitudes, non-power-of-two divisor -- VERIFIES SUCCESSFUL, which
is what pins the defect to this branch rather than to the property. The
control was checked for vacuity: negating its assertion FAILS, and a
reachability probe in its assumed region FAILS, so its success is a real
proof and not an empty one.

Note the `_Accum` zeroes do not mean `_Accum` is unaffected. The double
negation still happens; integer bits simply keep these quotients away from the
saturation rail, where a flipped sign is visible as MIN-instead-of-MAX. There
it surfaces as a wrong value.

### Bug 2 -- `1 << F` is undefined behaviour for F >= 31

```
fx_bits.h:266-267   long accum res_accum =
                      static_cast<long accum>(res64) / static_cast<long accum>(1 << F);
```

`1` is an `int`. Clang warns at the instantiation (`shift count >= width of
type`). The reached fraction lengths are:

| type | F | `1 << F` evaluates to | standard |
|---|---|---|---|
| `long _Fract` | 31 | `-2147483648` | C11 6.5.7p4, signed overflow |
| `unsigned long _Fract` | 32 | `1` | C11 6.5.7p3, count >= width |
| `long _Accum` | 31 | `-2147483648` | 6.5.7p4 |
| `unsigned long _Accum` | 32 | `1` | 6.5.7p3 |

This value is used as a **divisor**, so it is not a benign warning: at F=31 the
scale is negative, which flips the sign of every result independently of bug 1;
at F=32 the scale is 1, dropping the rescale entirely. `divilk(-64,32)` returns
`+4294967296` where the answer is `-2.0`.

The F=32 instantiations do not compile warning-free, and `-Werror` builds of
anything that instantiates `divi<unsigned long _Fract>` would reject them.

Proof: `harness_divi_shift.c` FAILS under `--overflow-check`. It needs no
fixed-point types at all -- it is a plain integer defect, which is why it is a
separate C harness.

### Bug 3 -- no intermediate headroom at s32.31 / u32.32 (weaker claim)

`divi` routes its result through `long accum` regardless of `XType`. For narrow
targets that is harmless because `long accum` (s32.31, range +-2^32) has
headroom. When `XType` *is* `long _Accum` the intermediate and the target are
the same format, and `static_cast<long accum>(res64)` is a **value** cast that
wraps:

```
2^31 ->  2147483648.0   exact
2^32 -> -4294967296.0   WRAPPED
2^33 and above ->  0.0  LOST
```

Hence `divilk(-64,-1) = 0` where the answer is 64.0.

**This is a weaker claim than bugs 1 and 2, and deliberately stated as such.**
TR 18037 4.1.3 makes conversion of a value not representable in an
*unsaturated* fixed-point type **undefined**, so the compiler wrapping to zero
is not itself wrong. The defect is that `divi` depends on that conversion for
inputs it accepts and documents no precondition excluding them.

Proof: `harness_divi_widecast.cpp` FAILS, asserting only that an
exactly-representable nonzero quotient does not come back as zero -- a property
that needs no accuracy contract either.

### A candidate fix, and what it confirms

Repairing bugs 1 and 2 in the power-of-two branch -- take `|n|` for the shift,
apply the sign once from both operands, and use `1LL << F` -- clears the branch
completely on five of six signed formats:

| format | pow2 cases | upstream wrong | patched wrong |
|---|---|---|---|
| `divihr` s0.7 | 1806 | 649 | **0** |
| `divir` s0.15 | 1806 | 649 | **0** |
| `divilr` s0.31 | 1806 | 1143 | **0** |
| `divihk` s8.7 | 1806 | 0 | 0 |
| `divik` s16.15 | 1806 | 0 | 0 |
| `divilk` s32.31 | 1806 | 1780 | 1044 |

Going to zero on five formats is the strongest available evidence that lines
257/268 and 267 are the right lines. `divilk`'s residue is bug 3, which those
two lines cannot fix -- it needs a wider intermediate than `long accum`.

### Verdicts, and simplifier independence

| harness | expected | `--no-simplify` |
|---|---|---|
| `harness_divi_bug1.cpp` | FAILED | FAILED |
| `harness_divi_control.cpp` | **SUCCESSFUL** | **SUCCESSFUL** |
| `harness_divi_widths.cpp` | FAILED | FAILED |
| `harness_divi_widecast.cpp` | FAILED | FAILED |
| `harness_divi_shift.c` | FAILED | FAILED |

Identical with the simplifier disabled, so none of it is a simplifier artefact.

## Methodology correction: the sqrt results are now PROOFS, not differential tests

The earlier sqrt numbers in this file (1.09x at u0.8, 2.48x at u0.16, "43% of
inputs over 1 ulp") were produced by running libc natively under
`clang -ffixed-point` and comparing against `long double`. That is a
**differential test**, not a proof: the reference was computed by the harness,
the rounding direction was my choice, and coverage was enumeration or sampling.

The point of camada's `mkFXPSqrt` is that the reference lives **inside the
solver**, so the comparison becomes a proof over all inputs of the format and
the harness never computes an expected value. That is now wired up:

* new irep2 nodes `fixedbv_sqrt2t` / `fixedbv_exp2t` -- TR 18037 operations, so
  distinct from the `ieee_*` family and carrying no rounding mode;
* frontend intrinsics `__ESBMC_fxp_sqrt_*` / `__ESBMC_fxp_exp_*`, one per
  format since C has no generic fixed-point parameter;
* lowered to `mkFXPSqrt` / `mkFXPExp` in `smt_solver.cpp`.

The `sqrt`/`abs` user-definition guard stays as it is: libc's body must remain
a call so that it is the thing being verified, not something replaced by the
solver's own operation.

### The oracle was validated before being trusted

An unvalidated oracle would invalidate every result downstream, so
`mkFXPSqrt` was pinned by its defining property first -- no reference
implementation involved:

```
raw_r^2 <= raw_x * 2^F < (raw_r+1)^2
```

which has exactly one solution per input, so proving it proves the oracle
computes the intended function. **VERIFICATION SUCCESSFUL at u0.8 and u0.16**,
and the negated form FAILS, so the proof is not vacuous.

Two corrections came out of doing this:

1. **`mkFXPSqrt` truncates toward zero; it is NOT round-to-nearest.** Camada
   documents it as "square root, rounded toward zero" (`camada.h:938`). Earlier
   text in this file and in the sqrt report described it as "correctly
   rounded" -- that description belongs to `mkFXPExp` (nearest, ties to even),
   not to sqrt. A harness that compares a library against this oracle must
   account for the direction or it reports a 1-ulp "error" that is the
   oracle's own.
2. My first two bracket attempts computed `r*r` **in the fixed-point type**,
   where it rounds to F fractional bits and is useless as a bracket. The
   bracket has to be evaluated on raw integers at full width. Both early
   failures were my harness, not the oracle.

### What the proof shows: an asymmetric, exactly-1-ulp defect

Against the truncating oracle the exact root lies in `[rb, rb+1)`, so libc's
documented "absolute errors < 2^(-fraction length)" permits exactly
`{rb, rb+1}`. Proved separately per direction so neither masks the other:

| property | u0.8 | u0.16 |
|---|---|---|
| libc <= exact + 1 ulp | **SUCCESSFUL** | -- |
| libc >= exact (never below) | **FAILED** | **FAILED** |
| libc >= exact - 1 ulp | **SUCCESSFUL** | -- |

So the error is two-sided and bounded by 1 ulp in each direction, but the claim
is *strictly* under one ulp, and the downward miss reaches a full ulp.

The sharpest witnesses are **exact perfect squares**, where no rounding
question exists at all:

```
sqrt(81/256)  = 9/16  = 0.5625     exactly representable; libc gives 143/256 = 0.5585938
sqrt(100/256) = 10/16 = 0.6250     exactly representable; libc gives 159/256 = 0.6210938
```

Both are one ulp low. Verified independently by running libc natively, so this
does not rest on the solver alone.

This supersedes the "2.48x the bound" framing: that ratio depended on my choice
of reference and rounding direction, whereas "one ulp low on an exactly
representable perfect square" is a claim about libc alone.
