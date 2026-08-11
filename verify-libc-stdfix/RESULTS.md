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
| `exp` | 2 | **2 BUGS** -- reachable saturated table entry in `exphk` only (36% error); spurious flush to zero in BOTH; relative bound exceeded on 74% (exphk) / 83% (expk) of in-band inputs |

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

### isqrt (uhksqrtus / uksqrtui): the same defect, proved

The violated claim sits on `isqrt` -- *"Integer square root - Accurate version:
Absolute errors < 2^(-fraction length)"* (sqrt.h:211-212). `uhksqrtus` and
`uksqrtui` call it, so the bound is violated on its own entry points.

isqrt takes an integer and returns an `_Accum`, so mkFXPSqrt of the same format
is not the reference. The property is stated directly, with nothing computed by
the harness: `r` is scaled by `2^F`, so `r^2` carries scale `2^(2F)` and

```
rb^2 <= n * 2^16 < (rb+1)^2          (F = 8 for u8.8)
```

evaluated on raw integers at full width. Proved separately per direction:

| property | verdict |
|---|---|
| `rb^2 <= n * 2^16` (never above the true root) | **SUCCESSFUL** |
| `n * 2^16 < (rb+1)^2` (within 1 ulp) | **FAILED** |

Exhaustive native confirmation over all 65536 inputs:

* **28228 inputs (43.07%)** exceed the documented 1-ulp bound
* worst error ~2 ulp, at n = 64189
* **153 of the 256 exact perfect squares are wrong**, every one 1 ulp low:

```
sqrt(25)  = 5  exactly -> libc  4.99609  (raw 1279, want 1280)
sqrt(49)  = 7  exactly -> libc  6.99609  (raw 1791, want 1792)
sqrt(100) = 10 exactly -> libc  9.99609  (raw 2559, want 2560)
sqrt(196) = 14 exactly -> libc 13.99609  (raw 3583, want 3584)
```

Perfect squares are the sharpest possible witnesses: the answer is exactly
representable, so no rounding-direction argument can excuse the miss. The
error is consistently *downward*, which matches a truncating rescale
(`r >>= (shift >> 1)`) rather than an approximation that is merely imprecise.

#### A harness bug caught on the way

The first run of this bracket "failed" for a reason that was mine: I read
`rb = 177` off the counterexample when the value was 45312, and briefly
concluded the scaling was wrong. It was not -- `rb/256 = 177.0` and the bracket
was correct all along. Re-deriving the scaling from native ground truth
(`isqrt(4) = 2`, `isqrt(65535) = 255.992`) settled it. Worth recording because
the misreading pointed at the harness when the defect was real.

## exp: two defects, proved against the correctly-rounded oracle

camada v0.17 added `mkFXPExp` -- exp correctly rounded to nearest, ties to even,
saturating at MAX and flushing below half an ulp. The pin in
`scripts/cmake/Options.cmake` is bumped to v0.17 accordingly.

### The oracle was validated first

Eight natively-measured anchors spanning the whole range, including both
boundary behaviours, all **SUCCESSFUL**:

```
raw    0 ->   128    exp(0) = 1
raw  128 ->   348    exp(1) = 2.7182818 -> 2.7187500
raw -128 ->    47    exp(-1) = 0.3678794 -> 0.3671875
raw  256 ->   946    exp(2) = 7.3890561 -> 7.3906250
raw  640 -> 18997    exp(5) = 148.4131591 -> 148.4140625
raw  704 -> 31321    exp(5.5) = 244.6919323 -> 244.6953125
raw  710 -> 32767    saturates (true value 256.43 exceeds MAX)
raw -800 ->     0    flushes (true value 0.00193 is below half an ulp)
```

Negating an anchor FAILS, so the proof is not vacuous. Full monotonicity over
all inputs was not proved -- two exp terms with a symbolic index does not
finish in reasonable time -- so the oracle rests on these anchors plus
non-vacuity, which is stated rather than glossed.

### What exphk claims

The bound is **relative**, and libc states it for one step of the range
reduction rather than end to end (`exphk.cpp`):

```
exp(x) = exp(hi)*exp(mid)*exp(lo) ~ exp(hi)*exp(mid)*(1 + lo)
   "with relative errors < |lo|^2 <= 2^-8"
```

Measured exhaustively over the 1419 inputs whose true value is strictly inside
the representable band:

* **1045 (73.64%)** exceed 2^-8 relative
* worst relative error **1.0** (100%) at x = -5.5390625
* worst absolute error **11840 ulp** at x = 5.5234375

A typical mid-range case, verified independently of the solver:
`x = 1.5859375`, `exp(x) = 4.883867859`, libc returns 4.84375 -- 5 ulp off, a
relative error of 0.008214 against the claimed 0.003906, i.e. **2.10x**.

### Defect A: the saturated table entry is reachable

`EXP_HI[11]` is `SACCUM_MAX`, a placeholder rather than `exp(6) = 403.4287935`,
which is not representable in s8.7. libc's own comment says so:

```
// Notice that when i = 88 and 89, e_hi will overflow short accum range.
```

But nothing excludes those indices. The guard rejects `x >= 0x1.64p2 = 5.5625`,
while index 88 needs only `x_rounded = 5.5` -- which every `x` in
`[5.4375, 5.5625)` rounds to. So the placeholder is multiplied by
`exp_mid * (1 + lo)`, with `exp_mid = 0.609` at that index, and the result
collapses:

```
x = 5.4375   exp = 229.8668   libc = 145.9922
x = 5.5000   exp = 244.6919   libc = 155.9922     (36% error)
x = 5.5469   exp = 256.4349   libc = 161.9922
```

Proof: `harness_exphk_saturated_entry.cpp` -- **FAILED**.

### Defect B: a representable value is flushed to zero

At index 0 the entry is `EXP_HI[0] = 0.0078125`, exactly one ulp, and `lo` can
be negative -- so `exp_hi * (exp_mid * (1 + lo))` underflows to zero. libc's
guard only flushes `x <= -0x1.63p2 = -5.5625`, so these inputs are inside the
supported domain, and their true value does *not* round to zero:

```
exp(-5.5390625) = 0.003930210     half an ulp = 0.003906250
```

which is above half an ulp, so the correctly-rounded result is raw 1. libc
returns 0. The window runs from about x = -5.5547 up to -5.3906.

This one is narrow, and it was worth double-checking rather than asserting:
the first reading was that flushing must be wrong because the value is
"representable", but what actually matters is whether it rounds to zero. It
does not, by a margin of 0.000024.

Proof: `harness_exphk_flush_zero.cpp` -- **FAILED**.

### expk (s16.15): one defect recurs, the other does not

Both entry points are now covered, and they differ in a way worth stating
because it would have been easy to assume the pattern generalised:

| defect | `exphk` (s8.7) | `expk` (s16.15) |
|---|---|---|
| A: reachable saturated table entry | **YES** -- index 88 hits `EXP_HI[11]` | **NO** -- max index reached is 22, `EXP_HI[23]` is unreachable |
| B: spurious flush to zero | **YES** | **YES** |
| in-band inputs over the relative bound | 73.64% | 83.13% (sampled) |

`expk`'s `EXP_HI[23]` is also an `ACCUM_MAX` placeholder -- `exp(12) = 162754.79`
does not fit s16.15 -- but it is never selected: the largest index reached is 22,
consistent with libc's own "indices <= 355" comment (355 >> 4 = 22). So the
reachable placeholder is specific to `exphk`'s narrower table, not a shared
design flaw. `expk` also uses a quadratic term (`1 + lo + lo^2/2`) where `exphk`
uses only `1 + lo`, and claims the correspondingly tighter `2^-16`.

Defect B does recur, from the same cause: `EXP_HI[0] = 0x1p-15` is exactly one
ulp and `lo` can be negative, so the product underflows. The guard flushes only
`x <= -11.0903320`, while

```
exp(-11.0898132) = 0.0000152671     half an ulp = 0.0000152588
```

is above half an ulp, so the correctly-rounded answer is raw 1. Verified at four
natively-measured inputs in that window.

Proof: `harness_expk_flush_zero.cpp` -- **FAILED**.

#### Symbolic coverage achieved, by changing solver

The pinned-input version is superseded. The full symbolic window (3392 inputs)
does discharge -- but only under **z3**:

| harness | window | solver | time | verdict |
|---|---|---|---|---|
| `harness_expk_flush_symbolic` | 3392 inputs | bitwuzla | **>90 min, no verdict** | -- |
| `harness_expk_flush_symbolic` | 3392 inputs | **z3** | **40.64s** | **FAILED** |
| `harness_expk_flush_narrow` | 32 inputs | bitwuzla | 2.70s | FAILED |

Worth recording as a solver-portfolio result rather than a limit of the
approach: correct rounding at s16.15 needs a 37-bit intermediate, and the two
backends differ by more than three orders of magnitude on that query. The
earlier "does not finish in reasonable time" conclusion was true of bitwuzla
only, and I should have tried the other backend before writing it down.

Both counterexamples were confirmed against native libc:

```
bitwuzla (narrow): x = -11.0888672   exp = 0.000015281506   libc 0, want raw 1
z3 (full window):  x = -11.0779419   exp = 0.000015449377   libc 0, want raw 1
                                     half ulp = 0.000015258789
```

The native trace also confirms the mechanism directly: at `idx = 0`,
`EXP_HI[0] = 0.0000305176` (one ulp) and `l2 = 0.974 < 1`, so the product
underflows to zero.

### Scope

Both exp entry points are covered. Nothing here has been reported upstream.

## An ESBMC bug that invalidated the earlier isqrt results

Confirming counterexamples concretely -- rather than trusting the verdicts --
turned up a defect in **ESBMC**, not libc: `__ESBMC_bitcast` between two
fixed-point formats of the same storage width **rescaled instead of
reinterpreting the bits**. A u0.32 raw pattern read back as u16.16 came out
shifted by the 16-bit difference in fraction length (0xB505A837 -> 0x0000B505).

`smt_bitcast.cpp`'s `is_fixedbv_type(to_type)` branch only handled a bitvector
source, so fixedbv->fixedbv fell through to a value-preserving typecast. Fixed
by routing it through `mkFXPToRawBV` + `mkFXPFromRawBV` when the widths match.

This mattered because LLVM libc's `isqrt` **ends** in exactly that cast
(`bit_cast<OutType>` of a `FracType` result), so every isqrt verdict before the
fix was computed on a corrupted value. After the fix, verdicts changed in both
directions -- harnesses that had passed began failing. Nothing that rests on
`isqrt` should be read from the pre-fix commits.

How it was caught: ESBMC reported a counterexample at `n = 2147549183`, and
re-running that input through native libc showed **both brackets holding** and
an error of 0.87 ulp, inside the bound. The cex did not reproduce, which is the
signal that pointed at the tool rather than the library.

## Two harness bugs of my own, found the same way

1. **The lower bracket forbade rounding up.** It asserted `rb <= floor(root)`,
   but libc claims an error *bound*, not truncation. At `n = 65534`,
   `sqrt(65534)*256 = 65534.99999` and libc correctly returns 65535 -- off the
   true root by 0.0000038 ulp. My bracket called that a defect. Restated
   direction-agnostically as `(rb-1)^2 < n*2^(2F) < (rb+1)^2`.
2. **Saturation was tested on the wrong value.** The first guard checked
   `rb == MAX`, but at `n = 65535` libc returns 65534, not MAX, so the guard
   missed it. The test has to be on the *true root* against the format maximum.

Both produced false positives that native confirmation caught.

## sqrt/isqrt: final verdicts, all counterexamples confirmed

Every counterexample below was re-run through native libc and reproduces
exactly. Saturation is excluded (where the true root exceeds the format
maximum, clamping is correct and no error bound applies).

| harness | format | verdict | ESBMC | witness, confirmed natively |
|---|---|---|---|---|
| `oracle_sqrt_exact.c` | u0.8 | SUCCESSFUL | 0.50s | oracle validated |
| `oracle_sqrt_ur.c` | u0.16 | SUCCESSFUL | 14.72s | oracle validated |
| `harness_sqrt_vs_oracle_high` | u0.8 | **SUCCESSFUL** | 0.60s | never >1 ulp above |
| `harness_sqrt_vs_oracle_low` | u0.8 | FAILED | 0.60s | `sqrt(81/256)=9/16` exact, libc 1 ulp low |
| `harness_sqrt_vs_oracle_ur` | u0.16 | FAILED | 0.60s | same defect at 16 bits |
| `harness_isqrt_bound` | u8.8 | FAILED | 0.70s | `n=32045`: **-1.86 ulp**, not saturating |
| `harness_isqrt_uk_bound` | u16.16 | FAILED | 0.70s | `n=2147483649`: **-1.68 ulp**, not saturating |

The two isqrt witnesses are the strongest form of the finding: both are far
from the saturation rail, so no clamping argument applies, and both exceed the
documented "< 1 ulp" by well over half an ulp again.

## ESBMC vs native libc: time to reach the same conclusion

Native sweeps check the identical property over the identical domain, compiled
`-O2`. The comparison is not about raw throughput -- it is about what each
approach can conclude.

| domain | inputs | native sweep | ESBMC | note |
|---|---|---|---|---|
| u0.8 sqrt | 2^8 | 0.00s | 0.60s | native wins; the domain is trivial |
| u0.16 sqrt | 2^16 | 0.00s | 0.60s | native wins |
| u8.8 isqrt | 2^16 | 0.00s | 0.70s | native wins |
| **u0.32 sqrt** | **2^32** | **90.20s** | **0.60s** | **ESBMC 150x faster** |
| **u16.16 isqrt** | **2^32** | **91.10s** | **0.70s** | **ESBMC 130x faster** |
| exphk saturated entry | s8.7 window | 0.00s | 59.17s | native wins |
| exphk flush-to-zero | s8.7 window | 0.00s | 1.50s | native wins |
| expk flush-to-zero | 4 inputs | 0.00s | 1.60s | native wins |

Two honest observations:

* **Below 2^16, enumeration beats BMC outright.** Sweeping 65536 inputs is
  free; encoding the same question for a solver is not. The 0.6-0.7s floor is
  parse and encode cost, not solving.
* **At 2^32 the crossover is decisive**, and it is not merely a constant
  factor: BMC does not enumerate, so `u0.32` costs ESBMC the same 0.6s as
  `u0.8`, while the native sweep grows linearly with the domain. Extending to
  a 64-bit format is another 4-billion-fold for enumeration and roughly free
  for BMC -- which is why the 64-bit `_Accum` formats are reachable at all.

The other difference is not a time at all: a native sweep answers "no
violation was found among the inputs I tried", while a SUCCESSFUL verdict
answers "no violation exists". `harness_sqrt_vs_oracle_high` is the only
positive result here, and only BMC could have produced it.

### exp timings

| harness | verdict | ESBMC |
|---|---|---|
| `oracle_exp_conc.c` (8 anchors) | SUCCESSFUL | 0.40s |
| `harness_exphk_saturated_entry` | FAILED | 59.17s |
| `harness_exphk_flush_zero` | FAILED | 1.50s |
| `harness_expk_flush_zero` | FAILED | 1.60s |

exp is where BMC is *most* expensive relative to native, because correct
rounding needs a wide intermediate: 19 bits at s8.7 and 37 at s16.15 (camada's
measured hardest-to-round bounds). The 59s case is the symbolic window sweep;
the flush cases pin fewer inputs.

## Complete sqrt/exp coverage: all 7 entry points proved in-solver

The last gap was `sqrtulr` (u0.32). Validating `mkFXPSqrt` at that width did not
discharge under either backend (bitwuzla >50 min, z3 >10 min) -- bracketing a
32-bit symbolic root with 128-bit products is the expensive part. But the
oracle is not needed to check libc's own claim: the bound is stated directly on
the result, the same shape that proved `uksqrtui` in 0.7s. That closes u0.32 in
**4.81s**.

Lesson worth keeping: the oracle is the right tool for questions libc does not
answer itself (what IS the correct value), but where the library states a bound,
asserting the bound directly is both sufficient and far cheaper.

| entry point | format | verdict | solver | time | witness (confirmed natively) |
|---|---|---|---|---|---|
| `sqrtuhr` | u0.8 | FAILED | bitwuzla | 0.60s | `sqrt(81/256)=9/16` exact, 1 ulp low |
| `sqrtur` | u0.16 | FAILED | bitwuzla | 0.60s | same defect |
| `sqrtulr` | u0.32 | FAILED | bitwuzla | 4.81s | `xb=2973817639`: **-1.039 ulp** |
| `uhksqrtus` | u8.8 | FAILED | bitwuzla | 0.70s | `n=32045`: **-1.86 ulp** |
| `uksqrtui` | u16.16 | FAILED | bitwuzla | 0.70s | `n=2147483649`: **-1.68 ulp** |
| `exphk` | s8.7 | FAILED | bitwuzla | 59.17s | saturated table entry + flush |
| `expk` | s16.15 | FAILED | **z3** | 40.64s | `x=-11.0779419`: flush, want raw 1 |

Every counterexample in this table was re-run through native libc and
reproduces. The one positive result -- `harness_sqrt_vs_oracle_high`, "libc is
never more than 1 ulp above the exact root at u0.8", SUCCESSFUL in 0.60s -- is
the only claim here that enumeration could not have established.

## Correction: the earlier coverage was libc's formats, not camada's

The tables above cover the formats **stdfix.h instantiates**, which is a strictly
smaller set than what camada's operations support. Stating "all 7 entry points"
was accurate about libc and misleading about the operations:

* **`mkFXPSqrt` is format-generic** -- an exact integer digit recurrence with no
  allowlist, valid at any width and signedness. All **12** TR 18037 formats are
  in scope; I had exercised **5**, all unsigned. Every signed format and both
  64-bit formats were untested.
* **`mkFXPExp` supports 6 formats** -- (16,7)s (16,8)u (32,15)s (32,16)u
  (64,31)s (64,32)u, i.e. every C `_Accum` type. I had exercised **2**, the
  signed narrow ones libc has entry points for.

The intrinsic surface is now declared for the full set: 12 `__ESBMC_fxp_sqrt_*`
and 6 `__ESBMC_fxp_exp_*`.

### mkFXPSqrt: all 12 formats

| format | type | method | verdict | time |
|---|---|---|---|---|
| s0.7 | `short _Fract` | symbolic bracket, all inputs | **SUCCESSFUL** | 0.50s |
| u0.8 | `unsigned short _Fract` | symbolic bracket, all inputs | **SUCCESSFUL** | 0.50s |
| s0.15 | `_Fract` | symbolic bracket, all inputs | **SUCCESSFUL** | 6.71s |
| u0.16 | `unsigned _Fract` | symbolic bracket, all inputs | **SUCCESSFUL** | 14.61s |
| s8.7 | `short _Accum` | symbolic bracket, all inputs | **SUCCESSFUL** | 1.40s |
| u8.8 | `unsigned short _Accum` | symbolic bracket, all inputs | **SUCCESSFUL** | 2.30s |
| s0.31 | `long _Fract` | anchors | SUCCESSFUL | 0.60s |
| u0.32 | `unsigned long _Fract` | anchors | SUCCESSFUL | 0.60s |
| s16.15 | `_Accum` | anchors | SUCCESSFUL | 0.60s |
| u16.16 | `unsigned _Accum` | anchors | SUCCESSFUL | 0.50s |
| s32.31 | `long _Accum` | anchors | SUCCESSFUL | 0.60s |
| u32.32 | `unsigned long _Accum` | anchors | SUCCESSFUL | 0.50s |

The signed formats also confirm the documented negative-operand behaviour
(no real square root, result zero), which none of the earlier unsigned-only
work touched.

**Where the symbolic bracket stops scaling, and why.** The bracket squares a
symbolic root, so at 32 bits it asks the solver about a 64-bit product of an
unknown, and at 64 bits a 128-bit one. None of the six wide formats discharged
in 40 minutes under bitwuzla, and u0.32 also failed under z3 (>10 min). The
anchor rows are therefore **validation on specific inputs, not proofs over all
inputs**, and are labelled as such rather than folded in with the six that are.

Anchors are perfect squares whose roots are exactly representable (0.25 -> 0.5,
2^-4 -> 2^-2), so no rounding question arises and a wrong wiring -- identity,
wrong sort, wrong scale -- still fails them.

### mkFXPExp: all 6 supported formats

Expected values computed natively in `long double`, rounded to nearest with ties
to even per camada's contract. None guessed.

| format | type | verdict | time | note |
|---|---|---|---|---|
| s8.7 | `short _Accum` | **SUCCESSFUL** | 0.40s | 8 anchors incl. saturation + flush |
| u8.8 | `unsigned short _Accum` | **SUCCESSFUL** | 0.50s | newly covered |
| s16.15 | `_Accum` | **SUCCESSFUL** | 0.50s | newly covered |
| u16.16 | `unsigned _Accum` | **SUCCESSFUL** | 0.50s | newly covered |
| s32.31 | `long _Accum` | **SUCCESSFUL** | 0.50s | newly covered, incl. negative arm |
| u32.32 | `unsigned long _Accum` | **SUCCESSFUL** | 0.60s | newly covered |

So `mkFXPExp` is validated on **every format it supports**, both 64-bit formats
included. The four newly covered ones have no stdfix.h entry point, so they
exercise the operation alone -- there is no library implementation to compare
against at those formats.

### What this does not change

The libc defects stand exactly as reported: they are findings about the formats
libc actually ships, and every counterexample was confirmed natively. What
changes is the coverage claim about the *operations* -- previously 5 of 12 sqrt
formats and 2 of 6 exp formats, now 12 and 6.

## Native exhaustive coverage: what was actually run

Asked directly whether the formats were tested exhaustively by running the
binary, the honest answer is **5 of 12**, and it is worth being precise about
why -- because the reason is not laziness in three of the seven gaps.

| format | width | domain | exhaustive native run? |
|---|---|---|---|
| u0.8 | 8 | 256 | **YES** |
| u0.16 | 16 | 65,536 | **YES** |
| u8.8 | 16 | 65,536 | **YES** |
| u0.32 | 32 | 4,294,967,296 | **YES** (90.20s) |
| u16.16 | 32 | 4,294,967,296 | **YES** (91.10s) |
| s0.7, s0.15, s8.7, s0.31, s16.15 | 8-32 | feasible | **no -- libc has no signed sqrt** |
| s32.31, u32.32 | 64 | 1.8e19 | **no -- 2^64, about 12,000 years** |

### libc's sqrt is unsigned-only, so five of the gaps are not gaps

Attempting the signed sweeps does not produce wrong numbers -- it **does not
compile**:

```
sqrt.h:198: implicit instantiation of undefined template
            'SqrtConfig<short _Fract>'
```

`sqrt.h` specialises `SqrtConfig` for exactly five fixed-point formats
(u0.8, u0.16, u0.32, u8.8, u16.16) plus two integer entries (`unsigned short`,
`unsigned int`). There is no signed specialisation and no 64-bit one. So the
five exhaustive sweeps that were run cover **every fixed-point format libc's
sqrt implements**, and the signed rows above are unimplementable rather than
untested.

This also corrects the framing of the 12-format sqrt table earlier in this
file: camada supports all 12 and ESBMC now exercises all 12, but only 5 have a
libc counterpart to compare against. The other 7 validate the *operation*, not
any library.

### A declared-but-missing entry point

`stdfix.yaml` declares **`sqrtulk`** (u32.32):

```yaml
  - name: sqrtulk
    return_type: unsigned long accum
    arguments:
      - type: unsigned long accum
```

but there is no `sqrtulk.cpp`, no CMake entry, and `SqrtConfig<unsigned long
_Accum>` is undefined -- so `fx::sqrt` on a u32.32 value fails to compile.
A caller reading the public spec would reasonably expect the function to exist.

Recorded as an observation rather than a defect claim: the yaml may be an
intentional forward declaration of planned work. It is worth asking, since the
declaration is visible to consumers while the symbol is not.

### The 64-bit domains are out of reach for enumeration, and that is the point

At the measured native rate (4.29e9 inputs in 91s, ~47M/sec) a single 2^64
sweep is about **12,000 years**. This is exactly the case BMC exists for -- and
it is also where the symbolic bracket is hardest, which is why s32.31 and
u32.32 are anchor-validated rather than proved and the racing runs are still
open after 14h. Neither method covers those two formats exhaustively today;
saying so is more useful than picking whichever framing sounds better.
