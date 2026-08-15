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
| s16.15 | `_Accum` | **symbolic bracket, all 2^32 inputs** | **SUCCESSFUL** | **65086s (18h05m, bitwuzla)** |
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

## s16.15 upgraded from anchor-validated to proved (18h05m)

The racing runs produced their first result: **`k` (s16.15) verified
SUCCESSFUL under bitwuzla after 65,086s** -- 18 hours 5 minutes in the decision
procedure. z3 was still working on the same query at 22h21m.

The verdict was checked rather than taken at face value:

* the log shows 3 VCCs generated and 3 remaining after simplification, so
  nothing was sliced away, and the run ended normally rather than crashing;
* the harness contains **no `__ESBMC_assume` at all** -- the input is a free
  32-bit pattern -- so vacuity is structurally impossible;
* both arms were probed anyway. Negating the negative-operand assertion FAILS
  in 0.6s, and a reachability probe on the positive arm (where the 18 hours
  went) also FAILS in 0.6s. Both arms are reachable.

So `mkFXPSqrt` at s16.15 is proved over every input of the format: the bracket
`raw_r^2 <= raw_x * 2^15 < (raw_r+1)^2` holds for all 2^31 non-negative inputs,
and negative inputs return zero as documented.

Note this is a format libc's sqrt cannot even instantiate (`SqrtConfig` is
unsigned-only), so it validates the operation rather than any library.

### Revised expectations for the remaining five

| format | frac bits | bitwuzla | z3 | note |
|---|---|---|---|---|
| s16.15 | 15 | **SUCCESSFUL 18h05m** | still running | settled |
| u16.16 | 16 | running 22h+ | running 22h+ | next most likely |
| s0.31 | 31 | running 22h+ | running 22h+ | ~126-bit product |
| u0.32 | 32 | running 22h+ | running 22h+ | ~128-bit product |
| s32.31 | 31 | running 22h+ | running 22h+ | 64-bit storage |
| u32.32 | 32 | running 22h+ | running 22h+ | 64-bit storage |

That s16.15 took 18h and u16.16 has not landed at 22h is consistent with the
one-extra-fraction-bit cost. The four formats with 31-32 fraction bits square a
root into ~126-128 bits rather than ~62, and nothing suggests they are close.

Two earlier predictions of mine to correct:

1. I wrote that a plateau in z3's memory suggested "neither backend looks close
   to closing these queries". bitwuzla closed one 2.5 hours later, on flat
   0.2 GB memory the whole time -- so flat memory says nothing about progress
   for a bit-blasting solver.
2. I projected z3 would exhaust the box's memory in ~24h by fitting an
   exponential through two samples. Growth flattened to ~0.5 GB/h; at 22h the
   box is at 106 GB of 247 GB with 141 GB free. Time, not memory, is the
   constraint.

   **Corrected again at 33h18m**: the flattening was a pause, not a ceiling.
   `lk` and `ulk` doubled again (17.0 -> 34.1 GB) and `ulr` nearly did
   (15.7 -> 30.9), taking the z3 total from 71 GB to 123 GB. Sampling twice
   three minutes apart shows near-zero instantaneous growth (<=0.02 GB), so z3
   grows in **steps** -- long flat stretches punctuated by jumps, presumably on
   search restarts with a larger structure. The average rate is real but the
   timing of the next step is not predictable from the flat stretches, which is
   what fooled both of my earlier projections in opposite directions.

   At 91 GB free that is roughly two more doublings of headroom. Swap is only
   1 GB, so a step past the ceiling OOMs cleanly rather than thrashing, and the
   z3 processes carry `oom_score_adj=800` against bitwuzla's 200 -- so the
   kernel sacrifices a z3 run and leaves all six bitwuzla runs, which is the
   right trade given bitwuzla produced the only result so far.

## Correction: uksqrtui/uhksqrtus call isqrt_fast, not isqrt

`sqrt.h` declares **two** integer square roots with **different** documented
claims, and I tested the shipped entry points against the wrong one:

```
// Integer square root - Accurate version:
// Absolute errors < 2^(-fraction length).        <- isqrt      (sqrt.h:211)

// Integer square root - Fast but less accurate version:
// Relative errors < 2^(-fraction length).        <- isqrt_fast (sqrt.h:236)
```

`uksqrtui.cpp:18` and `uhksqrtus.cpp:18` both `return fixed_point::isqrt_fast(x)`.
My harnesses called `fx::isqrt` and asserted the **absolute** bound. So the
earlier rows describing those two entry points were checking a function the
entry points do not call, against a claim they do not make.

Re-measured against the correct function and its own relative claim, both
still violate -- by more than the earlier numbers suggested:

| entry point | calls | claim | measured | ratio |
|---|---|---|---|---|
| `uhksqrtus` | `isqrt_fast` | relative < 2^-8 = 3.906e-3 | **5.649e-3** | **1.45x** |
| `uksqrtui` | `isqrt_fast` | relative < 2^-16 = 1.526e-5 | **2.050e-5** | **1.34x** |

At `n = 32045`, `isqrt_fast` is **258.9 ulp** below the true root where `isqrt`
is 1.86 ulp below -- consistent with "fast but less accurate", and still outside
the relative bound it claims for itself.

So there are two distinct findings, not one:

1. **`isqrt`** violates its absolute < 1 ulp claim (-1.86 ulp at u8.8,
   -1.68 ulp at u16.16). No shipped stdfix entry point calls it, so this is a
   defect in an internal function that `sqrt.h` documents and exposes.
2. **`isqrt_fast`** violates its relative < 2^-F claim (1.45x at u8.8, 1.34x at
   u16.16). This one **is** on the shipped path for `uhksqrtus`/`uksqrtui`.

The `fixed_point::sqrt` results (sqrtuhr/sqrtur/sqrtulr, 1 ulp low on exact
perfect squares) are unaffected -- those entry points do call `sqrt`.

## Parallel sharding: the wide-format proofs are tractable after all

The six 32/64-bit formats had been running 41h unsharded with one result
(s16.15 at 18h05m). Partitioning the input domain with `__ESBMC_assume` and
running the shards concurrently changes the picture completely.

The partition is a **complete case split**, not sampling: 32 shards of
134,217,728 consecutive raw values each, verified programmatically to cover
`[INT32_MIN, INT32_MAX]` with no gap and no overlap, summing to exactly 2^32.
If every shard verifies SUCCESSFUL the property holds over the whole domain.

Measured on s16.15, the format whose unsharded time is known:

| shard set | inputs each | time | note |
|---|---|---|---|
| 00-15 (negative half) | 134M | **0.5-1.1s** | `xb < 0` early return, trivial |
| 31 (positive, top of range) | 134M | **735.78s (12m)** | real bracket work |
| unsharded whole domain | 4.29e9 | **65086s (18h05m)** | for comparison |

So one positive shard is ~12 minutes where the whole domain was 18 hours. With
16 positive shards running concurrently the wall-clock for a full format proof
drops from 18h to roughly the slowest single shard -- a **~90x** improvement,
and it parallelises across the 32 cores that were previously idle behind one
sequential query.

Why it works: the assume prunes the search space before the solver starts, so
each shard is a genuinely smaller problem rather than the same problem with a
filter. The negative-half shards collapse to nothing because the early return
makes the bracket unreachable.

This is the technique that makes the two 64-bit formats (s32.31, u32.32)
plausibly reachable at all -- they were the ones with no path to exhaustive
native testing either (2^64 is ~12,000 years of enumeration).

## DISPATCH AUDIT: what each entry point actually calls

Read from every `libc/src/stdfix/*.cpp`. This should have been done first; two
substantial findings in this file were aimed at the wrong function.

| family | entry points | calls | tested? |
|---|---|---|---|
| `abs*` | 6 | `fixed_point::abs` | yes |
| `bits*` | 12 | `fixed_point::bitsfx` | yes |
| `*bits` | 12 | (inline bit_cast) | yes |
| `countls*` | 12 | `fixed_point::countls` | yes |
| `round*` | 12 | `fixed_point::round` | yes |
| `idiv*` | 8 | `fixed_point::idiv` | yes |
| **`divi*`** | **8** | **`fixed_point::divifx`** | **NO -- tested `divi`** |
| `rdivi` | 1 | `fixed_point::divi` | yes |
| `sqrtuhr`, `sqrtur`, `sqrtulr` | 3 | `fixed_point::sqrt` | yes |
| **`sqrtuhk`, `sqrtuk`** | **2** | **`fixed_point::sqrt`** | **NO -- never tested** |
| `uhksqrtus`, `uksqrtui` | 2 | `fixed_point::isqrt_fast` | corrected |
| `exphk`, `expk` | 2 | (inline bodies) | yes |

### Error 1: the divi findings target a function only rdivi calls

`fx_bits.h` declares two unrelated functions:

```
line 241  divi(int n, int d) -> XType            integer / integer -> fixed-point
line 342  divifx(IntType n, FXType d) -> IntType integer / fixed-point -> integer
```

All eight `divi*` entry points call **`divifx`**:

```cpp
LLVM_LIBC_FUNCTION(int, divir, (int n, fract d)) {
  return fixed_point::divifx<int, fract>(n, d);
}
```

`stdfix.yaml` confirms the signature: `divir(int, fract) -> int`.

The three defects recorded earlier under "divi, by root cause" -- the double
sign negation, the `1 << F` UB, and the missing intermediate headroom -- are all
in **`divi`**, which only `rdivi` reaches. So:

* they remain real defects in `divi`, and `rdivi` is a genuinely affected
  shipped entry point (one entry point, not eight);
* `REPORT-llvm-libc-divi-defects.md` overstates the blast radius and must be
  rewritten to say `rdivi` rather than the `divi*` family;
* **`divifx` is untested.** Eight shipped entry points have had no verification
  at all.

### Error 2: sqrtuhk and sqrtuk were never tested

Both call `fixed_point::sqrt` on `_Accum` formats (u8.8, u16.16). The five
sqrt entry points tested were `sqrtuhr`/`sqrtur`/`sqrtulr` plus the two
`isqrt_fast` ones -- these two were missed entirely. Earlier claims of "all 5
sqrt entry points" were wrong: there are **7**.

### Cost of not doing this first

Roughly two days of solver time went into wide-format oracle validation for
`mkFXPSqrt` while two shipped sqrt entry points sat untested and the divi work
targeted the wrong function. The dispatch audit takes one command.

## Results after the dispatch audit

### sqrtuhk / sqrtuk: the two entry points that had never been tested

Both call `fixed_point::sqrt`, so the claim is sqrt.h:211's absolute < 1 ulp.
Compared against camada's `mkFXPSqrt` on the same symbolic input -- operand and
result share the format here, so the oracle applies with no rescaling.

| entry point | x_raw | libc | true root (raw) | error | saturating? |
|---|---|---|---|---|---|
| `sqrtuhk` | 24591 | 2508 | 2509.0428 | **-1.043 ulp** | no |
| `sqrtuk` | 2282042691 | 12229306 | 12229307.0040 | **-1.004 ulp** | no |

Both **FAILED**, both confirmed natively, neither near the saturation rail. So
all seven sqrt entry points violate the bound they claim.

### divifx: first verification, and one positive result

The eight `divi*` entry points call `divifx`, which had never been tested. Its
contract quotes no error term (fx_bits.h:337-338):

> "Divide an integer operand by a fixed-point operand and return the
>  mathematically exact result as an IntType rounded towards 0."

Exactness plus truncation is fully checkable, so the property is a
division-free bracket on `q * d_raw` against `n * 2^F`.

**`diviur` (u0.16): VERIFICATION SUCCESSFUL (0.60s)** -- the first `divifx`
result, and the first *positive* result on any divi-family function.

Two constraints on that claim, stated rather than buried:

* **`n` is bounded to [-1024, 1024]**; the divisor is fully symbolic over its
  whole format. Unbounded `n` did not finish in 500s -- the property multiplies
  128-bit values, and both operands symbolic at full width is too much. So this
  is a proof over a restricted domain, not the whole input space.
* **Return-type overflow is excluded.** With `d_raw = 1` the exact quotient is
  `n * 2^F`, which exceeds `int` for most `n`; TR 18037 (quoted at
  fx_bits.h:223 for the sibling `idiv`) makes an overflowing integer result
  undefined. The first version of this harness asserted the truncation identity
  everywhere and "failed" on all four formats -- that was the harness, not
  libc. `divir` at `n = -196608, d_raw = 1` returns `INT_MIN` for a true
  quotient of `-6442450944`.

`divir`, `divik`, `diviuk` are still running.

### Corrected scope of the divi defects

The three defects in `fixed_point::divi` (double sign negation, `1 << F` UB, no
intermediate headroom) are reachable only through **`rdivi`** -- the one entry
point that calls `divi`. `REPORT-llvm-libc-divi-defects.md` still describes them
as affecting the `divi*` family and needs rewriting.

## divifx against camada's mkFXPDiv: all 8 entry points agree

Same design as the sqrt oracle work: symbolic inputs, libc on one side, camada's
exact operation on the other, diffed.

| entry point | operand format | division done in | verdict | time |
|---|---|---|---|---|
| `divir` | s0.15 | s16.15 | **SUCCESSFUL** | 5.31s |
| `diviur` | u0.16 | u16.16 | **SUCCESSFUL** | 8.91s |
| `divilr` | s0.31 | s32.31 | **SUCCESSFUL** | 25.63s |
| `diviulr` | u0.32 | u32.32 | **SUCCESSFUL** | 26.13s |
| `divik` | s16.15 | s16.15 | **SUCCESSFUL** | 5.61s |
| `diviuk` | u16.16 | u16.16 | **SUCCESSFUL** | 6.91s |
| `divilk` | s32.31 | s32.31 | **SUCCESSFUL** | 35.34s |
| `diviulk` | u32.32 | u32.32 | **SUCCESSFUL** | 58.67s |

All eight non-vacuous: negating the agreement assertion FAILS in every case.

Both non-vacuous: negating the agreement assertion FAILS for each. `diviuk` also
passes the independent truncation-bracket harness (0.58s), so two different
properties agree on it.

**This is the first family where libc comes out clean.** `divifx` is what all
eight `divi*` entry points actually call, and on these formats it matches an
exact reference.

### Correction: the _Fract formats were excluded in error

An earlier version of this section claimed the `_Fract` entry points "cannot be
compared this way at all". **That was wrong.** `mkFXPDiv` is format-generic --
no allowlist, unlike `mkFXPExp` -- so the limitation was never camada's.

The real problem was that I tried to represent `n` in s0.15/u0.16 themselves,
which hold only [-1, 1); `(unsigned _Fract)64` came out as ~2.97e222. The fix is
to do the camada-side division in a format with the **same fraction length but
integer headroom**: s0.15 -> s16.15, u0.32 -> u32.32, and so on. Widening the
divisor that way is exact, since the fraction length is unchanged.

With that, all four `_Fract` entry points verify. The remaining bounds are:
* **The quotient must fit the format the division happens in.** With a tiny
  divisor the quotient exceeds the maximum and the camada side overflows -- at
  `n=64, d=2^-16` the true quotient is 2097152 against u16.16's max of 65536.
  Bounded by `n <= 16` and a divisor floor, giving a quotient <= 256. This is a
  C range constraint, not a solver or camada one.
* **Non-negative quotients only** for the signed format. camada's division
  floors (camada.h:756-758) while `divifx` truncates toward zero, so the two
  legitimately differ by one where the signs differ. Restricting to matching
  signs makes the comparison exact instead of approximate.

So this is agreement over a restricted domain, not a proof over all inputs --
but the domain restriction is now only on `n` and the quotient magnitude, not on
which formats can be tested. **`divifx` is clean on every entry point libc
ships**, which makes it the only stdfix family to come out fully verified
against an exact reference.

### Four false failures caught before they became claims

The first run of this comparison failed on all four formats, at `n=64` with
tiny divisors. Hand-checking showed **libc was right every time**:
`divifx(64, 0.5) = 128` and `divifx(64, 2^-16) = 2097152` are both exact. The
failures were the two representability problems above. Four-for-four failures on
a first attempt is the signal to check the harness, not the library.

## Which bound applies to which sqrt function -- and the one that has none

`sqrt.h` states three bounds, and they attach to three different functions:

| line | bound | function | entry points reaching it |
|---|---|---|---|
| 35/57/83 | `max(1.5 * 2^-11, eps)` | Sollya initial-approximation tables | (internal) |
| **165** | `\|r - sqrt(x_frac)\| < max(1.5 * 2^-11, eps)` | **`sqrt_core`** | all, indirectly |
| **212** | **Absolute** errors < 2^-F | **`isqrt`** | **none** |
| **237** | **Relative** errors < 2^-F | **`isqrt_fast`** | `uhksqrtus`, `uksqrtui` |

**`fixed_point::sqrt` has no accuracy comment at all.** It is declared at
sqrt.h:184 and the lines above it are a TODO about division-free Newton
iterations. Yet `sqrtuhr`, `sqrtur`, `sqrtulr`, `sqrtuhk` and `sqrtuk` -- five of
the seven entry points -- call it.

### Correction: those five were tested against isqrt's bound

Earlier sections of this file assert "Absolute errors < 2^(-fraction length)"
against the five `sqrt` entry points. **That is line 212, which documents
`isqrt`.** Having already corrected the opposite error -- testing the
`isqrt_fast` entry points against `isqrt`'s bound -- I made the same
misattribution a second time, in the other direction.

### What the two testable bounds show

`harness_sqrt_core_bound.cpp` -- **VERIFICATION SUCCESSFUL (11.01s)**.
`sqrt_core` honours line 165 over every normalised u0.16 input
(`x_frac >= 16384`, its documented domain). At 16 fraction bits the bound
`1.5 * 2^-11` is 48 ulp, so this is a generous window and the core sits inside
it.

`harness_sqrt_rescale_gap.cpp` -- **FAILED** against a one-ulp contract that is
**inferred, not quoted**, and labelled as such in the harness.

### The rescale theory was wrong

I had attributed the error to the truncating rescale at sqrt.h:205
(`r >>= EXP_ADJUSTMENT - (x_exp >> 1)`). The counterexample refutes it:

```
x_raw = 63211 (0.964523315)
  exact root * 2^16   = 64363.0025
  sqrt_core           = 64361      -2.0025 ulp   (bound 48 ulp: WITHIN)
  fixed_point::sqrt   = 64361      -2.0025 ulp   (no documented bound)
```

Both return the same value -- the rescale is a no-op at this input -- so the
2 ulp error originates in `sqrt_core` itself, well within its own bound.

### Restating the sqrt findings honestly

* **`uhksqrtus` / `uksqrtui`**: violate `isqrt_fast`'s relative bound, the claim
  their own function makes. 1.51x and 3.47x, confirmed natively. **Firm.**
* **The five `sqrt` entry points**: measurably 1-2 ulp from the exact root, and
  153 of 256 exact perfect squares come back 1 ulp low. But `fixed_point::sqrt`
  documents no end-to-end bound, and the one bound in its call chain
  (`sqrt_core`, 48 ulp at u0.16) is honoured. So this is a **documentation gap**
  -- a caller gets up to tens of ulp with nothing saying so -- **not** a proven
  bound violation. `REPORT-llvm-libc-sqrtfx-error-bound.md` claims the latter
  and must be rewritten.
* **`isqrt`**: its absolute bound is violated (-1.86 ulp at u8.8), but **no
  shipped entry point calls `isqrt`**, so this is a defect in a documented
  internal function with no current caller.

## Sharded whole-domain verification of exp against camada (bitwuzla)

Both exp entry points, 32 shards each, partitions verified exact
(`exphk`: 32 x 2048 = 2^16; `expk`: 32 x 134217728 = 2^32; no gap, no overlap).
Property: libc's body versus camada's `mkFXPExp` on the same symbolic input,
asserting exact agreement. bitwuzla only.

| entry point | shards | SUCCESSFUL | FAILED | slowest shard |
|---|---|---|---|---|
| `exphk` (s8.7) | 32 | 30 | **2** (15, 16) | 528.79s |
| `expk` (s16.15) | 32 | 30 | **2** (15, 16) | 1399.47s |

The failing shards are the ones straddling raw 0 -- the mid-range, not the
boundary windows the earlier defect harnesses targeted. Sharding therefore found
**new** counterexamples, not the ones already known.

### The counterexamples, and they violate the stated relative bounds

All three confirmed against native libc and against exp computed in long double.

| entry point | x | exp(x) | libc | error | vs claimed bound |
|---|---|---|---|---|---|
| `exphk` | -1.3984375 | 0.246982573 | 0.2421875 (raw 31, want 32) | 0.61 ulp | **4.97x** over 2^-8 |
| `exphk` | +0.5 | 1.648721271 | 1.6562500 (raw 212, want 211) | 0.96 ulp | **1.17x** over 2^-8 |
| `expk` | +0.3828125 | 1.466403054 | 1.46636963 (raw 48050, want 48051) | 1.10 ulp | **1.49x** over 2^-16 |
| `expk` | **-3.8750305** | 0.020753705 | 0.02072144 (raw 679, want 680) | 1.06 ulp | **101.90x** over 2^-16 |

`x = 0.5` and `x = 0.383` are about as ordinary as inputs get -- no saturation,
no flush, no table edge. Unlike the sqrt situation, these **do** violate a bound
libc states for the function in question: `exphk.cpp` and `expk.cpp` both give a
relative error bound for the `(1 + lo)` / `(1 + lo + lo^2/2)` step, and the
end-to-end relative error exceeds it.

The absolute errors are all ~1 ulp and go in both directions (raw 31 where 32 is
correct, raw 212 where 211 is correct), so the arithmetic is not broken -- the
finding is that the error is larger than documented.

But the **relative** picture degrades badly for small results, and that is what
the source actually bounds. At `x = -3.875` the true value is 0.0207, so a 1 ulp
absolute error is **101.9x** the claimed 2^-16 relative bound. A relative bound
is the wrong shape for a function whose output spans five orders of magnitude
across its domain: near the bottom of the range one ulp of absolute error is
enormous in relative terms, and no fixed-point implementation returning a
representable value could satisfy it there. That is arguably a defect in the
documented claim rather than in the code.

Fourth counterexample, from expk shard 15 which finished last (1399s):

```
x = -3.875030518   exp = 0.020753704511   libc = 0.020721435547 (raw 679, want 680)
                   1.06 ulp absolute, 1.554853e-03 relative, 101.90x the bound
```

### What sharding bought

The unsharded whole-domain query at s16.15 never finished (>90 min under
bitwuzla, and z3 only managed a narrow window). Sharded, 31 of 32 shards
returned in under 30 seconds and the whole 2^32 domain was covered. For
`exphk` the slowest shard was 528s and the rest were seconds.

This is the same ~90x pattern measured on the sqrt shards, and it is what made
whole-domain exp verification possible at all.

## Is fixed_point::sqrt correctly rounded? No -- 53-81% of inputs

Correct rounding is a stronger property than any error bound: does the function
return the *nearest representable value* to the true root for every input? It is
worth asking separately because `fixed_point::sqrt` documents no bound at all, so
correct rounding is the natural default expectation for a function returning the
same format as its argument.

Exhaustive native measurement, all inputs of each format:

| entry point | format | inputs | not correctly rounded | worst error | exact roots wrong |
|---|---|---|---|---|---|
| `sqrtuhr` | u0.8 | 256 | **146 (57.0%)** | 1.09 ulp | 8 / 16 |
| `sqrtur` | u0.16 | 65536 | **53060 (81.0%)** | 2.48 ulp | 153 / 256 |
| `sqrtuhk` | u8.8 | 65536 | **34767 (53.1%)** | 1.09 ulp | 153 / 256 |

Of the u0.16 failures, 28228 are more than one ulp out -- so it is not merely a
tie-breaking disagreement.

Proved in-solver too, not just enumerated. `harness_sqrt_correctly_rounded.cpp`
states correct rounding without computing the root: camada's oracle truncates, so
the true root is in `[oracle, oracle+1ulp)` and the nearest representable value
is decided by squaring the midpoint,

```
root >= oracle + 1/2   <=>   (2*oracle + 1)^2 <= 4 * raw_x * 2^F
```

**VERIFICATION FAILED** in 0.70s, counterexample `xb = 254`, confirmed natively:

```
x = 254/256 = 0.992187500
  true root * 256 = 254.998039   -> nearest representable is raw 255
  libc returns      raw 254       -0.998 ulp
```

The true root is 0.998 of the way to the next representable value and libc
returns the one below it -- a clear rounding-down where rounding up is nearer.

### How this fits the other sqrt findings

This is the sharpest statement of the sqrt situation, and it needs no argument
about which comment applies to which function:

* `fixed_point::sqrt` is **not correctly rounded**, on a majority of inputs, at
  every format measured.
* It is **within ~2.5 ulp** everywhere, so it is a usable approximation.
* It documents **no bound**, so nothing is contradicted -- but a caller has no
  way to know either fact from the source.

That combination is the actual finding: an undocumented ~2.5-ulp approximation
where the naming and signature suggest a correctly-rounded result. The exact
perfect squares make it concrete -- `sqrt(81/256) = 9/16` and
`sqrt(100/65536) = 10/256` are exactly representable and still come back one ulp
low, which no rounding convention explains.

## Multi-property UB sweep

`--multi-property --overflow-check` reports every violated claim in one run
instead of stopping at the first, and with no user-written assertions the
checkers themselves are the oracle. Applied to libc's real templates.

### divi: 3 violations reported, 2 real

```
fx_bits.h:256  undefined behavior on shift operation shl   CWE-1335
               F >= 0 && F < 64 && (signed long int)n >= 0
fx_bits.h:257  undefined behavior on shift operation ashr  CWE-1335   <- ESBMC bug, see below
               k >= 0 && k < 64
fx_bits.h:266  arithmetic overflow on shl                  CWE-190, CWE-191
               !overflow("shl", 1, F)
```

Reporting all three in a single run confirms they are **independent**, not one
masking another -- which is what the earlier one-at-a-time runs could not show.

Of the three:

* **256 is a real defect and broader than first reported.** Left-shifting a
  negative `int64_t` is UB (C11 6.5.7p4) at *every* format, not only `F >= 31`.
  `rdivi` passes signed operands straight through, so it is on the shipped path.
* **266 is the originally reported `1 << F`.** ESBMC's condition
  `!overflow("shl", 1, F)` states it exactly.
* **257 was a false positive -- caused by an ESBMC gap, now fixed.** My first
  explanation ("the checker does not know `countr_zero`'s range") was wrong. The
  real cause: `cpp::countr_zero` compiles to **`__builtin_ctzg`**
  (`CPP/bit.h:104`), and ESBMC did not model any of the `__builtin_ctz*` family
  -- it printed `WARNING: no body for function __builtin_ctzg` and treated the
  call as returning nondet. An unconstrained `k` then genuinely can exceed 63,
  so the violation was real *given the model*, and the model was the defect.

  This is the same defect class as `__builtin_clzg` (esbmc/esbmc#6925), fixed
  the same way: `ctz(x) = popcount(~x & (x-1))`, with the generic form's second
  argument supplying the zero-operand fallback. Verified correct rather than
  merely permissive -- checked against a shift-loop reference over a symbolic
  input, plus concrete spot checks -- and pinned by
  `regression/esbmc/builtin_ctz{,_fail}`.

  **After the fix, divi reports 2 violations instead of 3.** Line 257 is gone;
  256 and 266 remain, which are the real defects.

### Every other family: clean

Same flags, no assertions, symbolic input:

| harness | function | verdict | violations |
|---|---|---|---|
| `ub_sqrt_uhr` | `fixed_point::sqrt` u0.8 | SUCCESSFUL | 0 |
| `ub_sqrt_ur` | `fixed_point::sqrt` u0.16 | SUCCESSFUL | 0 |
| `ub_isqrt_fast_uhk` | `isqrt_fast` | SUCCESSFUL | 0 |
| `ub_round_hk` | `round` | SUCCESSFUL | 0 |
| `ub_countls_hk` | `countls` | SUCCESSFUL | 0 |
| `ub_abs_hk` | `abs` | SUCCESSFUL | 0 |

So the UB is confined to `divi`. That is worth stating positively: the accuracy
findings elsewhere (sqrt not correctly rounded, exp's placeholder entry) are
*accuracy* problems in code that is otherwise free of undefined behaviour, and
`--overflow-check` clears the five families that pass on accuracy too.

## Audit: unmodelled functions across every harness

Prompted by the `__builtin_ctzg` gap, which I should have been checking for from
the start. Every harness re-run on the current binary, capturing both the verdict
and any `WARNING: no body for function` line.

**26 of 27 finding harnesses: zero unmodelled functions.** One hit, and it does
not affect its verdict:

| harness | verdict | `no body` |
|---|---|---|
| all 5 `sqrt*_vs_camada` | FAILED | 0 |
| both `isqrt_fast_*` | FAILED | 0 |
| both `isqrt_*_bound` | FAILED | 0 |
| `sqrt_correctly_rounded`, `sqrt_rescale_gap` | FAILED | 0 |
| `sqrt_core_bound` | SUCCESSFUL | 0 |
| all 3 exp defect harnesses | FAILED | 0 |
| all 8 `divi*_vs_camada` | SUCCESSFUL | 0 |
| `divi_bug1`, `divi_widecast` | FAILED | 0 |
| `divi_control` | SUCCESSFUL | 0 |
| **`divi_widths`** | FAILED | **1** |

### The one hit, and why the verdict stands

`divi_widths` reports `no body for function operator()#I#1` -- a **lambda**, not a
libc builtin. `divi` defines `is_power_of_two` as a lambda (fx_bits.h:249), and
if its result were nondet the power-of-two branch would be reachable for any
divisor, making every `divi` verdict suspect.

Tested rather than assumed, three ways:

1. **Lambdas are modelled correctly in general** -- `is_power_of_two(8)` is true
   and `(6)` is false, both PASSED.
2. **The branch is gated correctly inside libc's own code**: with `d = -3` (not a
   power of two) the general branch runs and the result is positive; with
   `d = -1` the fast branch runs and the defect fires. Both PASSED in one
   multi-property run. libc's internal `LIBC_ASSERT` on `initial_approx`
   (fx_bits.h:307) also passed, confirming the general branch really executed.
3. **The finding survives on fully concrete inputs.** Re-stated with literal
   `divi(-3, -1)` at s0.15 and s0.31 -- nothing symbolic, so no unmodelled call
   can influence branch selection -- **both still FAIL**.

### Root cause found, filed as esbmc/esbmc#6969

The trigger is **multiple template instantiations**, not the destructor. When a
function template containing a named lambda is instantiated more than once, only
the first instantiation gets a body for the closure's `operator()`; later ones
call a bodyless symbol and symex assigns a **nondet return value**. Minimal
reproducer, 7 lines:

```cpp
template <typename T> static int f() {
  auto g = [](int x) { return x > 0; };
  return g(1) ? 1 : 0;
}
int main() { return f<int>() + f<long>(); }   // warns
int main() { return f<int>(); }               // clean
```

Demonstrably unsound in general -- of three true assertions across three
instantiations, only the first passes. Reproduces on upstream master
(9a3d7e8a6c), so it is not a camada-branch artefact. Filed with the reproducer.

### Impact on this work: one harness, and its finding survives

Audited every harness. **`harness_divi_widths` is the only one that warns**,
because it is the only one instantiating a template at two types:

| harness | instantiations | warns |
|---|---|---|
| `harness_divi_widths` | `divi<_Fract>`, `divi<long _Fract>` | **yes (1)** |
| `harness_divi_bug1` | `divi<short _Fract>` | no |
| `harness_divi_widecast` | `divi<long _Accum>` | no |
| `harness_divi_control` | `divi<short _Fract>` | no |
| all sqrt / exp / divifx harnesses | one each | no |

Split into two single-instantiation harnesses -- `harness_divi_s015.cpp` and
`harness_divi_s031.cpp` -- **both still FAIL with zero warnings**. So the
"defect 1 is not width-specific" claim holds on evidence that cannot be affected
by #6969.

**Update: #6969 is fixed upstream (#6976, "Give a template instantiation's
lambda its own closure type").** With that in the tree the original
`harness_divi_widths.cpp` runs with **zero `no body` warnings** and still
reports FAILED, so the split is no longer needed to make the result trustworthy.
The split harnesses are kept anyway -- one instantiation each is a cheaper query
and states the per-format claim more directly -- but either form is now sound.

Nothing else in this work is exposed: every other harness instantiates each
template exactly once, and the two accuracy families (sqrt, exp) were also
confirmed by running native libc binaries, which involve no verifier at all.

### The earlier destructor explanation was a red herring

Inspecting the GOTO program settles it. The lambda's `operator()` **is fully
translated** -- it is in the program with a body:

```
operator() (...divi<#@BT@Fract>...operator()#I#1):
    RETURN: n::0 > 0 && (n::0 & n::0 - 1) == 0
    END_FUNCTION
```

That is exactly `is_power_of_two` from fx_bits.h:249. So the branch gate is
modelled, and `is_power_of_two` is unambiguously **on the verification path** --
`divi` cannot reach either branch without evaluating it.

What has no body is the lambda's **destructor**, called at scope exit:

```
FUNCTION_CALL: ~(lambda at fx_bits.h:249:26)(&is_power_of_two)   // line 262
DEAD ...divi<#@BT@Fract>...is_power_of_two
```

**This is an ESBMC frontend bug, not a reporting artefact.** I called it
"spurious" and that was wrong: the frontend emits a `FUNCTION_CALL` to a symbol
it never defines. `symex_function.cpp:532` then takes the bodyless path, which
assigns a **nondet value to the return** and (under
`--unknown-method-args-check`) **invalidates pointer arguments**. Those are real
state effects; a captureless destructor being semantically trivial does not make
the missing body correct.

What makes it harmless *here* is placement, established from the GOTO dump
rather than assumed:

| GOTO line | call | body? |
|---|---|---|
| 1358 | `operator()(&is_power_of_two, abs(d))` -- the gate | **yes** |
| 1438, 1488, 1548, 1600, 1646, 1826 | `~(lambda ...)(&is_power_of_two)` | no |

Every bodyless call is **after** the gate and after the value is computed, and
the destructor returns void so the nondet-return assignment does not fire.
Confirmed behaviourally: `divi(-3,-1)` at s0.15 is exactly `-32768`,
deterministically, both with and without `--unknown-method-args-check`.

So the verdict stands, but on the narrow grounds that the missing body cannot
influence anything reached before it -- not because the call is unimportant. A
bodyless call with a non-void return, or one placed before the value
computation, would have corrupted the result silently. It should be filed
against ESBMC.

I could not reduce it to a standalone reproducer: a named lambda in an explicit
scope, inside a template, and with `divi`'s early-return shape all translate
their destructors correctly. The trigger needs something more specific in
`divi`'s instantiation context, which is worth finding before filing.

This corrects my earlier phrasing. I had written that the warning was "real but
inert", inferred from the behavioural tests passing. It is better than that: the
function the warning names is modelled, and the one that is not modelled has no
semantics to model. The behavioural tests were consistent with this all along --
they passed because the gate genuinely works, not by luck.

### What the ctz gap actually invalidated

Scope, established by grep rather than assumption: `cpp::countr_zero` appears at
**exactly one place** in the fixed-point headers -- `fx_bits.h:254`, inside
`divi`. It is not reached by `sqrt`, `isqrt`, `isqrt_fast`, `exp`, `round`,
`abs`, `countls`, `bitsfx` or `divifx`.

So the gap affected **one reported line in one function**: the spurious
`fx_bits.h:257` shift violation. Everything else stands, and the two real `divi`
UB sites (256, 266) were confirmed independently of it.

`cpp::countl_zero` (`-> __builtin_clzg`) is used more widely, in both `fx_bits.h`
and `sqrt.h` -- but that builtin was modelled earlier in this branch
(esbmc/esbmc#6925), before any of the sqrt or exp measurements were taken.
Re-confirmed: `clzg(1,32) = 31` and `clzg(0,32) = 32` both PASS.

### Process change

Every harness verdict from here on is reported with its `no body` count, and a
non-zero count is investigated before the verdict is believed. Grepping the
verifier's own warnings should have been step one, not a late audit.
