# Metamorphic testing plan for the fp2bv layer

**Subsystem:** `src/solvers/smt/fp/fp_conv.{h,cpp}` (class `fp_convt`), the
lowering of IEEE 754 binary floating-point operations to bit-vector terms.
**Reference semantics:** IEEE 754-2019. No SMT solver is treated as correct.
**Status (2026-11-11, `7e3d4a7df8`):** proposal. Line numbers cite that commit.
Section 2.6 records one measured observation (H1, since fixed) and eight unverified
hypotheses from reading the code. Nothing else here has been run.

Clause numbers follow my reading of IEEE 754-2019. I could not fetch the
standard text in this session, so Phase 0 re-checks every clause number in
Section 3 against the published document before any of them reaches a commit
message.

## 1. Goal and trust argument

ESBMC answers floating-point queries through `fp_convt` whenever the backend
has no floating-point theory (Boolector, Yices, the SMT-LIB printer) or when
the user passes `--fp2bv`. A wrong term in that lowering produces a wrong
verdict in both directions: a false `VERIFICATION SUCCESSFUL` or a spurious
counterexample.

Two facts shape the method.

First, `fp_conv.cpp` shares ancestry with Z3's `fpa2bv` converter. The file
says so twice: the `CMW: is this always large enough?` note at
`fp_conv.cpp:2050` and "Z3's fpa2bv does the same" at `:1617`. Agreement
between `--z3` native floating point and `--z3 --fp2bv` is therefore weak
evidence, because a defect inherited from the common design shows up on both
sides and cancels.

Second, every call to an `fp_convt` method builds the whole circuit, all `ite`
arms included, whatever the operand values are. One call to
`mk_smt_fpbv_add` executes nearly every line of that function. Line coverage
of `fp_conv.cpp` saturates after a handful of tests and says almost nothing
about which semantic cases were exercised. Section 8 defines the coverage
metrics that replace it.

The trust chain we propose has four links, in decreasing order of authority:

1. The text of IEEE 754-2019.
2. An exact-rational reference evaluator (`refloat`, Section 5.1), about 300
   lines, written from the standard's definitions and small enough to review
   against them line by line.
3. Three independent implementations that must agree with `refloat` on every
   vector where they apply: MPFR, Berkeley SoftFloat, and the host FPU.
4. Metamorphic relations, each one admitted only after `refloat` has checked
   it exhaustively on small formats (the validity gate, Section 4.2).

Solvers enter below that chain, as subjects under test and as differential
signals for triage.

## 2. Architecture of the fp2bv layer

### 2.1 How a solver gets its floating-point converter

`create_solver` in `src/solvers/solve.cpp:244-281` asks the backend factory
for an `fp_convt *fp_api` and then decides:

```cpp
if (fp_api == nullptr || fp_to_bv)
  ctx->set_fp_conv(new fp_convt(ctx));   // generic bit-vector lowering
else
  ctx->set_fp_conv(fp_api);              // backend's own converter
```

Every backend class inherits from `fp_convt`. Those with a floating-point
theory override its virtuals; the others inherit the lowering unchanged.

| Backend | Overrides of `mk_smt_fpbv_*` / casts in `*_conv.h` | Default path | With `--fp2bv` |
|---|---|---|---|
| Z3 | 29 | native FP | fp2bv |
| Bitwuzla | 29 | native FP | fp2bv |
| CVC4, cvc5 | 29 each | native FP | fp2bv |
| MathSAT | 27 | native FP (partial) | fp2bv |
| Boolector | 0 | fp2bv | fp2bv |
| Yices | 0 | fp2bv | fp2bv |
| SMT-LIB printer (`--smtlib`) | 0 | fp2bv | fp2bv |
| MiniSat | no `fp_api` | fp2bv | fp2bv |

Two modes sit outside this plan's scope: `--fixedbv` (fixed-point encoding)
and `--ir-ieee` (`ir_ieee_conv.cpp`, real-arithmetic enclosures). Both are
over-approximations with different contracts.

### 2.2 Dispatch from irep2 to `fp_convt`

| Expression | Call site | `fp_convt` method |
|---|---|---|
| `ieee_add/sub/mul/div` | `smt_solver.cpp:580-586` | `mk_smt_fpbv_add/sub/mul/div` |
| IEEE remainder | `smt_solver.cpp:590` | `mk_smt_fpbv_rem` |
| `ieee_fma` | `smt_solver.cpp:886` | `mk_smt_fpbv_fma` |
| `ieee_sqrt` | `smt_solver.cpp:1016` | `mk_smt_fpbv_sqrt` |
| `nearbyint` | `smt_solver.cpp:1198` | `mk_smt_nearbyint_from_float` |
| `==`, `!=` on floats | `smt_solver.cpp:1335,1354` | `mk_smt_fpbv_eq` |
| `<`, `<=`, `>`, `>=` | `smt_solver.cpp:1513-1612` | `mk_smt_fpbv_lt/lte/gt/gte` |
| `abs`, unary minus | `smt_solver.cpp:1434,1685` | `mk_smt_fpbv_abs/neg` |
| float constants | `smt_solver.cpp:2180-2187` | `mk_smt_fpbv`, `_nan`, `_inf` |
| model extraction | `smt_solver.cpp:3728` | `get_fpbv` |
| int to float | `smt_casts.cpp:164,170` | `mk_smt_typecast_{u,s}bv_to_fpbv` |
| float to float | `smt_casts.cpp:176,196` | `mk_smt_typecast_from_fpbv_to_fpbv` |
| float to int | `smt_casts.cpp:188,192` | `mk_smt_typecast_from_fpbv_to_{u,s}bv` |
| bitcast | `smt_bitcast.cpp:627,810` | `mk_from_bv_to_fp`, `mk_from_fp_to_bv` |
| `isnan/isinf/isnormal/isfinite/signbit` | `smt_fp_conv.cpp:664-853` | `mk_smt_fpbv_is_*` |
| rounding mode operand | `smt_fp_conv.cpp:868-932` | `mk_smt_fpbv_rm` |

The rounding mode is a 3-bit vector holding `ieee_floatt::rounding_modet`:
`ROUND_TO_EVEN = 0`, `ROUND_TO_AWAY = 1`, `ROUND_TO_PLUS_INF = 2`,
`ROUND_TO_MINUS_INF = 3`, `ROUND_TO_ZERO = 4`
(`src/util/arith/ieee_float.h:79-83`). C programs reach it through the global
`__ESBMC_rounding_mode` that `fesetround` writes
(`src/c2goto/library/fenv.c:3-47`).

### 2.3 The lowering itself

A floating-point value of sort `(ebits, sbits)` is a plain bit-vector of width
`ebits + sbits` laid out as sign, biased exponent, trailing significand;
`sbits` counts the hidden bit. `mk_from_bv_to_fp` only re-tags the sort
(`:2368`) and `mk_from_fp_to_bv` is the identity (`:2373`), so bitcasts cost
nothing and preserve every bit, NaN payloads included.

Each arithmetic method follows one template:

1. **Special-case guards.** A chain of `(c_i, v_i)` pairs covers NaN,
   infinities, and zeros, combined at the end by nested `mk_ite` with `c1`
   outermost.
2. **`unpack`** (`:1892`). Splits the operand into sign, significand with the
   hidden bit made explicit, and unbiased exponent (`mk_unbias`, `:1963`).
   With `normalize = true` it left-shifts a subnormal significand and reports
   the shift in `lz` (`mk_leading_zeros`, `:1975`, a recursive halving
   count). Addition passes `normalize = false` and aligns inside `add_core`.
3. **Exact core** on widened vectors: `add_core` (`:1153`) with a sticky bit,
   a `2*sbits` product for multiplication (`:1413`), a
   `2*sbits + extra_bits` quotient for division (`:1526`), a restoring
   square-root loop of `sbits + 3` iterations (`:307`), a `2*sbits + 5`
   aligned sum for FMA (`:532`), and for remainder a shift across the whole
   exponent range of `2^ebits - 3` bits (`:1618`).
4. **`round`** (`:2007`). Takes sign, an `sbits + 4` significand, and an
   `ebits + 2` signed exponent. It computes `TINY` (`:2056`) to choose
   between normalization and denormalization shifts, folds the shifted-out
   bits into a sticky bit, asks `mk_rounding_decision` (`:2218`) for the
   increment from `last`, `round`, `sticky`, sign, and mode, post-normalizes
   on carry-out (`SIGovf`, `:2118`), and detects overflow (`OVF1`, `OVF2`,
   `:2048,2141`). On overflow the mode and sign select between infinity and
   the largest finite value (`:2171-2185`).

### 2.4 Where each semantic concern lives

| Concern | Location in `fp_conv.cpp` |
|---|---|
| Rounding increment for all five modes | `mk_rounding_decision` `:2218-2249` |
| Overflow result per mode and sign | `round` `:2155-2185` |
| Gradual underflow, subnormal results | `round` `:2051-2100` (`TINY`, `sigma`, sticky) |
| Rounding carry into next binade | `round` `:2113-2127` |
| Subnormal operands | `unpack` `:1921-1951`, `mk_is_denormal` `:1815` |
| NaN generation | `mk_smt_fpbv_nan` `:46-56` |
| NaN, infinity, zero guards per op | head of each `mk_smt_fpbv_*` |
| Signed-zero results of add | `mk_smt_fpbv_add` `:1236-1333` |
| Comparisons | `mk_smt_fpbv_eq` `:1703`, `mk_smt_fpbv_lt` `:1729` |
| Classification | `:1784-1872` |
| Sign-bit operations | `mk_smt_fpbv_abs` `:1874`, `mk_smt_fpbv_neg` `:1884` |
| Float to integer (always toward zero) | `mk_to_bv` `:636-772` |
| Integer to float | `:926-1141` |
| Format conversion | `:786-924` |
| Round to integral | `mk_smt_nearbyint_from_float` `:73-248` |
| Remainder | `mk_smt_fpbv_rem` `:1559-1701` |

### 2.5 Formats ESBMC can send to fp2bv

From `src/util/lang/c_types.cpp:6-47` and `:220-228`:

| Format | Width | Fraction bits `f` | Exponent bits `e` | `sbits` | C type |
|---|---|---|---|---|---|
| binary16 | 16 | 10 | 5 | 11 | `_Float16`, `__fp16` |
| bfloat16 | 16 | 7 | 8 | 8 | `__bf16` |
| binary32 | 32 | 23 | 8 | 24 | `float` |
| binary64 | 64 | 52 | 11 | 53 | `double` |
| 96-bit synthetic | 96 | 80 | 15 | 81 | 96-bit `long double` targets |
| binary128 | 128 | 112 | 15 | 113 | `__float128`, 128-bit `long double` |

The 96-bit row is not the x87 extended format: x87 stores an explicit integer
bit and 64 significand bits, while ESBMC models an implicit-bit format with
80 fraction bits. No hardware or SoftFloat oracle matches it. Only the
parametric oracles (`refloat`, MPFR) can judge it. The width of `long double`
depends on `config.ansi_c.long_double_width`, so every test names its format
explicitly.

`fp_convt` itself accepts any `(ebits, sbits)`. The unit tier uses that to
run toy formats such as `(e=3, f=4)`, where exhaustive enumeration takes
seconds.

### 2.6 Observations from the code

These are starting targets. H1 is measured; the rest are unverified
hypotheses and must not be reported as defects before a probe confirms them.

| ID | Location | Observation | Status |
|---|---|---|---|
| H1 | `:46-56` | `mk_smt_fpbv_nan` built significand value 1, so every generated NaN was `0x7F800001` in binary32 (quiet bit clear, a signaling pattern on x86-64 and AArch64; 754-2019 §6.2.1 recommends the quiet encoding). `ieee_floatt::pack()` had the same encoding. | **Measured, then fixed** in the PR that adds this plan: both now emit the quiet NaN (`0x7FC00000`), pinned by `regression/floats/fp2bv_quiet_nan`. |
| H2 | `:233` | `mk_zero_ext(renorm_delta, sbits - ebits - 2)` underflows `unsigned` when `sbits < ebits + 2`. bfloat16 has `sbits = ebits = 8`. | Unverified |
| H3 | `:1501` | `assert(ebits <= sbits)` in division. Holds for all six formats; the `else` branch in `unpack` (`:1938-1950`) is then unreachable from C. | Unverified dead-code candidate |
| H4 | `:860,916` | `exponent_overflow` is the constant `false`; the `sig_inf` arm never fires and overflow is left to `round`. | Unverified dead-code candidate |
| H5 | `:1001,1112` | `exp_worst_case_sz` uses host `log()` on doubles. A libm that returns `2.9999999999999996` for `log(8)/log(2)` shifts the result by one. | Unverified |
| H6 | `:2050` | `lz` is `ebits + 2` bits wide while it counts zeros of an `sbits + 4` vector. Safe when `sbits + 4 < 2^(ebits+1)`, which all six formats satisfy; toy formats must respect the same bound. | Constraint on test formats |
| H7 | `:1596-1597,1699` | Remainder returns `pzero` for `x = 0` and then repairs the sign in a final `ite`. `rem(-0, y)` must be `-0`. | Unverified |
| H8 | `:653-656,742-748` | The comment says negative inputs give an unspecified result, but only NaN and infinity take `c1`. `(unsigned)(-0.5)` must be 0 (C11 6.3.1.4p1 truncates first). | Unverified |
| H9 | `:2245-2248`, `smt_fp_conv.cpp:902-932` | A symbolic rounding mode with value 5, 6, or 7 makes `mk_rounding_decision` truncate while `round` picks infinity on overflow. Whether such a value can reach the solver depends on the clamp in `convert_rounding_mode`. | Unverified |

## 3. IEEE 754 compliance surface

### 3.1 What ESBMC models and what it does not

`fp_convt` has no output for status flags. Invalid, division-by-zero,
overflow, underflow, and inexact (754-2019 §7.2 to §7.6) are observable only
through their default results. Three consequences follow for the tests:

- A signaling NaN behaves like a quiet NaN in every arithmetic operation. We
  test that it yields NaN and that sign-bit operations and bitcasts preserve
  its bits.
- Whether tininess is detected before or after rounding (§7.5) changes only
  the underflow flag, so the choice is unobservable here.
- "Exact" and "inexact" are test-design categories that the oracle computes.
  The tests reach them indirectly, through mode-invariance relations.

### 3.2 Behavior matrix

| Area | Behavior to test | 754-2019 clause | `fp_convt` entry |
|---|---|---|---|
| Encodings | Field layout, bias, subnormal and zero encodings, infinity, NaN | §3.3, §3.4 | `unpack`, `mk_unbias`, `mk_bias`, constants |
| Rounding | roundTiesToEven, roundTiesToAway | §4.3.1 | `mk_rounding_decision` |
| Rounding | roundTowardPositive, roundTowardNegative, roundTowardZero | §4.3.2 | `mk_rounding_decision`, `round` |
| Arithmetic | add, subtract, multiply, divide, squareRoot, fusedMultiplyAdd | §5.4.1 | `mk_smt_fpbv_{add,sub,mul,div,sqrt,fma}` |
| Arithmetic | remainder (exact, quotient rounded to nearest even) | §5.3.1 | `mk_smt_fpbv_rem` |
| Integral rounding | roundToIntegral in each direction | §5.3.1, §5.9 | `mk_smt_nearbyint_from_float` |
| Conversion | convertFromInt, signed and unsigned | §5.4.1 | `mk_smt_typecast_{s,u}bv_to_fpbv` |
| Conversion | convertFormat widening and narrowing | §5.4.2 | `mk_smt_typecast_from_fpbv_to_fpbv` |
| Conversion | convertToIntegerTowardZero; C leaves NaN, infinity, and out-of-range undefined (C11 6.3.1.4p1) | §5.8 | `mk_to_bv` |
| Sign bit | copy, negate, abs: bit operations, no NaN special case | §5.5.1 | `mk_smt_fpbv_neg`, `mk_smt_fpbv_abs` |
| Comparison | less, lessEqual, greater, greaterEqual, equal, notEqual; unordered with NaN; `-0 = +0` | §5.6.1, §5.11 | `mk_smt_fpbv_{lt,lte,gt,gte,eq}` |
| Classification | isNaN, isInfinite, isFinite, isNormal, isSubnormal, isZero, isSignMinus | §5.7.2 | `mk_smt_fpbv_is_*`, `mk_is_denormal` |
| Infinity | `inf + x`, `inf - inf`, `inf * 0`, `x / inf`, `sqrt(+inf)` | §6.1, §7.2 | guards in each op |
| NaN | Any NaN operand gives NaN; invalid operations give a quiet NaN | §6.2, §7.2 | guards, `mk_smt_fpbv_nan` |
| Sign of zero | `x - x` is `+0` except `-0` under roundTowardNegative; product and quotient sign is the XOR; `sqrt(-0) = -0`; `fma` zero sums | §6.3 | add, mul, div, sqrt, fma |
| Division by zero | finite nonzero `/ 0` gives signed infinity | §7.3 | `mk_smt_fpbv_div` |
| Overflow | Result per mode and sign: infinity or largest finite | §7.4 | `round` |
| Underflow | Gradual: subnormal results correctly rounded, down to zero | §7.5 | `round` (`TINY`) |
| Cancellation | Massive cancellation in add, sub, fma; exactness of `x - y` for close operands | §5.4.1 | `add_core`, fma |

### 3.3 Input classes

Each operand is drawn from ten classes per sign: zero; smallest subnormal;
interior subnormal; largest subnormal; smallest normal; interior normal;
power of two; largest finite; infinity; and NaN (quiet with payload,
signaling pattern, and `fp_convt`'s own `...0001` pattern). Unary operations
cover 20 signed classes and binary operations the 400 ordered pairs.

### 3.4 Result events

The oracle tags each vector with the events its exact result triggers. They
are the units of semantic coverage in Section 8:

`exact`, `inexact-below-half`, `tie`, `inexact-above-half`, `carry-out`
(rounding moves the result into the next binade), `overflow`,
`overflow-at-max` (exact result between the largest finite and the overflow
threshold), `subnormal-exact`, `subnormal-inexact`, `rounds-to-min-normal`,
`underflow-to-zero`, `total-cancellation`, `partial-cancellation` (at least
`sbits/2` leading bits cancel), `zero-sign-by-rule`, `invalid`,
`div-by-zero`, `nan-propagation`.

## 4. Metamorphic relations

### 4.1 Notation and comparison operators

- `∘` is an operation evaluated under rounding mode `rm`. The five modes are
  RNE, RNA, RTP, RTN, RTZ.
- `mirror(rm)` swaps RTP and RTN and fixes RNE, RNA, RTZ. Negating a real
  number turns rounding up into rounding down, which is why relations with a
  sign flip must mirror the mode.
- `a ≡ b`: both are NaN, or their bit patterns are identical. This is the
  default comparison. It distinguishes `+0` from `-0` and ignores NaN sign and
  payload, which 754-2019 §6.2.3 leaves as a recommendation.
- `a ≐ b`: `a ≡ b`, or both are zeros of either sign.
- `a ≤ b`: IEEE comparison; stated only when neither side is NaN.
- "Exact" means the real-number result is representable in the format.

A relation is violated only when its precondition holds. The harness encodes
the precondition as `__ESBMC_assume` in symbolic mode and as a filter in
concrete mode.

### 4.2 Validity gate for relations

A metamorphic relation that is false in IEEE arithmetic produces false alarms
at best. At worst, someone weakens the oracle to make it pass.

Before a relation enters the catalog, `mr/validate.py` evaluates it with
`refloat` over **every** operand tuple of the toy formats `(e=3, f=2)`,
`(e=3, f=4)`, and `(e=4, f=5)`, in all five modes, and over binary16
exhaustively for unary relations and on 10^7 sampled pairs for binary ones.
A single counterexample rejects the relation or narrows its precondition.
The gate also reports the *vacuity rate*: the fraction of tuples that satisfy
the precondition. A relation whose precondition holds on fewer than 0.1% of
tuples needs a dedicated generator (Section 6.4).

Relations marked "gate decides" below are the ones whose published proofs
carry side conditions I do not want to restate from memory.

### 4.3 Relations that are invalid and must not be used

| Real-algebra identity | Why it fails in IEEE 754 |
|---|---|
| `(x + y) + z = x + (y + z)` | Each addition rounds. `(1 + 2^-24) + 2^-24` is `1` in binary32 RNE; `1 + (2^-24 + 2^-24)` is `1 + 2^-23`. |
| `(x + y) - y = x` | Rounding of `x + y` loses low bits of `x`; fails outright when `x + y` overflows. |
| `x * (y + z) = x*y + x*z` | Three roundings on the right, two on the left. |
| `(x * y) / y = x`, `(x / y) * y = x` | Both operations round; also fails on overflow, underflow, zero, infinity. |
| `x / y = x * (1 / y)` | `1 / y` is rounded unless `y` is a power of two (see D4). |
| `x - x = 0` | NaN for infinity and NaN; the zero is `-0` under RTN. |
| `x + 0 = x` | `(-0) + (+0)` is `+0` except under RTN. Use A2. |
| `0 * x = 0` | NaN for infinity and NaN; the sign follows the XOR rule. |
| `x == x` | False for NaN. |
| `!(x < y)` implies `x >= y` | Both are false when an operand is NaN. |
| `sqrt(x) * sqrt(x) = x`, `sqrt(x*y) = sqrt(x)*sqrt(y)` | Each step rounds. |
| `fma(x, y, z) = x*y + z` | The right side rounds twice. Equal only when `x*y` is exact (F5). |
| `(float)(double)` of a double result equals the float result | Double rounding. Valid only under the conditions of V3. |
| `-(x + y) = (-x) + (-y)` in a fixed directed mode | The mode must be mirrored (A3). |
| `(int)(x + 0.5)` rounds to nearest | Fails for `0.49999997f` and for large odd values. |

### 4.4 Addition and subtraction

| ID | Relation | Precondition | Modes |
|---|---|---|---|
| A1 | `x + y ≡ y + x` | none | all |
| A2 | `x + (-0) ≡ x` | `x` not NaN | all |
| A2b | `x + (+0) ≡ x` | `x` not NaN, `x` not `-0` | all |
| A2c | `(-0) + (+0)` is `-0` under RTN and `+0` otherwise | none | all |
| A3 | `-(x +_rm y) ≡ (-x) +_mirror(rm) (-y)` | for RNE, RNA, RTZ: the exact sum is nonzero, or `x` and `y` are zeros of the same sign | all |
| A4 | `x - y ≡ x + (-y)` | none | all |
| A5 | Sterbenz: `x - y` is exact, so the five modes give `≐` results, and `(x - y) + y ≡ x` | `x`, `y` finite, `y/2 <= x <= 2y`; for bitwise equality across modes also `x != y` | all |
| A6 | Bracketing: `RTN(x+y) <= RNE(x+y) <= RTP(x+y)`; RNE, RNA, and RTZ results each equal the RTN or the RTP result; `RTP ≐ RTN` or RTP is the successor of RTN | result not NaN | cross-mode |
| A7 | Monotonicity: `y1 <= y2` implies `x + y1 <= x + y2` | no NaN among operands or results | all, same mode both sides |
| A8 | `x + x ≡ x * 2` | `x` not NaN | all |
| A9 | 2Sum: with `s = x+y`, `a' = s-y`, `b' = s-a'`, `e = (x-a') + (y-b')`, the reals satisfy `s + e = x + y` | finite operands, `s` finite; gate decides the subnormal edge | RNE |
| A10 | `x + y` is infinite with the common sign when exactly one operand is infinite or both are infinities of one sign; `(+inf) + (-inf)` is NaN | none | all |

A9 is checked with exact rational arithmetic on the decoded `s` and `e`. The
floating-point relation `s + e == x + y` would round again and prove nothing.

### 4.5 Multiplication

| ID | Relation | Precondition | Modes |
|---|---|---|---|
| M1 | `x * y ≡ y * x` | none | all |
| M2 | `x * 1 ≡ x`; `x * (-1) ≡ -x` | `x` not NaN | all |
| M3 | `(-x) *_rm y ≡ -(x *_mirror(rm) y)`; `(-x) * (-y) ≡ x * y` | none | all |
| M4 | `x * 2^k` keeps the fraction field of `x` and adds `k` to its exponent field | `x` normal, `x * 2^k` normal, `2^k` representable | all |
| M5 | `(x * 2^k) * 2^-k ≡ x` | as M4, and both powers representable | all |
| M6 | Monotonicity: for `y >= 0`, `x1 <= x2` implies `x1 * y <= x2 * y` | no NaN among operands or results | all |
| M7 | Zero and infinity: finite `x` times `±0` is a zero with XOR sign; `inf * 0` is NaN; `inf * x` is infinity with XOR sign for nonzero `x` | as stated | all |
| M8 | Error-free product: `p = x*y`, `e = fma(x, y, -p)`; the reals satisfy `p + e = x*y` | finite; `p` finite; exponent of `x*y` at least `emin + sbits - 1` | RNE |
| M9 | Bracketing across modes, as A6 | result not NaN | cross-mode |

### 4.6 Division

| ID | Relation | Precondition | Modes |
|---|---|---|---|
| D1 | `x / 1 ≡ x`; `x / (-1) ≡ -x` | `x` not NaN | all |
| D2 | `x / x ≡ 1` | `x` finite, nonzero | all |
| D3 | `(-x) /_rm y ≡ -(x /_mirror(rm) y)`; `(-x)/(-y) ≡ x / y` | none | all |
| D4 | `x / 2^k ≡ x * 2^-k` | `2^k` and `2^-k` both representable (subnormal powers allowed); `x` not NaN | all |
| D5 | finite nonzero `/ ±0` is infinity with XOR sign; `0/0` and `inf/inf` are NaN; finite `/ inf` is zero with XOR sign | as stated | all |
| D6 | Monotonicity in the numerator for `y > 0`; antitonicity in the denominator for `x > 0`, `0 < y1 <= y2` | no NaN | all |
| D7 | Exact bracketing: with `q_d = x /_RTN y` and `q_u = x /_RTP y`, the rationals satisfy `q_d * y <= x <= q_u * y` | `y > 0`, `q_d` and `q_u` finite | cross-mode |
| D8 | Bracketing across modes, as A6 | result not NaN | cross-mode |

D4 holds through overflow and underflow because both sides round the same
real number once. That makes it the most useful division relation for the
subnormal range.

### 4.7 Square root

| ID | Relation | Precondition | Modes |
|---|---|---|---|
| S1 | `sqrt(+0) ≡ +0`, `sqrt(-0) ≡ -0`, `sqrt(+inf) ≡ +inf`; `sqrt(x)` is NaN for `x < 0` and for NaN | as stated | all |
| S2 | `sqrt(m * m) ≡ m` | `m >= 0` and `m*m` exact (for example an integer `m < 2^(sbits/2)`) | all |
| S3 | `sqrt(x * 4^k) ≡ sqrt(x) * 2^k` | `x > 0`; `x`, `x * 4^k`, and `sqrt(x) * 2^k` normal | all |
| S4 | `sqrt(x * x) ≡ abs(x)` | `x*x` neither overflows nor underflows; gate decides | RNE |
| S5 | Monotonicity: `0 <= x1 <= x2` implies `sqrt(x1) <= sqrt(x2)` | finite | all |
| S6 | Exact bracketing: with `r = sqrt_RTN(x)`, the rationals satisfy `r^2 <= x < succ(r)^2` | `x > 0` finite | RTN |
| S7 | The result is never subnormal and never overflows; its sign bit is set only for `-0` | `x >= 0` | all |

### 4.8 Fused multiply-add

| ID | Relation | Precondition | Modes |
|---|---|---|---|
| F1 | `fma(x, y, z) ≡ fma(y, x, z)` | none | all |
| F2 | `fma(x, 1, z) ≡ x + z` | no NaN | all |
| F3 | `fma(x, y, -0) ≡ x * y` for RNE, RNA, RTZ, RTP; `fma(x, y, +0) ≡ x * y` for RTN | no NaN | per mode |
| F4 | `fma(-x, y, -z)` under `rm` `≡ -fma(x, y, z)` under `mirror(rm)` | as A3, applied to the exact value `x*y + z` | all |
| F5 | `fma(x, y, z) ≡ (x * y) + z` | `x*y` exact and finite | all |
| F6 | `fma(x, y, -(x*y))` is exact: the five modes give `≐` results | as M8 | cross-mode |
| F7 | `fma(inf, 0, z)` and `fma(0, inf, z)` are NaN for every `z` | none | all |

F3 picks the zero addend per mode because of §6.3: an exact zero sum of
opposite signs is `-0` only under RTN.

### 4.9 Remainder

`mk_smt_fpbv_rem` takes no rounding mode; the operation is exact.

| ID | Relation | Precondition |
|---|---|---|
| R1 | `rem(x, y) ≡ rem(x, -y)` | none |
| R2 | `rem(-x, y) ≡ -rem(x, y)` | none |
| R3 | `rem(x, y) ≡ x` | `x` finite, and `y` infinite or `abs(x) <= abs(y)/2` with `y` nonzero |
| R4 | `rem(rem(x, y), y) ≡ rem(x, y)` | none |
| R5 | `abs(rem(x, y)) <= abs(y) / 2`, checked on rationals | `x` finite, `y` finite nonzero |
| R6 | A zero result has the sign of `x` | result is zero |
| R7 | `rem(inf, y)` and `rem(x, 0)` are NaN | none |
| R8 | `rem(x * 2^k, y * 2^k) ≡ rem(x, y) * 2^k` | all six values normal |

The encoding shifts across `2^ebits - 3` positions, about 2045 bits for
binary64. Remainder campaigns therefore run on toy formats, binary16,
bfloat16, and binary32, with a short list of binary64 vectors.

### 4.10 Round to integral

| ID | Relation | Precondition | Modes |
|---|---|---|---|
| I1 | `rint(rint(x)) ≡ rint(x)` | none | all |
| I2 | `rint(x) ≡ x` | `abs(x) >= 2^(sbits-1)`, or `x` is zero or infinite | all |
| I3 | The sign bit of `rint(x)` equals the sign bit of `x` | `x` not NaN | all |
| I4 | `rint_rm(-x) ≡ -rint_mirror(rm)(x)` | none | all |
| I5 | `rint_RTN(x) <= x <= rint_RTP(x)`; their difference is 0 or 1; RTZ equals RTN for `x >= 0` and RTP otherwise; RNE and RNA each equal one of the two | `x` finite | cross-mode |
| I6 | For `x = n + 1/2` with integer `n`: RNE gives the even neighbor, RNA the neighbor of larger magnitude | `abs(x) < 2^(sbits-1)` | RNE, RNA |

I3 pins the case the code comments on at `:98`: `rint(-0.3)` is `-0` under
RNE, RTZ, and RTP.

### 4.11 Comparisons and classification

| ID | Relation | Precondition |
|---|---|---|
| C1 | Exactly one of `x < y`, `x == y`, `x > y` holds | no NaN |
| C2 | `<`, `<=`, `>`, `>=`, `==` are all false and `!=` is true | at least one NaN |
| C3 | `(x < y) = (y > x)`; `(x <= y) = (y >= x)` | none |
| C4 | `(x <= y) = (x < y) or (x == y)` | none |
| C5 | `-0 == +0`; neither is less than the other | none |
| C6 | `(x < y) = (-y < -x)` | none |
| C7 | `(x == y) = (bits(x) = bits(y))` | no NaN, not both zero |
| C8 | For non-negative non-NaN operands, `(x < y) = (bits(x) <u bits(y))`; for negative operands the unsigned order is reversed | same sign |
| C9 | `(x == x) = not isnan(x)` | none |
| C10 | Transitivity of `<` and `<=` | no NaN |
| K1 | Exactly one of isNaN, isInfinite, isZero, isSubnormal, isNormal holds | none |
| K2 | `isfinite(x) = not isinf(x) and not isnan(x)` | none |
| K3 | Class is invariant under `neg` and `abs` | none |
| K4 | `neg(neg(x))` has the bits of `x`; `abs(x)` has the bits of `x` with the sign cleared; `signbit(neg(x)) = not signbit(x)` | none, **NaN included**, compared bit for bit |

### 4.12 Conversions

| ID | Relation | Precondition | Modes |
|---|---|---|---|
| V1 | Widening is exact: `narrow(widen(x)) ≡ x`, and `widen` gives `≡` results in all modes | `x` not NaN | all |
| V2 | Narrowing is monotone and obeys the cross-mode bracketing of A6 | not NaN | all |
| V3 | `narrow(widen(x) op widen(y)) ≡ x op y` for `+ - * /` and `sqrt` | wide `sbits >= 2 * narrow sbits + 2` (binary16 to binary32 and binary32 to binary64 qualify); narrow result normal; gate decides the subnormal range and RNA | RNE, directed |
| V4 | `(intN)(float)n == n` | `n` exactly representable | all |
| V5 | `ubv_to_fp(n) ≡ sbv_to_fp(zext(n, 1))` | none | all |
| V6 | `sbv_to_fp_rm(-n) ≡ -sbv_to_fp_mirror(rm)(n)` | `n != 0`, `n != INT_MIN` | all |
| V7 | Integer to float is monotone; cross-mode bracketing; exact iff the odd part of `n` fits in `sbits` bits | none | all |
| V8 | `(int)x == (int)rint_RTZ(x)`; `(int)(-x) == -(int)x` | `trunc(x)` in range of the target, and for the second also `trunc(-x)` | n/a |
| V9 | `(int)x == 0` for `abs(x) < 1`, for both signed and unsigned targets | none | n/a |
| V10 | `(float)(int)x ≐ rint_RTZ(x)` | `trunc(x)` in range and exactly convertible back | n/a |
| V11 | `bits(from_bits(b)) = b` for every bit pattern `b` | none, NaN payloads included | n/a |

NaN, infinity, and out-of-range inputs to float-to-integer conversion yield a
fresh unconstrained value in `mk_to_bv` (`:653-659`, issue #9448). Tests must
not assert a value there. They assert only that the in-range neighbors of the
boundary are exact, for example the largest double below `2^31`, the value
`-2^31 - 0.5` (in range for `int32` after truncation), and `-0.5` for
unsigned targets.

### 4.13 Representation-level relations

| ID | Relation | Precondition |
|---|---|---|
| P1 | `from_bits(bits(x) + 1)` is the successor of `x`, and `x + minsub` under RTP is `≐` to it | `x` finite, `x >= +0` |
| P2 | For a normal power of two `x`: `succ(x) - x` is twice `x - pred(x)`, both exact | `pred(x)` normal |
| P3 | Largest subnormal `+ minsub ≡` smallest normal, exactly, in all modes | none |
| P4 | Largest finite `+ ulp/2`: infinity under RNE and RNA, largest finite under RTZ and RTN | binary formats |
| P5 | An operation on finite operands never returns a NaN pattern unless §7.2 lists it as invalid | none |

## 5. Oracles

### 5.1 O1: `refloat`, the exact-rational reference (authoritative)

`refloat` is pure Python on the standard library (`fractions.Fraction`,
`math.isqrt`), parametric in `(e, f)`. It has three parts.

```python
def decode(bits, fmt):            # 754-2019 §3.4
    s, E, T = split(bits, fmt)
    if E == fmt.emax_field:
        return NaN(s, T) if T else Inf(s)
    if E == 0:
        return Finite(s, Fraction(T, 2**fmt.f) * 2**fmt.emin)       # zero or subnormal
    return Finite(s, (1 + Fraction(T, 2**fmt.f)) * 2**(E - fmt.bias))

def round_to_format(sign, q, fmt, rm):    # q: exact nonzero Fraction, |value|
    # 754-2019 §4.3: round as if the exponent range were unbounded,
    # except that the quantum never drops below 2**(emin - f)  (§7.5).
    e = max(floor_log2(q), fmt.emin)
    quantum = Fraction(2) ** (e - fmt.f)
    n, rest = divmod(q, quantum)          # n integer, 0 <= rest < quantum
    half = quantum / 2
    up = {RNE: rest > half or (rest == half and n % 2 == 1),
          RNA: rest >= half,
          RTP: rest > 0 and not sign,
          RTN: rest > 0 and sign,
          RTZ: False}[rm]
    n += up
    if n == 2 ** (fmt.f + 1):             # carry into the next binade
        n, e = n // 2, e + 1
    if e > fmt.emax:                      # §7.4 overflow
        to_inf = rm in (RNE, RNA) or (rm == RTP and not sign) or (rm == RTN and sign)
        return encode_inf(sign, fmt) if to_inf else encode_max(sign, fmt)
    return encode_finite(sign, n, e, fmt), events(rest, half, ...)
```

Each operation then reads as the standard states it: handle NaN and the
§6.1/§7.2/§7.3 cases, compute the exact rational, apply the §6.3 rule if the
exact result is zero, otherwise call `round_to_format`. Square root uses
`isqrt` on a scaled integer and keeps the remainder so the half-quantum
comparison stays exact. Remainder computes `n = round_half_even(x / y)` on
rationals and returns `x - n*y`, which is always representable.

Its authority comes from size and directness. The rounding function is 20
lines and corresponds clause by clause to §4.3, §7.4, and §7.5; a reviewer
can check it in one sitting. No bit-level trick appears anywhere in it, so it
shares no design with `fp_conv.cpp`.

### 5.2 O2 to O4: independent cross-checks of the oracle

| Oracle | What it is | Formats | Modes | Caveats |
|---|---|---|---|---|
| O2 MPFR | C driver using `mpfr_set_emin/emax`, precision `sbits`, and `mpfr_subnormalize` | all six plus toy formats | RNE, RTP, RTN, RTZ | MPFR's `MPFR_RNDA` is "away from zero", not roundTiesToAway, so RNA is out. `mpfr_subnormalize` needs the ternary value passed through correctly. |
| O3 Berkeley SoftFloat 3 with TestFloat | Software IEEE implementation; `testfloat_gen` emits operands with expected results | binary16, binary32, binary64, binary128 | RNE, RNA (`near_maxMag`), RTP, RTN, RTZ | Its `extF80` is real x87 and does not match ESBMC's 96-bit format. |
| O4 Host FPU | C driver with `fesetround`, `volatile` operands, `-frounding-math -ffp-contract=off` | binary32, binary64; binary16 where the host has it | RNE, RTP, RTN, RTZ | Guard against compiler folding; `fma` must be the hardware instruction. |

**Oracle agreement gate.** `oracles/agree.py` runs every generated vector
through O1 and each applicable cross-check. Any disagreement stops the
pipeline *before* ESBMC runs, and we resolve it against the standard text.
An oracle disagreement is never a test failure of fp2bv. Exhaustive
agreement runs cover all unary operations on binary16 and all binary
operations on the three toy formats.

RNA has two independent judges (O1, O3) and the 96-bit format has two (O1,
O2). Every other cell of the format-by-mode grid has at least three.

### 5.3 What differential testing adds, and what it cannot

Differential runs are cheap and good at triage. They do not decide
correctness.

| fp2bv vs oracle | fp2bv on solver A vs solver B | Native FP vs oracle | Reading |
|---|---|---|---|
| differs | agree | agrees | Defect in `fp_convt`. |
| differs | agree | differs the same way | Defect shared with the solver's FP theory (likely the Z3 ancestry). Only the oracle catches this row. |
| differs | disagree | any | Bit-vector solver bug, or a defect in one backend's `mk_*` bit-vector conversion. |
| agrees | agree | differs | Solver FP bug; report upstream. fp2bv did its job as mitigation. |
| agrees | agree | agrees | Pass. |

A fourth subject joins the comparison at no cost: ESBMC's constant folder
`ieee_floatt` (`src/util/arith/ieee_float.h`). The simplifier folds concrete
operands with it before any solver sees them. Running each vector once with
literal operands (folded) and once with pinned nondeterministic operands
(solved) exposes any inconsistency between the two implementations inside
ESBMC, and the oracle says which one is wrong.

## 6. Test generation

All generators emit one JSON object per line:

```json
{"fmt":[8,23],"op":"add","rm":"RNE","in":["0x3f800000","0x33800000"],
 "out":"0x3f800000","nan":false,"events":["tie"],"src":"rb/tie-even","seed":412}
```

`out` and `events` come from O1 after the agreement gate.

### 6.1 Targeted boundary values

Per format, a fixed dictionary of about 60 values per sign: `0`, `minsub`,
`2*minsub`, `maxsub - minsub`, `maxsub`, `minnorm`, `succ(minnorm)`,
`1 - ulp/2`, `1`, `1 + ulp`, `2 - ulp`, `2`, `2^(sbits-1) - 1/2`,
`2^(sbits-1)`, `2^sbits - 1`, `2^sbits`, `max/2`, `pred(max)`, `max`, `inf`;
every power of two from `2^emin` down through the subnormal powers; integer
conversion limits `2^7`, `2^15`, `2^31`, `2^63`, `2^64` and their
predecessors, successors, and half-offsets; and four NaN patterns (quiet
minimum payload, quiet maximum payload, signaling `...0001`, signaling with
only the top payload bit below the quiet bit). The cross product over binary
operations gives about 14,000 pairs per format and mode.

### 6.2 Representation-aware random generation

Uniform random bit patterns almost never hit a subnormal or an exponent
boundary. Uniform random reals almost never hit a tie. The generator instead
draws each field separately:

- **Sign:** uniform.
- **Exponent field:** 30% from `{0, 1, 2, bias-1, bias, bias+1, max-2, max-1, max}`,
  20% within `sbits + 2` of the other operand's exponent (so alignment shifts
  land on the guard, round, and sticky positions), 50% uniform.
- **Significand field:** equal shares of uniform; all zeros; all ones; a
  single set bit; a run of ones from the top; a run of ones from the bottom;
  and "sparse" (three random bits).

### 6.3 Rounding-boundary generation by inverse construction

The rounding logic is exercised by results that sit on or beside a tie. We
construct operands from the desired result:

- **Add, sub.** Pick a result `r` and a mode, take the midpoint `m` between
  `r` and its neighbor, pick `x`, set `y = m - x`, and keep the pair if `y`
  is representable. Then perturb `y` by one unit in the last place in each
  direction for the just-below and just-above cases.
- **Mul.** Pick an odd significand `a`. Choose the low `sbits` bits of the
  product as `100...0` (tie), `011...1`, or `100...01`, and solve for `b`
  with the modular inverse of `a` modulo `2^sbits`. Keep `b` if it fits.
- **Div.** Pick `y` and a target `q` on a near-tie, set `x` to the nearest
  representable value to `q * y`, and let the oracle classify what came out.
  Exact ties need a subnormal quotient, so a second pass scales `x` down
  until the quotient is subnormal.
- **Sqrt.** Square a midpoint `m`, round `m^2` down and up, and use both as
  inputs. The square root of a floating-point number is never an exact
  midpoint, so these are the closest approaches.
- **FMA.** Take a mul tie pair `(x, y)` and set `z` to cancel all but the low
  `k` bits of the product, for `k` in `{1, 2, sbits/2, sbits}`.
- **Conversions.** For narrowing, enumerate wide values whose low
  `wide_f - narrow_f` bits are `100...0`, `011...1`, `100...01`. For integer
  to float, do the same on integers wider than `sbits`.
- **Overflow and underflow thresholds.** Results within two ulps of
  `max + ulp/2`, of `minnorm`, and of `minsub/2`, in every mode.

The generator states what it aimed for in `src`; the oracle's `events` field
records what actually happened. Coverage counts use `events` only.

### 6.4 Metamorphic follow-up generation

For each source vector and each applicable relation, `gen/mutate.py` derives
the follow-up inputs: swap operands (A1, M1, F1), negate and mirror the mode
(A3, M3, D3, F4, I4, V6), scale by `2^k` for every `k` that keeps the
precondition (M4, D4, S3, R8), widen (V1, V3), re-run in the other four modes
(A6, M9, D8, I5), and so on. Relations with a low vacuity rate get a
constructive generator: A5 draws `y` and then `x` uniformly in `[y/2, 2y]`;
S2 draws integers below `2^(sbits/2)`; F5 draws `y` as a power of two.

### 6.5 Symbolic generation

In symbolic mode nothing is generated: operands are unconstrained variables,
the precondition is an assumption, and the solver searches the whole format.
An `UNSAT` answer is a proof of the relation for that format, conditional on
the bit-vector solver. This mode covers only relations, since `refloat`
cannot be expressed inside the query.

## 7. Execution and comparison

### 7.1 Two tiers

**Tier U: unit tests against `fp_convt` directly.** A Catch2 harness under
`unit/solvers/fp2bv/` creates a bit-vector backend, constructs
`fp_convt fp(ctx)` on top of it (the pattern
`unit/solvers/array_conv_bounded_index.test.cpp` uses for the array
flattener), and calls the `mk_smt_*` methods. This tier bypasses the
frontend, the simplifier, and symex, so nothing can fold or rewrite the
operation, and it can instantiate formats C cannot name.

```cpp
// concrete vector: fp2bv must compute exactly the oracle's bits
smt_astt x = bv_const(ctx, v.in[0], fmt), y = bv_const(ctx, v.in[1], fmt);
smt_astt r = fp.mk_smt_fpbv_add(x, y, fp.mk_smt_fpbv_rm(v.rm));
smt_astt ok = v.nan ? fp.mk_smt_fpbv_is_nan(r)
                    : ctx->mk_eq(r, bv_const(ctx, v.out, fmt));
ctx->assert_ast(ctx->mk_not(ok));
REQUIRE(ctx->dec_solve() == smt_convt::P_UNSATISFIABLE);

// symbolic relation A1: no x, y, rm violates commutativity
smt_astt x = fresh(fmt), y = fresh(fmt), rm = fresh_rm_in_0_to_4();
ctx->assert_ast(ctx->mk_not(same(fp.mk_smt_fpbv_add(x, y, rm),
                                 fp.mk_smt_fpbv_add(y, x, rm))));
REQUIRE(ctx->dec_solve() == smt_convt::P_UNSATISFIABLE);
```

`same(a, b)` implements `≡` in bit-vector terms: both NaN, or `mk_eq`.
Constant operands still pass through the backend's own term simplifier; that
is acceptable because the backend's bit-vector semantics are a separate
subject (Section 5.3) and the symbolic mode has no constants to fold.

**Tier E: C programs through the whole tool.** The harness pins operands by
bit pattern through nondeterministic values, so the simplifier cannot fold
the operation and the solver must evaluate the lowered circuit:

```c
unsigned xb = nondet_uint(), yb = nondet_uint();
__ESBMC_assume(xb == XB && yb == YB);
float r = from_bits(xb) OP from_bits(yb);     /* memcpy-based */
assert(to_bits(r) == EXPECT);
```

I ran this shape for seven vectors (a tie, a wrong-tie control, a subnormal
underflow to zero, an overflow, `0/0` against two NaN patterns, and
`(-0) + (+0)`) under `--bitwuzla --fp2bv`, `--bitwuzla`, `--z3 --fp2bv`, and
`--z3` on ESBMC 8.6.0. All 28 verdicts matched the IEEE result except the
NaN-pattern probes, which produced H1.

Production programs batch 64 vectors behind a nondeterministic index, so one
ESBMC run checks 64 vectors and the counterexample names the failing row.
Directed modes come from `fesetround`; RNA from a direct assignment to
`__ESBMC_rounding_mode`, which Phase 0 must confirm is accepted from C.
Every `test.desc` pins `--floatbv --fp2bv` and the solver.

Tier E matters because real verification runs go through
`convert_rounding_mode`, the cast dispatch in `smt_casts.cpp`, and the
simplifier. Tier U cannot see defects in those.

### 7.2 What is compared

| Check | Left | Right | Comparison |
|---|---|---|---|
| Oracle | fp2bv result bits | `refloat` result | `≡` |
| Relation | source output | follow-up output | per relation (`≡`, `≐`, `<=`, rational) |
| Solver consistency | fp2bv on solver A | fp2bv on solver B | bit equality, NaN included |
| Fold consistency | literal operands (folded by `ieee_floatt`) | pinned operands (solved) | `≡` |
| Native differential | native FP result | `refloat` result | `≡`; reported upstream |

### 7.3 What counts as a failure

| Code | Condition | Action |
|---|---|---|
| F-ORACLE | fp2bv bits differ from `refloat` on a non-NaN result, or one is NaN and the other is not | Defect in fp2bv. Minimize, file, add regression. |
| F-MR | A relation's negation is satisfiable under its precondition | Replay the model's operands through `refloat`. If `refloat` confirms both sides should relate, defect in fp2bv. If not, the relation is invalid: fix the catalog and the validity gate that let it through. |
| F-SOLVER | fp2bv results differ between two bit-vector backends | Dump the query, reduce, attribute to backend or solver. |
| F-FOLD | Folded and solved results differ | Oracle decides which side is wrong. |
| F-CRASH | Assertion, abort, or signal inside ESBMC | Defect, with the format and operation that triggered it. |
| INCONCLUSIVE | Timeout or resource limit | Recorded with the time budget. Never counted as a pass. |

Not failures: NaN sign or payload differences (logged in a separate NaN
report; H1 is fixed, so a NaN-bit difference now signals a regression), and any result of an unspecified
float-to-integer conversion.

### 7.4 Minimization

1. **Vector level.** `run/minimize.py` shrinks operands while `refloat` and
   fp2bv still disagree: clear significand bits one at a time, move the
   exponent toward the bias, switch to RNE if the failure survives, and
   retry the same relative bit pattern in the next smaller format. The
   result is usually a pair with one to three set bits.
2. **Program level.** For Tier E failures that do not reproduce in Tier U
   (so the defect is outside `fp_convt`), reduce the C file with C-Reduce
   (the `creduce-reducer` agent). The interestingness test is "ESBMC's
   verdict with `--fp2bv` differs from the oracle's expected verdict", with
   the expected verdict recomputed by `refloat` on every candidate so a
   reduction cannot drift to an unrelated failure.
3. **Formula level.** For F-SOLVER, dump the query through the `--smtlib`
   backend and reduce it with an SMT-LIB delta debugger against the two
   solvers.

Each minimized case becomes `regression/fp2bv/<op>_<event>_<issue>/` with
bit-pattern constants in `main.c`, a `test.desc` that pins `--floatbv --fp2bv`
and the solver, and the expected verdict. Per the repository rule, the test
must be shown to fail on the unfixed binary first.

## 8. Validation metrics

### 8.1 Source coverage of `fp_conv.cpp`

Measured with the existing pipeline (`./scripts/build.sh -b Debug -k ON`,
`llvm-profdata`, `llvm-cov export` on the `esbmc` binary). Take the baseline
from Codecov's `file_report` for `src/solvers/smt/fp/fp_conv.cpp` in Phase 0.

Targets: 100% of the lines in `fp_conv.cpp` reachable from the six C formats,
and a written list of the remainder. Three places are already known to need
synthetic formats or to be dead: the `ebits > sbits` branch of `unpack`
(`:1938-1950`), the width-dependent branches in `mk_to_bv` (`:693-704`) and
the integer-to-float casts (`:975-984`, `:1003-1016`), and the `abort()` in
`mk_is_rm`. Unit-tier coverage does not move the Codecov badge, because the
export names only the `esbmc` binary; the Tier E regression tests carry that
number. Dead branches get reported as dead-code candidates and are not
excluded from the measurement.

As Section 1 explains, this metric saturates early. It is a floor.

### 8.2 Circuit-arm coverage

For each vector, which arm of each `ite` did the operand values select?
`refloat` computes this without touching production code, by evaluating the
same predicates on the decoded operands. The tracked predicates are, per
operation, each special-case guard `c1..c7`, and in `round`: `TINY`,
`sigma < 0`, the sticky bit, each of the 5 x 8 `(mode, last/round/sticky)`
combinations of `mk_rounding_decision`, `SIGovf`, `OVF1`, `OVF2`, the
denormal-exponent selection, and the four overflow-result selections.

Target: every predicate observed both true and false for every operation
that calls `round`, in binary32 and binary64.

### 8.3 Semantic coverage matrix

A cell is `(operation, format, mode, operand-class pair, event)` from
Sections 3.3 and 3.4. Cells that cannot occur (for example `tie` for `sqrt`,
or `overflow` for `rem`) are listed as infeasible with a one-line reason,
after the validity gate confirms no toy-format vector reaches them.

Targets: 100% of feasible cells for binary32 and binary64; 100% of feasible
`(operation, mode, event)` triples for binary16, bfloat16, binary128, and the
96-bit format; exhaustive vectors for the toy formats.

### 8.4 Boundary and special-value coverage

- Every value of the Section 6.1 dictionary appears in every operand
  position of every operation, in every mode.
- All 400 ordered class pairs per binary operation.
- For each format: results equal to `minsub`, `maxsub`, `minnorm`, `max`,
  each infinity, and each zero are produced by every operation that can
  produce them.

### 8.5 Relation coverage

Per relation: validity-gate status, vacuity rate, number of concrete source
vectors on which the precondition held, and the symbolic verdict per format
(`proved`, `violated`, `inconclusive at N seconds`). A relation with zero
non-vacuous executions counts as not run.

Target: every relation of Section 4 executed non-vacuously on at least 10^4
concrete vectors per applicable format, and proved symbolically on the toy
formats and binary16.

### 8.6 Bug-detection effectiveness

Coverage shows what ran. It does not show that a defect would have been
caught, so we measure that directly with a mutant catalog:
`scripts/fp2bv-mt/mutants/*.patch`, about 40 hand-written single-point faults
in `fp_conv.cpp`. Examples: drop `sticky` from `last_or_sticky`; swap
`rm_zero_or_neg` and `rm_zero_or_pos`; change the `sigma_cap` from
`sbits + 2` to `sbits + 1`; replace `mk_bvsle` by `mk_bvult` in `TINY`; make
`inc_taway` equal `inc_teven`; return `pzero` for `(-0) + (-0)`; skip the
final zero-sign repair in remainder; use `mk_bvule` in place of `mk_bvult` in
`mk_smt_fpbv_lt`; drop the `both_zero` disjunct from `mk_smt_fpbv_eq`.

For each mutant we record which tier, which oracle check, and which relation
first detects it. Two scores are reported: the kill rate of the oracle
vectors, and the kill rate of the relations alone. The second number shows
how much the metamorphic layer contributes when no oracle vector applies.

Targets: 100% of mutants killed by the full suite; every mutant that
survives the relations-only run is listed with the reason.

Regression coverage: every confirmed defect has a test under
`regression/fp2bv/`, and reverting its fix makes that test fail.

## 9. Implementation roadmap

Sizes are rough: S is a few days, M one to two weeks, L three weeks or more.

### Phase 0: audit and baseline (S)

- **Tasks.** Re-check every clause number in Section 3 against the standard.
  Probe H2 to H9 with single-purpose programs. Confirm how RNA and a symbolic
  rounding mode reach the solver from C. Record the Codecov baseline for
  `fp_conv.cpp`.
- **Artifacts.** A hypothesis register with a verdict per row; issues for
  the confirmed ones; this document updated.
- **Validation.** Each verdict cites a command and its output.
- **Dependencies.** A built `esbmc` with Bitwuzla, Z3, and Boolector.
- **Risks.** H2 may abort the tool on bfloat16 and block later bfloat16
  phases; fix it first if so.

### Phase 1: `refloat` and the oracle agreement gate (M)

- **Tasks.** Implement `decode`, `encode`, `round_to_format`, the 20
  operations, and event tagging. Write the MPFR and host-FPU drivers and the
  SoftFloat runner. Implement `agree.py`.
- **Artifacts.** `scripts/fp2bv-mt/refloat/`, `scripts/fp2bv-mt/oracles/`.
- **Validation.** Exhaustive agreement on the toy formats for all operations
  and modes each cross-check supports; exhaustive unary agreement on
  binary16; 10^8 sampled binary32 and binary64 vectors against the host FPU
  and SoftFloat. `pylint` clean.
- **Dependencies.** MPFR and SoftFloat available on the CI image.
- **Risks.** `mpfr_subnormalize` misuse makes O2 wrong in the subnormal
  range; the toy-format exhaustive run against O1 exposes that quickly.

### Phase 2: generators and vector files (M)

- **Tasks.** Implement Sections 6.1 to 6.3 and the coverage accounting of
  Sections 8.3 and 8.4.
- **Artifacts.** `scripts/fp2bv-mt/gen/`; checked-in vector files for the
  pull-request tier (about 20,000 vectors, compressed).
- **Validation.** The semantic matrix reaches its Section 8.3 targets with
  the generated set; each infeasible cell has a recorded reason.
- **Dependencies.** Phase 1.
- **Risks.** The multiplication tie solver yields few in-range pairs for
  binary128; fall back to near-ties there and say so in the report.

### Phase 3: Tier U harness and oracle vectors (M)

- **Tasks.** Write `fp2bv_harness.h` (backend setup, constants, `same`,
  fresh variables), the vector-driven test, and the synthetic-format test.
  Register targets in `unit/solvers/CMakeLists.txt` with `new_unit_test`,
  guarded by `ENABLE_BITWUZLA` and `ENABLE_BOOLECTOR`.
- **Artifacts.** `unit/solvers/fp2bv/*.test.cpp`.
- **Validation.** All vectors pass or produce filed F-ORACLE issues. Five
  hand-picked mutants from Section 8.6 each make the suite fail.
- **Dependencies.** Phases 1 and 2.
- **Risks.** Solver time for binary128 division and for remainder. Cap per
  query, report INCONCLUSIVE, and keep those on the weekly tier.

### Phase 4: relation catalog, validity gate, symbolic proofs (L)

- **Tasks.** Encode Section 4 in `mr/catalog.py`. Implement `validate.py`
  and follow-up generation. Emit the symbolic relation tests for Tier U.
- **Artifacts.** `scripts/fp2bv-mt/mr/`, `fp2bv_relations.test.cpp`, a
  generated relation table with gate status and vacuity rates.
- **Validation.** Every relation passes the gate or is narrowed; the invalid
  identities of Section 4.3 are run through the same gate and each is
  *rejected*, which shows the gate can say no.
- **Dependencies.** Phases 1 and 3.
- **Risks.** A "gate decides" relation (A9, S4, V3) fails on a subnormal or
  RNA corner. That is the gate working; narrow the precondition and record
  the counterexample.

### Phase 5: Tier E, differential runs, triage (M)

- **Tasks.** Implement `emit/c_program.py` (batched vectors and symbolic
  relations), `run/campaign.py` (solver matrix, time budgets, seeds), the
  fold-consistency run, and `run/triage.py` implementing the Section 5.3
  table.
- **Artifacts.** `scripts/fp2bv-mt/emit/`, `scripts/fp2bv-mt/run/`, a
  campaign report format (JSON and a Markdown summary).
- **Validation.** A seeded mutant is detected end to end and triaged to the
  right row of the table. A fixed seed reproduces a campaign bit for bit.
- **Dependencies.** Phases 2 and 4.
- **Risks.** Changes to operand pinning could let the simplifier fold the
  operation. The emitter adds a control vector with a wrong expected value
  to every batch; a batch whose control passes is discarded as vacuous.

### Phase 6: mutant catalog and metrics report (M)

- **Tasks.** Write the mutant patches. Script build, run, and scoring.
  Produce the Section 8 report.
- **Artifacts.** `scripts/fp2bv-mt/mutants/`, `metrics.md` generated per
  run.
- **Validation.** Section 8.6 targets met, survivors explained.
- **Dependencies.** Phases 3 to 5.
- **Risks.** Equivalent mutants. Prove equivalence with a symbolic Tier U
  query on a toy format before excusing one.

### Phase 7: CI integration and regression distillation (S)

- **Tasks.** Wire the tiers below. Distill confirmed defects and one
  representative per semantic event into `regression/fp2bv/`, register the
  suite in `regression/CMakeLists.txt`, and list the new tests in
  `regression/COV_TESTS` so the Codecov job runs them.
- **Artifacts.** Workflow changes, the new regression suite, a
  `scripts/fp2bv-mt/README.md`.
- **Validation.** Each tier meets its time budget on CI hardware. The
  coverage gate (`esbmc-coverage-pr`, Mode B) passes on the PR.
- **Dependencies.** All earlier phases.
- **Risks.** The 120-second per-test cap on the DebugOpt job. Keep every
  pull-request test under 30 seconds and mark heavier ones `THOROUGH`.

### CI tiers

| Tier | Workflow | Contents | Budget |
|---|---|---|---|
| Pull request | `ci-pull-request.yml` | Tier U with the checked-in vectors on binary16 and binary32; symbolic relations on toy formats; `regression/fp2bv` CORE tests | under 3 minutes |
| Nightly | `ci-nightly.yml` | Tier U and Tier E random campaign with a fresh logged seed, all six formats, three bit-vector backends, fold consistency, native differential | 60 minutes |
| Weekly | `ci-weekly.yml` | Exhaustive binary16 unary, symbolic relations on binary16 and binary32, mutant catalog, full metrics report | 6 hours |

A nightly failure opens an issue carrying the seed, the minimized vector, and
the triage row. PRs that touch `src/solvers/smt/fp/` carry the
`needs-svcomp-run` label, since they can move competition verdicts.

## 10. Directory layout

```
scripts/fp2bv-mt/
  README.md
  refloat/        formats.py decode.py rounding.py ops.py events.py
  oracles/        mpfr_oracle.c hostfpu_oracle.c softfloat_runner.py agree.py
  gen/            boundary.py random_repr.py rounding_boundary.py mutate.py
  mr/             catalog.py validate.py
  emit/           unit_vectors.py c_program.py
  run/            campaign.py triage.py minimize.py metrics.py
  mutants/        NN-short-name.patch
unit/solvers/fp2bv/
  fp2bv_harness.h
  fp2bv_vectors.test.cpp        oracle vectors, six formats
  fp2bv_relations.test.cpp      symbolic and concrete relations
  fp2bv_formats.test.cpp        toy and synthetic formats, width branches
  vectors/                      generated, checked in, with generator seed
regression/fp2bv/
  <op>_<event>_<id>/main.c
  <op>_<event>_<id>/test.desc   --floatbv --fp2bv, solver pinned
```

## 11. Open decisions

1. **NaN encoding (H1).** Decided: generate the quiet NaN. A program that
   inspects NaN bits would otherwise get a different answer under `--fp2bv`
   than on any target it models. Open question left for maintainers: whether
   NaN sign and payload should become checked properties (Section 7.3 keeps
   them in a separate report).
2. **Toy formats in production code.** H6 bounds the formats `round` handles.
   Either document the bound as a precondition of `fp_convt` or assert it.
3. **Out-of-range rounding modes (H9).** If Phase 0 shows values 5 to 7 can
   reach the solver, pick one behavior and make `round` and
   `mk_rounding_decision` agree on it.
4. **Where the oracle lives.** `refloat` is test tooling in Python. Porting
   it to C++ for Tier U would remove the checked-in vector files, at the cost
   of a second copy to keep reviewed. I would keep one Python oracle and
   generated vectors.
