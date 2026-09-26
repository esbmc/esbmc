# Plan: native fixed-point (_Fract/_Accum/_Sat) support in ESBMC

Fixed-point today is a legacy *float fallback*: `--fixedbv` makes `float`
mean "fixed bit-vector" (`config.ansi_c.use_fixed_for_float`), and nothing
in the pipeline understands Embedded C (ISO/IEC TR 18037) fixed-point
types. The goal is to invert that: floats are always IEEE, and
`_Fract`/`_Accum`/`_Sat` become first-class types lowered to camada's
existing FXP theory.

Current inventory (surveyed 2026-08-09):

- Flags: `--floatbv`/`--fixedbv` (`options.cpp:643-646`);
  `use_fixed_for_float` set in `config.cpp:180`, branched on in **9 files**
  (bmc, smt_solver.h, clang adjust/literals, c_types, config,
  python float literals, cprover_library).
- Libc: every clib variant is built **twice** (multilib line
  `c2goto/CMakeLists.txt:62`: `--fixedbv -D__ESBMC_FIXEDBV` vs
  `--floatbv`); only `libm/sqrt.c` actually guards on `__ESBMC_FIXEDBV`.
  `cprover_library.cpp` selects the variant at load time.
- irep2: `fixedbv_type2t{width, integer_bits}` (`irep2_type.h:424`) and
  `constant_fixedbv2t` exist; the solver already lowers fixedbv
  mul/div/mod/compare to shifted BV ops (`smt_solver.cpp:686-1522`).
  **Missing:** signedness and saturation in the type.
- Clang: `-ffixed-point` is mature in clang 22; our frontend only touches
  `APValue::FixedPoint` in literal evaluation (`clang_c_convert.cpp:5372`)
  — no `FixedPointType` handling in type conversion.
- camada: FXP sort kind + 34 methods already shipped (`mkFXPSort`,
  `mkFXPFromBV/ToBV/ToFXP` + `*Overflow` twins, full arith/compare/shift
  set, `FXPSortData{Width, FracBits, IsSigned}`).
  **Visible gaps already:** no FXP<->FP conversions, no saturating
  operation variants (only overflow predicates).

---

## Phase 1 — remove the float-fallback machinery

Pure deletion; no new semantics. Everything float-related becomes
unconditionally IEEE.

1. Delete `--floatbv`/`--fixedbv` from `options.cpp`; delete
   `use_fixed_for_float` from `config.{h,cpp}` and chase the compiler
   through the 9 branching files (each has a dead branch to fold).
2. c2goto: drop the fixedbv multilib arm (`CMakeLists.txt:62,87,91,100+`),
   halving the clib build; drop the `__ESBMC_FIXEDBV` block in
   `libm/sqrt.c`; simplify the variant selection in
   `cprover_library.cpp`.
3. Solver: `pick_logic()` reads `floatbv` (`camada_conv.cpp`) — the
   int-encoding/fixedbv interplay branches in `smt_solver.cpp`,
   `smt_casts.cpp`, `smt_bitcast.cpp` keep only their floatbv arms *for
   floats*; the fixedbv arms stay but now serve only true fixed-point
   (they become Phase 2's foundation).
4. Tests: the `--fixedbv` suite dies with the flag —
   `cbmc/01_cbmc_Fixedbv*` and the `--fixedbv --ir` combinations
   (e.g. Fixedbv24) tested the float-fallback being removed. Delete them
   (git history preserves them); note it in the commit message.
5. Acceptance: full suite, zero deltas outside the deleted tests; grep
   gate: no `use_fixed_for_float`, no `__ESBMC_FIXEDBV`, no
   `isset("fixedbv")` anywhere.

Risk: low — mechanical, but the 9 files include literal parsing
(`clang_c_convert_literals.cpp`) where the dead branch must be folded
carefully; and `--ir` mode's fixedbv handling (`ir_ieee` reals for
fixedbv) needs its branch kept for Phase 2 or explicitly dropped.

## Phase 2 — the types exist end to end

Smallest vertical slice: declare, assign, add, assert over every TR 18037
type, through camada FXP, both verdicts.

1. **Type design (the one real decision):** extend `fixedbv_type2t` with
   `is_signed` and `is_saturating` (irep2 field addition, migrate both
   directions, field_names) — or a parallel `ufixedbv` type. Recommend
   extension: camada's `FXPSortData` already carries `IsSigned`, and
   saturation is per-type in C but per-operation in the encoding, so the
   type carries it and conversion consults it.
   `_Fract` = `integer_bits == 0`; `_Accum` = `integer_bits > 0`.
2. Frontend: pass `-ffixed-point` to the clang invocation; convert
   `clang::FixedPointType` (all 24 combinations: {short,plain,long} x
   {signed,unsigned} x {_Fract,_Accum} x {,_Sat}) reading width/scale
   from clang's TargetInfo rather than hardcoding; convert fixed-point
   literals (`0.5r`, `1.5uk`, ...) via the existing
   `APValue::FixedPoint` entry.
3. Solver: `convert_sort(fixedbv)` -> `mkFXPSort(width, frac, signed)`;
   arithmetic dispatches to `mkFXPAdd/Sub/Mul/Div/Neg/Shl/Shr`;
   saturating types wrap each op as
   `ite(mkFXP*Overflow(...), SAT_MAX/MIN, mkFXP*(...))` until/unless
   camada grows native `*Sat` variants (report as gap, Phase 3 protocol).
   Comparisons via `mkFXPLt/Le/Gt/Ge/Equal`. Casts fixed<->fixed via
   `mkFXPToFXP(+Overflow)`, fixed<->int via `mkFXPFromBV/ToBV`.
4. Model extraction: counterexample values for FXP sorts (scale the raw
   BV by 2^-frac for display) — touches `get_by_ast` and
   `c_expr2string`/witness formatting.
5. Tests: `regression/fixedbv/` — per-type smoke (declare/assign/assert,
   one passing + one failing each for a representative six of the 24),
   literal round-trips, wraparound vs `_Sat` clamping at both rails,
   div-by-zero (`mkFXPDivByZero` exists — decide the C-level property).
   **Oracle:** clang -ffixed-point executes natively; every test's
   expected verdict is checked against the compiled binary's actual
   behaviour first (the fmod lesson: never trust the model, not even the
   reference implementation's docs).
6. Acceptance: the new suite green on bitwuzla + z3; full suite no-delta;
   the Phase 1 grep gates still hold.

## Phase 3 — conversions and the camada gap report

The hard semantics live here: TR 18037 4.1.4 conversion rules.

1. Test matrix, oracle-checked as above: fixed<->fixed across
   width/scale/signedness (rounding is implementation-defined — match
   clang's), fixed<->int (truncation toward zero), **fixed<->float/double**
   (the known API hole: camada has no `mkFXPToFP`/`mkFPToFXP`),
   `_Sat` saturation at every conversion boundary, negative zero /
   rail values / the asymmetric `-1.0r`.
2. Fixed<->float: **DONE** (2026-08-09). Camada shipped `mkFXPToFP(e,to,RM)`,
   `mkFPToFXP`, `mkFPToFXPSat`, `mkFPToFXPOverflow` on
   `feat/fxp-fp-conversions`; ESBMC passes RM::ROUND_TO_EVEN for
   fixed->float and picks Sat by destination for float->fixed.
   conv_float_unsupported retired in favour of three value tests. The
   whole round trip -- ESBMC measures semantics, reports demand, camada
   implements, ESBMC consumes -- took one day and produced zero semantic
   disagreements, which is the argument for the oracle protocol.
3. Deliverable: `REPORT-fxp-api-gaps.md` in the camada repo — each gap
   with the ESBMC-side composition used as the spec, a hard test
   instance, and measured cost.
4. Acceptance: conversion suite green; the report delivered; any
   camada-side additions re-validated with the same tests.

## Phase 4 — the library (stdfix.h)

TR 18037 7.18a: `absfx`, `roundfx`, `countlsfx`, the `bitsfx`/`fxbits`
pairs, `mulifx`/`divifx`/`idivfx` families, plus `stdfix.h` itself in
`c2goto/headers/`.

1. Model strategy per function, applying this branch's hard-won rule:
   *intrinsic where the solver has the exact operation, C model where
   composition is clearer* — `absfx` is a one-op intrinsic (like
   `ieee_rem`, with the non-fixed-identifier guard from the `basic21`
   lesson); `roundfx` wires to camada's `mkFXPToFXPRound`,
   which carries the tie direction as a parameter (fpneg-style) —
   camada builds it against its own clang-22 oracle; ESBMC pins the
   direction argument from the same oracle at wiring time and must
   NOT hand-roll the bias composition (width trap); `countlsfx` is a leading-zero variant (camada
   has the tree LZC — possible reuse/gap); `bitsfx`/`fxbits` are
   raw-BV reinterpret (`mkFXPFromRawBV`/`ToRawBV`, already shipped).
2. **The sqrt family (`sqrtuhr`/`sqrtur`/`sqrtulr`/`sqrtuhk`/`sqrtuk`/
   `sqrtulk`)**: not TR-core but shipped by the embedded toolchains this
   work targets (avr-libc et al.), unsigned-only by construction. Camada
   has no `mkFXPSqrt` and should not grow one preemptively: the Phase-1
   deleted `sqrt.c` Babylonian model returns from git history, retyped
   over real unsigned fixed types — a bounded Newton iteration is a clean
   C model, and if it measures badly the camada gap report gets a
   `mkFXPSqrt` entry with the model as the spec (the fp.rem protocol).
   Exact name set and semantics pinned against the target toolchain's
   headers plus the execution oracle when this phase starts, not from
   documentation.
3. Tests: one per function against the native-execution oracle, plus the
   `_fail` twin convention.
4. Acceptance: stdfix suite green; a C program using every stdfix
   function verifies end to end.

---

## Cross-cutting

- **Branch/PR shape:** each phase is its own PR onto `camada` (or master
  once merged), commits per the repo convention, every commit with its
  regression tests. Phase 1 must land before 2 (the flag removal changes
  what `fixedbv_type2t` *means*).
- **Measurement discipline** (from the LZC/A3A4/dict65 record): oracle
  first, equal budgets, pinned cores, probe-verified binaries, clean env
  (`FORCE_COLOR` kills TypeError-pattern tests). Performance work is
  explicitly *out of scope* until correctness lands; no encoding cleverness
  without an instance family.
- **Known hazards:** `_Sat` division overflow (MIN/-1), the `--ir` modes
  (does int-encoding get FXP-as-Real? propose: reject `_Fract` under
  `--ir` in Phase 2, revisit later); goto2c/`c_expr2string` display of
  fixed literals; python frontend float-literal path mentions
  `use_fixed_for_float` (Phase 1 fold must not disturb python floats).
- **Out of scope:** `__fract` GCC spellings beyond what clang maps,
  `_Fract` complex types, DSP intrinsics. The stdfix sqrt family is *in*
  scope (Phase 4) despite being extension-tier — it is what the embedded
  audience actually calls.
