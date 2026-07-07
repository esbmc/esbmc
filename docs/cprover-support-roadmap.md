# CPROVER (CBMC) Support — Roadmap

**Status:** IN PROGRESS
**Branch:** `cbmc-goto-frontend` (foundation landed; not yet merged to `master`)
**Date:** 2026-06-10
**Related:** PR #2443 (CPROVER migration compatibility), the `verify-rust-std` / Kani initiative

---

## 1. Goal

Let ESBMC **read a CBMC goto-binary directly** (`esbmc --binary prog.goto`) and verify
it with results consistent with CBMC's semantics — no external conversion step, no manual
library linking. This subsumes and replaces the out-of-process
[goto-transcoder](https://github.com/esbmc/goto-transcoder) (Rust) converter, which is the
current reference oracle.

Two driving consumers:

1. **`verify-rust-std` / Kani.** Kani emits CBMC goto-binaries for Rust proof harnesses;
   ESBMC verifies them as a second, independent engine.
2. **General CBMC interop.** Ingesting GOTO produced by `goto-cc` / `goto-instrument`.

"Fully support" = a CBMC binary that CBMC verifies (SUCCESS/FAIL) produces the **same
verdict** in ESBMC, for the C and Rust-derived subset exercised by the regression corpus.

---

## 2. Pipeline

```
  CBMC .goto (0x7f 'G' 'B' 'F', v6)
        │
        ▼
  read_cbmc_goto_object          parse_cbmc_goto: varint words, S/N/C irep refs,
  (src/goto-programs/            symbol + function tables -> intermediate structs
   read_cbmc_goto_object.cpp)              │
        ▼                                  ▼
  cbmc_adapter                   rewrite CBMC irep conventions -> ESBMC:
  (src/goto-programs/            fix_type / fix_expression / to_esbmc_irep,
   cbmc_adapter.cpp)             target renumbering, struct-tag cache
        │
        ▼
  symbolt::from_irep  +  convert(irept, goto_programt)   (existing ESBMC consumers)
        │
        ▼
  ┌─ ESBMC additions ───────────────────────────────┐   auto-synthesised in-process when a
  │ __ESBMC_main entry wrapper + CPROVER intrinsic   │   CBMC binary is detected
  │ bodies, via the C frontend on a boilerplate TU   │   (esbmc_parseoptions.cpp)
  └──────────────────────────────────────────────────┘
        │
        ▼
  goto_functionst  ──►  goto-program processing  ──►  symex / solver  ──►  verdict
```

Format detection is by magic header in `goto_binary_reader::read_goto_binary`
(`0x7f GBF` → CBMC path; `GBF` → native ESBMC path). The on-disk irep grammar is shared
heritage, so the CBMC reader is a near-clone of `irep_serializationt` differing only in the
word encoding (CBMC: 7-bit LEB128 varint; ESBMC: big-endian 32-bit), the header/version,
and the symbol/function table layout.

---

## 3. Current state (landed on `cbmc-goto-frontend`)

| Area | Status | Where |
|------|--------|-------|
| CBMC binary reader (varint, header v6, S/N/C refs, symbol/function tables) | ✅ | `src/goto-programs/read_cbmc_goto_object.{h,cpp}` |
| Adapter: `fix_type`, `fix_expression`, `to_esbmc_irep`, target renumbering, struct-tag cache | ✅ | `src/goto-programs/cbmc_adapter.{h,cpp}` |
| Format auto-detect dispatch | ✅ | `src/goto-programs/goto_binary_reader.cpp` |
| Auto-link ESBMC additions (in-process synthesis; `--no-cprover-additions` opt-out) | ✅ | `src/esbmc/esbmc_parseoptions.cpp`, `src/esbmc/options.cpp` |
| CPROVER irep → intrinsic-call migration (overflow_result, r_ok) | ✅ (PR #2443) | migration + `regression/goto-transcoder/` |
| Unit tests (varint/string/header, real v6 parse, load into context/goto_functions) | ✅ | `unit/goto-programs/read_cbmc_goto_object.test.cpp` |
| Parity harness vs goto-transcoder reference | ✅ | `goto-transcoder/scripts/esbmc_parity.sh` |
| CBMC→ESBMC instruction-type mapping table (§4.1, Phase 1) | ✅ (PR #5717) | `cbmc_adapter.{h,cpp}::map_cbmc_instruction_type` |
| Entry-point bridging: `__ESBMC_main` dispatches into `__CPROVER__start` (§4.2, Phase 1) | ✅ (PR #5719) | `esbmc_parseoptions.cpp::retarget_esbmc_main` |
| Pointer predicates: `pointer_offset` operand-wrap crash fix, `w_ok`/`rw_ok` stubs (§4.4, Phase 2, partial) | ✅ (PR #5737) | `cbmc_adapter.cpp`, `migrate.cpp` |
| Float-classification predicates: `isnan`/`isinf`/`isnormal` crash fix + verdict parity (ieee arith promotion, `sign`→`signbit(x)!=0` rewrite) (§4.4, Phase 2) | ✅ (PR #5741) | `cbmc_adapter.cpp` |
| Pointer subtype double-wrap fix (§4.3) — every local pointer DECL without an immediate initializer was silently downgraded to `void*` | ✅ (PR #5750) | `cbmc_adapter.cpp::fix_type` |
| Builtin-call rewrite for `malloc`/`sqrtf` FUNCTION_CALLs (§4.8, Phase 2) | ✅ (PR #5750) | `cbmc_adapter.cpp::fix_builtin_call` |
| Builtin-call rewrite for `alloca`/`__builtin_alloca` FUNCTION_CALLs → `side_effect("alloca")` (§4.8, Phase 2) | ✅ (PR #5793) | `cbmc_adapter.cpp::fix_builtin_call` |
| Builtin-call rewrite for `free` FUNCTION_CALLs → OTHER `free` codet (deallocation, use-after-free/double-free detection) (§4.8, Phase 2) | ✅ (PR #5792) | `cbmc_adapter.cpp::fix_builtin_call` |
| Builtin-call rewrite for `fabs`/`fabsf`/`fabsl` FUNCTION_CALLs → `abs` expr (§4.8, Phase 2) | ✅ (PR #5789) | `cbmc_adapter.cpp::fix_builtin_call` |
| Libm body bridge: `ceil`/`floor`/`trunc`/`round` (+`f`/`l`) resolve to the operational-model bodies (§4.8, Phase 2) | ✅ (PR #5814) | `esbmc_parseoptions.cpp::link_cbmc_libm_bodies` |
| Libm body bridge extended to `copysign`/`fmin`/`fmax`/`fdim` (+`f`/`l`) (§4.8, Phase 2) | ✅ (PR #5815) | `esbmc_parseoptions.cpp::link_cbmc_libm_bodies` |
| Builtin-call rewrite for `realloc` FUNCTION_CALLs → `(ptr==NULL)?malloc:realloc` conditional (§4.8, Phase 2) | ✅ (PR #5794) | `cbmc_adapter.cpp::fix_builtin_call` |
| Builtin-call rewrite for `nearbyint`→`nearbyint` / `fma`→`ieee_fma` FUNCTION_CALLs (§4.8, Phase 2) | ✅ (PR #5796) | `cbmc_adapter.cpp::fix_builtin_call` |
| Expression rewrite for `ieee_float_notequal` → `notequal` (float `!=`; §4.4, Phase 2) | ✅ (PR #TBD) | `cbmc_adapter.cpp::fix_expression` |

**Verified today:** every pre-built CBMC binary in the corpus loads to a goto program
**byte-identical** to the goto-transcoder reference (6/7; the 7th, `mul_contract.goto`, is
rejected identically by both — see §4.3). `esbmc --binary hello-gb.goto` verifies end to
end with no manual library step.

---

## 4. Known gaps

These are the items between "loads and trivially verifies" and "fully supports CPROVER".

### 4.1 Instruction-type fidelity (Phase 1) — ✅ DONE (PR #5717)
ESBMC and CBMC share the `goto_program_instruction_typet` heritage, so values **0–18
agree** (`OTHER=4`, `SKIP=5`, `ATOMIC_BEGIN=10`, `ASSIGN=13`, `FUNCTION_CALL=16`,
`THROW=17`, `CATCH=18`). They **diverge at 19**: CBMC `INCOMPLETE_GOTO` vs ESBMC
`THROW_DECL`. Resolved by `cbmc_adapter.cpp::map_cbmc_instruction_type()`, an explicit,
auditable table: identity for shared kinds, a named diagnostic for `START_THREAD`/
`END_THREAD` (ESBMC models concurrency as intrinsic calls) and `INCOMPLETE_GOTO`. Pinned
by a unit test mapping every shared kind to its ESBMC enumerator.

### 4.2 Entry-point bridging (Phase 1) — ✅ DONE (PR #5719)
CBMC's entry is `__CPROVER__start`; ESBMC's symex looks for `__ESBMC_main`. Previously the
auto-synthesised additions provided an `__ESBMC_main` that called the *boilerplate*
`c:@F@main`, not the CBMC program's `main`/harness — verification ran over an effectively
empty program and could report a spurious SUCCESSFUL. Resolved by
`retarget_esbmc_main()` in `esbmc_parseoptions.cpp`: an explicit `--function` wins,
otherwise a CBMC binary dispatches into `__CPROVER__start`. Regression-tested with real
CBMC 6.8.0 binaries (`cbmc_entry_bridge`, `cbmc_entry_bridge_fail`) — the failing-assert
case is the load-bearing guard, since without bridging it would spuriously report
SUCCESSFUL. **`--function` harness selection — ✅ fixed (PR #5816).** Previously
`--function myharness` aborted with `main symbol 'myharness' not found`: the harness names a
function in the loaded CBMC binary, but `synthesize_cprover_additions` left `config.main` (set
from `--function` by `config.cpp`) in place while compiling the boilerplate TU, so the
boilerplate's own entry-point synthesis (`clang_c_main`) looked for the harness *there* and
failed. Fixed by neutralising `config.main` for the boilerplate compile (mirroring the existing
`cmdline.args` save/restore); the `create_goto_program` retarget then applies `--function` to
the loaded binary. Verdict parity with CBMC tested both ways (`cbmc_harness`,
`cbmc_harness_fail`) — the passing-harness case has a `main` with `assert(0)` to prove the
harness, not `main`, is the entry. This unblocks the `verify-rust-std`/Kani flow, which selects
proof harnesses by name.

### 4.3 Type system: anonymous structs and wide constants (Phase 3)
`cbmc_adapter.cpp::expand_anon_struct` aborts on CBMC's anonymous-aggregate naming
(`tag-#anon#ST[...]`), which the contracts library uses heavily — this is why
`mul_contract.goto` is rejected (identically to the reference). Needs an LL(k) parser for
CBMC's `ST[...]`/`SYM`/`*{...}` type-name grammar (a skeleton exists in the original Rust
`adapter.rs::Anon2Struct`). Separately, the hex→binary constant rewrite goes through
`uint64_t`, so constants wider than 64 bits (e.g. 128-bit) are wrong.

**Pointer subtype double-wrap — ✅ fixed.** Found while chasing an unrelated `malloc`
verdict mismatch (§4.8): `fix_type`'s `pointer` branch, unlike the near-identical `array`
branch three lines below it, wrapped the pointed-to type's positional sub in an
intermediate, id-less group irep before assigning it to `"subtype"`
(`self.add("subtype") = operands;` where `operands.get_sub() = self.get_sub();`, instead
of `self.add("subtype") = self.get_sub()[0];` the way `array` correctly does it).
`typet::subtype()` (`util/type.h`) is a direct `find("subtype")` with no unwrapping, so
`migrate_type` received the wrapper — an irep with no matching case — and silently fell
through to `void`. This affected **any local pointer declared without an initializer that
pins its type from elsewhere** (`int *p;` followed by a later assignment), which is common
enough that it was previously undiscovered simply because nothing had exercised a
`malloc`-then-typed-write pattern far enough to notice the pointer was void* the whole
time. Fixed by mirroring the `array` branch's direct assignment exactly.

### 4.4 Intrinsic & expression coverage (Phase 2) — 🔶 IN PROGRESS
`fix_expression` recognises a fixed set of expression ids that get their CBMC-raw operands
wrapped into the `"operands"` named-sub `exprt::operands()` expects; anything missing from
that set either passes through unwrapped (silent downstream breakage) or, if the id also
has no `migrate_expr` handler, aborts. Concrete gaps found and fixed by direct testing
against real CBMC binaries: `pointer_offset` was missing from the wrap-set despite
`migrate_expr` already supporting it — **this caused a segfault**, not a clean error, since
`migrate_expr`'s unary-operand access read past an empty operand list. `w_ok` and `rw_ok`
(CBMC's `r_or_w_ok_exprt` family alongside the pre-existing `r_ok`: `__CPROVER_w_ok` is the
writable-object-check counterpart, `__CPROVER_rw_ok` the combined read+write check — CBMC's
own typechecker builds all three from the same node type, distinguished only by id, per
`c_typecheck_expr.cpp`) had no support at all; both now get the same treatment as the
pre-existing `r_ok` stub (unconditionally `true` — a known-unsound placeholder,
`// FUTURE: call __ESBMC_r_ok / __ESBMC_w_ok / __ESBMC_rw_ok`; the `w_ok`-vs-CBMC-FAILED
case is a KNOWNBUG regression, `cbmc_w_ok_false`, mirroring the pre-existing `r_ok_false`
limitation). `same_object` was checked and needs no change — CBMC's typechecker desugars it
at parse time into `pointer_object(a) == pointer_object(b)`, so it never reaches the adapter
as a `same_object`/`same-object` node in the first place.

**Float inequality `ieee_float_notequal` — ✅ landed.** CBMC represents a float `!=`
as an `ieee_float_notequal` irep (IEEE-754 semantics: `NaN != NaN` is true), the exact
counterpart of the already-handled `ieee_float_equal`. But only `ieee_float_equal` had an
adapter rewrite (`→ "="`); `ieee_float_notequal` had **no** `migrate_expr` handler, so any
CBMC binary containing a float `!=` **aborted** with `migrate expr failed` — including every
libm model that guards on `x != x` (e.g. `exp`'s `isnan`/`isfinite` inline checks, which is
how it first surfaced). Fixed in `fix_expression` by rewriting `ieee_float_notequal` to
ESBMC's native `notequal`, whose floatbv SMT encoding already implements IEEE semantics
(verified: `float n=0.0f/0.0f; assert(n != n);` verifies SUCCESSFUL natively) — so the
rewrite is faithful, not just crash-avoiding. Mirrors the `ieee_float_equal → "="` line
exactly; `notequal` is already in the operand-wrap set. Verdict parity tested both directions
(`cbmc_float_ne` SUCCESSFUL / `cbmc_float_ne_fail` FAILED) plus a NaN case that pins the IEEE
semantics rather than mere crash-avoidance (`cbmc_float_ne_nan`: `n != n` on `n = 0.0f/0.0f`
verifies SUCCESSFUL — a bitwise-equality `notequal` would report FAILED here), dual-solver
(Bitwuzla + Z3).

Still open: `__CPROVER_assume`/`assert` (only relevant if they surface as expressions
rather than instruction-level ASSUME/ASSERT, unconfirmed), array/quantifier predicates,
IEEE-754 rounding-mode operations, `byte_update`, big-endian byte ops. Needs a systematic
audit of the CBMC `irep_idt` vocabulary against the adapter's wrap-set, not just
gap-by-gap discovery.

Confirmed **not** a gap: the `printf` family. CBMC inlines its own
`<builtin-library-printf>` model (a bodied function returning `__VERIFIER_nondet_int`), so
`printf` reaches the adapter as a *bodied* function, not a bodyless external — ESBMC loads
it and matches CBMC's verdict with no rewrite. §4.8's speculation that it shares the
bodyless-external shape does not hold for CBMC 6.8.0.

**Float-classification predicates, investigated by direct testing against real CBMC
binaries.** `math.h`'s `isnan`/`isinf`/`isnormal` lower (via `__builtin_isnan` etc.) to
CBMC ireps of the same name, which `migrate_expr` already fully supports (unary,
`op0()`) — but none were in the adapter's operand-wrap set, so all three **segfaulted**
(`op0()` on an empty operand list). Fixed by adding `isnan`/`isinf`/`isnormal` (plus
`isfinite`/`nearbyint`/`signbit`, defensive — `migrate_expr` supports all three, but this
corpus never exercises them: `isfinite`/`nearbyint` fall back to an unimplemented-function
nondet return in both CBMC's own model and ESBMC's libm operational model, sidestepping
the exprt path entirely on both sides; `signbit` is ESBMC's own id for the
sign-extraction predicate — real CBMC binaries emit `"sign"`, which the adapter now
renames to `"signbit"` for exactly the `isinf` path documented below).
Segfault→crash-free confirmed for `isnan`/`isinf`/`isnormal` via real CBMC binaries.

**Float-arithmetic type promotion — ✅ landed.** `isnan` didn't reach CBMC's verdict
because of an *unrelated* Phase 2 gap: CBMC emits float division as a plain `"/"` node,
and `migrate_expr`'s `"/"` handler (`exprt::div` → `div2tc`) is type-blind, always
building the generic (non-IEEE) division regardless of operand type. ESBMC's own C
frontend avoids this by promoting `"/"`/`"+"`/`"-"`/`"*"` to
`"ieee_div"`/`"ieee_add"`/`"ieee_sub"`/`"ieee_mul"` whenever the type is `floatbv`
(`clang_c_adjust_expr.cpp::adjust_float_arith`); the CBMC adapter had no equivalent
promotion, so `goto_check.cpp`'s division-by-zero property (which explicitly skips
`ieee_div`, "as it's defined behavior") wrongly fired on CBMC-sourced float division.
Fixed by porting the same type-driven promotion into `fix_expression`: `isnan` now
matches CBMC's verdict exactly (reclassified `cbmc_isnan` KNOWNBUG→CORE), and general
float arithmetic from CBMC binaries is verdict-tested directly (`cbmc_float_arith`,
`cbmc_float_arith_fail`). Contained to the CBMC-binary path only — `fix_expression` is
never reached by ESBMC's native frontends, so this cannot regress native C/C++/Python.
Incidentally, promoting `+`/`-`/`*` (not just `/`) also gives CBMC-sourced float
arithmetic the overflow/NaN instrumentation `goto_check.cpp` applies to `ieee_*` nodes,
which the untyped generic path silently skipped entirely (not wrongly instrumented —
just uninstrumented). Two scope cuts, both intentional and low-risk, left for follow-up:
- **SIMD/vector floats.** `adjust_float_arith`'s `is_vector() && subtype().is_floatbv()`
  fallback isn't replicated — a CBMC float-vector `"+"` node has `type: vector`, not
  `floatbv`, so the promotion simply doesn't fire (identical to pre-fix behaviour, no
  regression, just unaddressed). No CBMC-vector-binary test corpus exists yet to verify
  against.
- **Rounding-mode symbol dependency.** The promoted `ieee_*` nodes rely on
  `migrate_expr` defaulting to `c:@__ESBMC_rounding_mode` when no explicit
  `rounding_mode` operand is present — that symbol isn't defined by the CBMC
  reader/adapter itself, only by `esbmc_parseoptions.cpp`'s `synthesize_cprover_additions`
  step, which every normal `--binary` invocation runs. `--no-cprover-additions` skips it,
  but currently fails earlier for an unrelated reason (`main symbol not found`) before
  this would matter — so no live bug today, but a latent gap if that unrelated failure is
  ever fixed independently. `src/ld-frontend/ir_gen/ld_converter.cpp` already solves the
  identical problem for a different frontend by defining the symbol itself, guarded by
  `find_symbol`, if absent — the CBMC adapter should do the same rather than depend on an
  unrelated driver step always running first.

**`isinf` sign-bit predicate — ✅ landed.** glibc's `isinf` (via `__builtin_isinf_sign`)
additionally uses CBMC's `"sign"` predicate — a sign-bit extraction typed `bool`, used
directly as the condition of the classifier's `x ? -1 : 1` ternary. It has no
`migrate_expr` counterpart under that name; ESBMC's own equivalent is `"signbit"`, but
that irep is structurally fixed to `int32` (`ESBMC_DEFINE_OVERFLOW_INT32_1OP`), returning
1 iff the sign bit is set. The root cause was purely this `bool`-vs-`int32` type mismatch:
feeding an `int32` where the ternary condition demands a `bool` (rename-only reproduces it
as `goto_check.cpp`'s `is_bool_type(i.cond)` assertion — the int32 signbit sitting in the
ternary's condition slot). Fixed entirely in the adapter's `fix_expression`: CBMC's
bool-typed `sign(x)` is rewritten to `notequal(signbit(x), 0)` — exactly the form ESBMC's
own C frontend emits for a sign-bit test in boolean context
(`clang_c_adjust_expr.cpp::__builtin_isinf_sign` wraps `signbit` in `typecast(_, bool)`,
which is the same predicate). `signbit2t`'s well-exercised `int32` SMT encoding
(`convert_signbit`) is left untouched, so the earlier `retype`-based attempt's downstream
Bitwuzla error ("expected Boolean term") never arises.

The fix **must** be CBMC-scoped, not in the shared `migrate_expr`: a `migrate`-level "if
the signbit exprt is `bool`-typed, wrap in `!= 0`" gate looks equivalent but *regresses
native float tests* (`Float_lib1`, `Float-no-simp9` crash in Bitwuzla `mk_eq` on a
width mismatch). ESBMC's own libm models routinely produce bool-typed `signbit` ireps too
(the `typecast(signbit, bool)` above, folded), and baseline relies on `migrate` collapsing
them to the bare `int32` node — native parents accept the `int32`, only CBMC's strict
ternary condition does not. Since the two are structurally identical (bool-typed `signbit`,
float operand), `migrate` cannot tell them apart; `fix_expression` runs only on the CBMC
`--binary` path, so doing the rewrite there is the only way to fix `isinf` without
perturbing native handling. `isinf` now matches CBMC's verdict exactly (reclassified
`cbmc_isinf` KNOWNBUG→CORE; negative direction covered by `cbmc_isinf_fail`).

### 4.5 Symbol metadata (Phase 2)
The adapter maps a subset of symbol flags (`is_type`, `is_macro`, `is_parameter`, `lvalue`,
`static_lifetime`, `file_local`, `is_extern`). `is_weak`, `is_volatile`, `is_thread_local`,
`is_property`, etc. are dropped. Audit which affect soundness.

### 4.6 Contracts subsystem (Phase 4)
`__CPROVER_contracts_*` (requires/ensures/assigns/frees, `is_fresh`, object/write sets) is a
whole subsystem. ESBMC has its own contracts (`src/goto-programs/contracts/`); the work is
to bridge CBMC's encoding onto it rather than re-implement.

### 4.7 Versioning & robustness (Phase 5) — 🔶 malformed-input recovery landed
Only CBMC binary **version 6** is accepted (a wrong version, like a non-magic header, is
already a clean `log_error` + `return true`). The low-level reader no longer `abort()`s on
malformed input: an over-wide varint (>32 bits), an over-long varint (>64 shift bits), and an
unterminated irep now set a `cbmc_irep_readert::failed()` flag that short-circuits the rest of
the parse (subsequent reads no-op, the S/N/C child loops stop) and surfaces through
`parse_cbmc_goto`'s bool return as a recoverable error rather than crashing the whole process
(PR #5811). Pinned by unit tests for each malformed shape plus a truncated-binary parse.
The symbol/function/instruction table counts are also bounded against the bytes left in the
stream before `reserve()` runs — each element is ≥1 byte, so a count larger than the remaining
input is corrupt and rejected rather than driving a multi-gigabyte allocation or a
multi-billion-iteration spin (PR #5812). **Still open:** multi-version tolerance (accept/adapt
versions other than 6).

### 4.8 Builtin-call rewrites (malloc, libm, ...) never reach CBMC-sourced GOTO (Phase 2) — 🔶 `malloc`/`sqrtf`/`alloca`/`free`/`fabsf`/`realloc`/`nearbyint`/`fma` landed, family audit still open
Distinct from §4.4 (expression-id coverage): this is about **instruction-level
FUNCTION_CALL targets**, not expression ireps. ESBMC's own C frontend never emits a real
`malloc`/`sqrtf` function call at all — it recognises these calls **syntactically** and
rewrites them at conversion time: `sqrtf`/`sqrt`/`sqrtl` at AST-adjust time
(`clang_c_adjust_expr.cpp`, `compare_float_suffix`), `malloc`/`alloca` at
AST-to-GOTO-lowering time (`goto-programs/builtin_functions.cpp::do_mem`, during
`goto_convert`). CBMC binaries never go through either pass — `read_cbmc_goto_object`
builds `goto_programt` directly from the already-GOTO-shaped CBMC irep via
`goto_program_irep.cpp::convert()`, a mechanical 1:1 translator with no builtin
recognition — so these calls previously surfaced as bodyless externals at symex time,
returning unconstrained nondet and diverging from CBMC's verdict.

**Fixed**: `cbmc_adapter.cpp::fix_builtin_call`, called from `instruction_to_esbmc_irep`
before the generic FUNCTION_CALL handling runs, recognises `malloc` and `sqrtf`/`sqrt`/
`sqrtl` by callee name and rewrites the instruction's `code` from a `function_call`
statement into the `assign` shape the native pipeline would have produced — a
`side_effect_exprt("malloc", ...)` (mirroring `do_mem`'s own char-element-type fallback,
sound since `sizeof(T)` is always pre-folded to a byte count by the time a `.goto` file
exists) or an `ieee_sqrt` expression respectively. The instruction's own `typeid` is
overridden to `ASSIGN` to match (§4.1's shared numbering means `13` is the same value on
both sides). All six `malloc`/pointer verdict-parity probes and two `sqrtf` probes built
during this work now match CBMC exactly (`cbmc_malloc`, `cbmc_malloc_fail`,
`cbmc_malloc_large`, `cbmc_ptr_decl`, `cbmc_sqrtf`, `cbmc_sqrtf_fail`).

Two further bugs surfaced and were fixed along the way, both real independent of this
rewrite mechanism itself:
- **Dangling reference**: the first `fix_builtin_call` draft held `const irept &lhs`/
  `args` referencing into `code.get_sub()`, then called `code.get_sub().clear()` before
  reading them — UB that "worked" for small constants and segfaulted for others purely by
  luck of allocator reuse. Fixed by copying out (`const irept lhs = sub[0];` etc.) before
  any mutation.
- **Comment fields skip `fix_expression`**: `fix_expression` only ever recurses into
  `get_sub()`/`get_named_sub()`, never `get_comments()` — so a constant embedded directly
  in `#size` (needed for the malloc `side_effect_exprt`) never got its hex→binary
  normalisation, and `migrate_expr` downstream silently computed the wrong array size from
  the raw hex string. This produced correct-looking-but-wrong dereference bounds, not a
  crash — the value `4` happened to be short enough to not visibly break, `100` didn't.
  Fixed by calling `fix_expression` explicitly on the copy embedded in `#size` before use.

**`alloca` — ✅ landed (PR #5793).** CBMC emits `alloca` as a *bodyless* `__builtin_alloca`
`FUNCTION_CALL` external, so ESBMC returned nondet and a *valid* alloca use reported `FAILED`
where CBMC says `SUCCESSFUL`. `alloca` is byte-for-byte the same allocation as `malloc` bar
the statement string, so `build_malloc_rhs` was parametrised into `build_mem_rhs(lhs, args,
statement)` and shared; `fix_builtin_call` rewrites `alloca`/`__builtin_alloca` into the
`side_effect("alloca")` ASSIGN `do_mem(is_malloc=false)` produces (`migrate_expr` →
`sideeffect2t` allockind `alloca` → `symex_alloca`, automatic storage freed on function
return). Tests `cbmc_alloca` (valid, SUCCESSFUL) / `cbmc_alloca_fail` (out-of-bounds, FAILED).

**`free` — ✅ landed (PR #5792).** Unlike the value-returning builtins, CBMC emits `free`
as a *bodyless* `FUNCTION_CALL` external (it inlines its own `<builtin-library-free>` model,
but the callee reaches the adapter as an undefined `free` symbol), so ESBMC returned nondet
and silently dropped the deallocation — **missing use-after-free and double-free** on
CBMC-sourced GOTO (`SUCCESSFUL` where CBMC says `FAILED`). `fix_builtin_call` now rewrites it
into the OTHER-instruction `free` codet `goto_convertt::do_free` produces (`migrate_expr` →
`code_free2t` → `symex_free`, the real deallocation). `free` differs from malloc/sqrt: it has
a **nil lhs** (void return) and maps to **OTHER (4)**, not ASSIGN (13), so it is handled
before the nil-lhs guard and the caller derives the instruction type from the rewritten
statement. Tests `cbmc_free` (clean, SUCCESSFUL) / `cbmc_free_fail` (use-after-free, FAILED);
double-free also verified against CBMC.

**`fabsf` — ✅ landed (PR #5789).** `fabsf`/`fabs`/`fabsl` are rewritten to ESBMC's native
`abs` expr — the same shape `clang_c_adjust_expr.cpp` builds for a syntactically-recognised
call; `migrate_expr`'s abs handler reads `op0()`, so `abs` is added to `fix_expression`'s
operand-wrap set for the argument to reach it.

**`realloc` — ✅ landed (PR #5794).** CBMC emits `realloc` as a *bodyless* `FUNCTION_CALL`
external, so ESBMC returned nondet and a *valid* realloc use reported `FAILED` where CBMC
says `SUCCESSFUL`. More involved than the rest of the family: `do_realloc` produces a
`(ptr == NULL) ? malloc(size) : realloc(ptr)` conditional, not a single side-effect.
`build_realloc_rhs` reconstructs that `if_exprt` at irep level — malloc branch reuses
`build_mem_rhs`, realloc branch is a `side_effect("realloc", ptr)` with the byte size in
`#size` (`migrate_expr` → `sideeffect2t` allockind `realloc` → `symex_realloc`). The null
guard is load-bearing: `symex_realloc` assumes a live source object, so `realloc(NULL, …)`
must route through malloc. Verified against CBMC on valid-grow, out-of-bounds, data
preservation, and `realloc(NULL,n)`. Tests `cbmc_realloc` / `cbmc_realloc_fail`.

**`nearbyint` / `fma` — ✅ landed (PR #5796).** Both are emitted by CBMC as bodyless
`FUNCTION_CALL` externals but — unlike `ceilf`/`floorf`/`truncf`/`roundf` — have native expr
forms `migrate_expr` computes concretely: `nearbyint`/`nearbyintf`/`nearbyintl` → the
`nearbyint` expr, `fma`/`fmaf`/`fmal` → `ieee_fma`. Both default the rounding mode to
`c:@__ESBMC_rounding_mode` like `ieee_sqrt`. `build_sqrt_rhs` was generalised to
`build_unary_fp_rhs(lhs, args, id)` (shared by sqrt, nearbyint, and abs) and `ieee_fma` added
to the operand-wrap set. Tests `cbmc_nearbyint`/`cbmc_fma` (+ `_fail`), all dyadic values.

**Still open**: `malloc`, `sqrtf`/`sqrt`/`sqrtl`, `alloca`/`__builtin_alloca`, `free`,
`fabsf`/`fabs`/`fabsl`, `realloc`, `nearbyint`, and `fma` are recognised — the
malloc/free/alloca/realloc allocation family is now complete. The `printf`-family
`goto_convertt::do_*` special-cases are the same class of gap and share the fix's shape
(`fix_builtin_call` already dispatches on callee name — extending it is additive), but
weren't attempted here to keep each change reviewable. `ceilf`/`floorf`/`truncf`/`roundf` are
**out of shape** — they have no native expr form and route through the libm C operational
model as bodied functions, a distinct mechanism.

**`ceil`/`floor`/`trunc`/`round` (+`f`/`l`) — ✅ landed (PR #5814), via that distinct
mechanism.** CBMC emits them as bodyless `FUNCTION_CALL` externals under their plain names
(`ceil`); ESBMC's operational-model bodies (`libm/{ceil,floor,round,trunc}.c`) exist but are
linked by `add_cprover_library` under the C-frontend-mangled id (`c:@F@ceil`), and the
additions boilerplate referenced nothing so they weren't linked at all — so ESBMC returned
nondet and a valid `ceil(2.3)==3.0` reported `FAILED` where CBMC says `SUCCESSFUL`. Fixed in
`esbmc_parseoptions.cpp`: the additions boilerplate now takes the addresses of the twelve
functions (forcing `add_cprover_library` to link their bodies), and `link_cbmc_libm_bodies`
copies each bodied `c:@F@name`'s body **and type** onto the bodyless plain-named declaration
after the binary loads — `argument_assignments` binds actual args via the copied type's
parameter names, which match the copied body (`goto-symex/symex_function.cpp`). Verdict parity
tested both directions (`cbmc_ceil`/`cbmc_ceil_fail`, `cbmc_round`/`cbmc_round_fail`); the
failing cases confirm the body is really computed (e.g. `round(2.5)==3.0`), not nondet. This
is the reusable path for any bodyless libm/libc external CBMC references; extend the name list
as the corpus grows.

**Extended (PR #5815) to `copysign`/`fmin`/`fmax`/`fdim` (+`f`/`l`)** — the other *exact-result*
libm functions whose operational-model bodies match CBMC's verdict. Deliberately excludes
transcendentals (`sin`/`cos`/`exp`/`log`/`pow`, ...), whose approximations differ between the two
tools, and `fmod`, whose CBMC model is itself nondet (a precise ESBMC body would diverge in the
over-approximation direction). Tests `cbmc_copysign`/`_fail`, `cbmc_fmax`/`_fail`.

**Ruled out as an alternative fix** (for the remaining libm family, from the #5743
diagnosis pass): making `esbmc_parseoptions.cpp`'s `synthesize_cprover_additions`
boilerplate *call* `sqrtf` so ESBMC's normal C-frontend linking supplies a body doesn't
work — because there is no body to link (`ieee_sqrt` is an operator, not a library
function), so this produces no observable effect. Tried and reverted.

---

## 5. Phased plan

Each phase is independently shippable and gated by a concrete acceptance test.

### Phase 1 — Correct execution of straight-line + control-flow harnesses
- Build the CBMC↔ESBMC instruction-type mapping table; remove the blanket cast in
  `goto_program_irep.cpp`'s consumer path for CBMC-sourced ireps (§4.1).
- Entry-point bridging: synthesised `__ESBMC_main` dispatches into the CBMC entry (§4.2).
- **Acceptance:** `mul.goto` and the `verify-rust-std` non-contract harnesses verify with
  the same verdict as CBMC, with no manual flags.

### Phase 2 — Intrinsic & expression coverage
- Enumerate CBMC's expression/intrinsic vocabulary; add a tested mapping table; extend the
  intrinsic-call bodies (the synthesised additions) to cover them (§4.4, §4.5).
- Recognise known builtin `FUNCTION_CALL` targets (`malloc` ✅, `sqrtf` ✅, `alloca` ✅,
  `free` ✅, `fabsf` ✅, `realloc` ✅, `nearbyint` ✅, `fma` ✅, other libm still open) and
  rewrite them to their native-pipeline equivalents,
  the instruction-level counterpart to §4.4's expression-level rewriting (§4.8).
- **Acceptance:** a curated suite of single-feature CBMC binaries (pointer predicates,
  overflow, byte ops, FP rounding, builtin calls) all verify to the CBMC verdict.

### Phase 3 — Full type system
- Port/replace `Anon2Struct` to resolve `tag-#anon#...` aggregates; widen constant handling
  beyond 64 bits (§4.3).
- **Acceptance:** `mul_contract.goto` (and other contracts-library-touching binaries) load
  without rejection; struct/union/array/pointer round-trips match the reference.

### Phase 4 — Contracts
- Bridge `__CPROVER_contracts_*` onto ESBMC's contracts (§4.6).
- **Acceptance:** function-contract harnesses from `verify-rust-std` verify.

### Phase 5 — Hardening & CI
- Multi-version tolerance and recoverable errors instead of `abort()` (§4.7).
- Promote the parity harness to CI; build a CBMC-binary regression corpus (needs `goto-cc`
  in CI, as the goto-transcoder suite already requires).
- **Acceptance:** parity + regression run green in CI on every PR.

---

## 6. Validation strategy

Two oracles, used together:

1. **Reference parity (translation correctness).** `goto-transcoder/scripts/esbmc_parity.sh`
   diffs ESBMC's direct CBMC load (`--no-cprover-additions`, to compare the raw
   reader+adapter output) against the goto-transcoder convert→load path. Any divergence is a
   reader/adapter bug. This is the day-to-day signal while CBMC's own output remains the
   ground truth.
2. **Verdict parity (semantic correctness).** For binaries with assertions, compare the
   ESBMC verdict against CBMC's. This is the Phase-1+ acceptance signal.

Unit tests (`unit/goto-programs/read_cbmc_goto_object.test.cpp`) pin the low-level reader
(varint, escaping, header) and the load-into-context path so regressions are caught without
the full pipeline.

As native coverage grows, the goto-transcoder dependency in the parity loop shrinks; the end
state is CBMC-verdict parity as the sole oracle and goto-transcoder retired.

---

## 7. Risks & open questions

- **Empirical adapter rules.** Several `fix_type`/`fix_expression` rules were reverse-
  engineered from observed CBMC output ("not sure about this", "CBMC can't decide binary vs
  hex"). They must be preserved exactly while extended; the byte-identical parity gate is the
  guardrail.
- **IREP2 acceptance.** The adapter emits legacy `irept`; `migrate_expr` must accept every
  construct. CBMC-specific exprs not yet in IREP2 will surface here first.
- **Entry-point semantics.** Reconciling `__CPROVER__start`/`main` with `__ESBMC_main`/
  `c:@F@main` without breaking the native source path needs care (§4.2).
- **CI cost.** A real CBMC-binary corpus needs `goto-cc` available in CI.

---

## 8. References

- Reader / adapter: `src/goto-programs/read_cbmc_goto_object.{h,cpp}`,
  `src/goto-programs/cbmc_adapter.{h,cpp}`, `src/goto-programs/goto_binary_reader.cpp`
- Consumers reused: `src/util/symbol.cpp` (`from_irep`),
  `src/goto-programs/goto_program_irep.cpp` (`convert`)
- Additions synthesis: `src/esbmc/esbmc_parseoptions.cpp`
  (`has_cbmc_binary_input`, `synthesize_cprover_additions`), `src/c2goto/cprover_library.cpp`
- Tests: `unit/goto-programs/read_cbmc_goto_object.test.cpp`,
  `regression/goto-transcoder/`, `goto-transcoder/scripts/esbmc_parity.sh`
- Reference converter: <https://github.com/esbmc/goto-transcoder> (`adapter.rs`, `cbmc.rs`,
  `bytereader.rs`)
- Prior art: PR #2443 (CPROVER migration compatibility, commit `24d9591a62`)


# Kani support

For Rust, we have a few extra remarks.

## Overview

Analysis of 1,378 Kani benchmark tests run against ESBMC backend reveals systematic failures across 8 major issue categories. This document organizes failures into actionable umbrella issues with concrete minimal reproducer test cases.

**Statistics:**
- Total Tests: 1,378
- Crashes (SIGSEGV rc=139): ~1,200 (87%)
- Aborts (SIGABRT rc=134): ~50 (4%)
- Successfully Parsed: ~128 (9%)

---

## UMBRELLA #1: Transmute/Type Reinterpretation Crashes

**Status**: 🔴 BROKEN
**Impact**: ~200+ tests
**Severity**: 🔴 CRITICAL
**Exit Codes**: rc=139 (both parse & verify)

**Description**:
ESBMC crashes with segmentation fault when processing `transmute` operations and type-casting intrinsics. Both safe `transmute` and unsafe `transmute_unchecked` variants fail during both parse and verify phases. The lowering layer cannot properly represent bit-level type reinterpretation in the C intermediate representation.

**Affected Operations**:
- `mem::transmute()` - type reinterpretation
- `mem::transmute_unchecked()` - unchecked variant
- Type casts across different layouts
- Pointer/reference address preservation

**Test Candidates** (pick one):
```
_RNvNtNtCsfemxtvIyyHd_4core10intrinsics6verify19check_typed_swap_u8
_RNvNtNtCsfemxtvIyyHd_4core10intrinsics6verify24transmute_2ways_i8_to_u8
_RNvNtNtCsfemxtvIyyHd_4core10intrinsics6verify26transmute_2ways_f32_to_i32
_RNvNtNtCsfemxtvIyyHd_4core10intrinsics6verify34transmute_unchecked_2ways_i8_to_u8
_RNvNtNtCsfemxtvIyyHd_4core10intrinsics6verify27check_transmute_ptr_address
_RNvNtNtNtCsfemxtvIyyHd_4core10intrinsics6verify10struct_mod28transmute_2ways_arr_to_tuple
_RNvNtNtCsfemxtvIyyHd_4core10intrinsics6verify29should_succeed_tuple_to_array
```

**Minimal Reproducer**:
```
_RNvNtNtCsfemxtvIyyHd_4core10intrinsics6verify19check_typed_swap_u8
```
(Simplest failing case - basic u8 swap operation)

---

## UMBRELLA #2: Arithmetic Verification Failures

**Status**: 🔴 BROKEN
**Impact**: ~150+ tests
**Severity**: 🔴 CRITICAL
**Exit Codes**: rc=139 (both parse & verify)

**Description**:
ESBMC crashes during symbolic execution of checked/unchecked arithmetic operations, particularly multiply operations with edge cases. Tests for widening multiplication, carrying multiplication, and edge case validation all fail during the verification phase.

**Affected Operations**:
- `u{8,16,32,64,128}::checked_mul()`
- `u{8,16,32,64,128}::unchecked_mul()`
- `i{8,16,32,64,128}::checked_mul()`
- `i{8,16,32,64,128}::unchecked_mul()`
- Widening multiply (`u8::widening_mul_u8()`)
- Carrying multiply (`u8::carrying_mul_u8()`)

**Test Candidates** (pick one):
```
_RNvNtNtCsfemxtvIyyHd_4core3num6verify15widening_mul_u8
_RNvNtNtCsfemxtvIyyHd_4core3num6verify22carrying_mul_u32_small
_RNvNtNtCsfemxtvIyyHd_4core3num6verify22unchecked_mul_u32_edge
_RNvNtNtCsfemxtvIyyHd_4core3num6verify24checked_unchecked_mul_i8
_RNvNtNtCsfemxtvIyyHd_4core3num6verify27unchecked_mul_i128_large_neg
_RNvNtNtCsfemxtvIyyHd_4core3num6verify25widening_mul_u64_mid_edge
```

**Minimal Reproducer**:
```
_RNvNtNtCsfemxtvIyyHd_4core3num6verify15widening_mul_u8
```
(Basic u8 widening multiply - smallest failing case)

---

## UMBRELLA #3: Pointer Operations Crash

**Status**: 🔴 BROKEN
**Impact**: ~60+ tests
**Severity**: 🔴 CRITICAL
**Exit Codes**: rc=139 (both parse & verify)

**Description**:
ESBMC segfaults on pointer manipulation intrinsics (align_offset, read, offset_from, etc.). Crashes occur during C generation phase in the lowering layer, indicating fundamental issues in pointer semantics translation from Rust to C.

**Affected Operations**:
- `*const T::read()` / `*mut T::read()` - unsafe pointer dereference
- `*const T::align_offset()` / `*mut T::align_offset()` - alignment calculation
- `*const T::offset_from()` / `*mut T::offset_from()` - pointer distance
- Pointer alignment verification

**Test Candidates** (pick one):
```
_RNvNtNtCsfemxtvIyyHd_4core3ptr6verify15check_read_u128
_RNvNtNtCsfemxtvIyyHd_4core3ptr6verify21check_align_offset_u8
_RNvNtNtCsfemxtvIyyHd_4core3ptr6verify23check_align_offset_4096
_RNvNtNtCsfemxtvIyyHd_4core3ptr6verify22check_align_offset_zst
```

**Minimal Reproducer**:
```
_RNvNtNtCsfemxtvIyyHd_4core3ptr6verify21check_align_offset_u8
```
(Basic u8 alignment offset - simplest pointer operation)

---

## UMBRELLA #4: Memory Swap Operations Unsupported

**Status**: 🔴 BROKEN
**Impact**: ~20+ tests
**Severity**: 🟠 HIGH
**Exit Codes**: rc=139 (both parse & verify)

**Description**:
`mem::swap` and collection swap operations (Vec, VecDeque) crash during both parsing and verification. These are fundamental memory safety operations that ESBMC cannot currently verify.

**Affected Operations**:
- `mem::swap::<T>()` - primitive type swap
- `Vec::swap()` - vector element swap
- `VecDeque::swap()` - double-ended queue swap
- `mem::swap()` with aggregate types

**Test Candidates** (pick one):
```
_RNvNtNtCsldpw2oyRQaa_5alloc3vec6verify18verify_swap_remove
_RNvNtNtNtCsldpw2oyRQaa_5alloc11collections9vec_deque6verify19check_vecdeque_swap
_RNvNtNtCsfemxtvIyyHd_4core3mem6verify20check_swap_primitive
_RNvNtNtCsfemxtvIyyHd_4core3mem6verify22check_swap_adt_no_drop
```

**Minimal Reproducer**:
```
_RNvNtNtCsfemxtvIyyHd_4core3mem6verify20check_swap_primitive
```
(Primitive type swap - simplest swap operation)

---

## UMBRELLA #5: Float-to-Integer Conversion Crashes

**Status**: 🔴 BROKEN
**Impact**: ~100+ tests
**Severity**: 🟠 HIGH
**Exit Codes**: rc=139 (both parse & verify)

**Description**:
Unchecked float-to-integer conversion operations fail across all precision levels (f16, f32, f64, f128) and all target integer types (i8 through i128, u8 through u128, isize, usize). Both checked and unchecked variants crash.

**Affected Operations**:
- `f16::to_int_unchecked()` / `to_int_checked()`
- `f32::to_int_unchecked()` / `to_int_checked()`
- `f64::to_int_unchecked()` / `to_int_checked()`
- `f128::to_int_unchecked()` / `to_int_checked()`
- All target types: i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize

**Test Candidates** (pick one):
```
_RNvNtNtCsfemxtvIyyHd_4core3num6verify31checked_f32_to_int_unchecked_i8
_RNvNtNtCsfemxtvIyyHd_4core3num6verify32checked_f32_to_int_unchecked_i32
_RNvNtNtCsfemxtvIyyHd_4core3num6verify32checked_f64_to_int_unchecked_i64
_RNvNtNtCsfemxtvIyyHd_4core3num6verify33checked_f16_to_int_unchecked_i128
_RNvNtNtCsfemxtvIyyHd_4core3num6verify34checked_f128_to_int_unchecked_usize
```

**Minimal Reproducer**:
```
_RNvNtNtCsfemxtvIyyHd_4core3num6verify31checked_f32_to_int_unchecked_i8
```
(f32 to i8 conversion - simplest float-to-int case)

---

## UMBRELLA #6: Slice and Collection Operations Crash

**Status**: 🔴 BROKEN
**Impact**: ~15+ tests
**Severity**: 🟠 HIGH
**Exit Codes**: rc=139 (both parse & verify)

**Description**:
Slice operations (reverse) and Option/Vec conversions fail in the verifier. These are fundamental data structure operations that ESBMC cannot abstract properly.

**Affected Operations**:
- `[T]::reverse()` - slice reversal
- `Option<T>::as_slice()` - option to slice conversion
- Slice borrowing and manipulation

**Test Candidates** (pick one):
```
_RNvNtNtCsfemxtvIyyHd_4core5slice6verify13check_reverse
_RNvNtNtCsfemxtvIyyHd_4core6option6verify15verify_as_slice
```

**Minimal Reproducer**:
```
_RNvNtNtCsfemxtvIyyHd_4core5slice6verify13check_reverse
```
(Basic slice reverse - simplest collection operation)

---

## UMBRELLA #7: Contract System Causes Aborts (Parse Succeeds, Verify Fails)

**Status**: 🟠 PARTIAL
**Impact**: ~50+ tests
**Severity**: 🟠 HIGH
**Exit Codes**: rc=0 (parse), rc=134 (verify - SIGABRT)

**Description**:
Tests that successfully parse (rc=0) generate valid GOTO programs, but verification phase terminates with SIGABRT. Root cause appears to be:
1. Contract registration system losing metadata (200+ "dropping thread_local" warnings)
2. Closure handling in `kani_force_fn_once` / `kani_apply_closure`
3. Panic scenario verification assertions failing
4. Resource exhaustion or assertion violations in verification engine

This is distinct from umbrella issues #1-6 because parse phase succeeds.

**Affected Operations**:
- `Duration::checked_sub()` with contracts
- Panic scenario verification (`*_panics` variants)
- Any operation with higher-order contracts or closures

**Test Candidates** (pick one):
```
_RNvNtNtCsfemxtvIyyHd_4core4time15duration_verify20duration_checked_sub
_RNvNtNtCsfemxtvIyyHd_4core4time15duration_verify27duration_checked_sub_panics
```

**Minimal Reproducer**:
```
_RNvNtNtCsfemxtvIyyHd_4core4time15duration_verify20duration_checked_sub
```
(Duration checked subtraction - parse succeeds, verify aborts)

**Notable Warning Pattern**:
```
WARNING: CBMC adapter: dropping 'thread_local' on symbol _R...kani_register_contract...
WARNING: CBMC adapter: dropping 'thread_local' on symbol _R...kani_force_fn_once...
WARNING: CBMC adapter: dropping 'thread_local' on symbol _R...tmp_statement_expression
```
(Hundreds of these in verify.log for this category)

---

## UMBRELLA #8: Generic Type Instantiation Limits

**Status**: 🟠 PARTIAL
**Impact**: ~50+ tests
**Severity**: 🟡 MEDIUM
**Exit Codes**: rc=139 (both parse & verify)

**Description**:
Tests with deeply nested generic type parameters cause ESBMC to crash during lowering. The C type representation becomes too complex or hits internal limits. Symbol names suggest complex generic instantiation (e.g., `10struct_mod28transmute_2ways_arr_to_tuple` indicates struct-generic-array-transmute combinations).

**Affected Operations**:
- Transmute with generic aggregate types (structs, tuples, arrays)
- Generic closures with Option/Result wrappers
- Deeply nested type parameters in contracts

**Test Candidates** (pick one):
```
_RNvNtNtNtCsfemxtvIyyHd_4core10intrinsics6verify10struct_mod28transmute_2ways_arr_to_tuple
_RNvNtNtNtCsfemxtvIyyHd_4core10intrinsics6verify10struct_mod29transmute_2ways_struct_to_arr
_RNvNtNtNtCsfemxtvIyyHd_4core10intrinsics6verify10struct_mod31transmute_2ways_struct_to_tuple
_RNvNtNtNtCsfemxtvIyyHd_4core10intrinsics6verify6i8_mod28transmute_2ways_arr_to_tuple
_RNvNtNtNtCsfemxtvIyyHd_4core10intrinsics6verify7arr_mod28transmute_2ways_arr_to_tuple
```

**Minimal Reproducer**:
```
_RNvNtNtNtCsfemxtvIyyHd_4core10intrinsics6verify10struct_mod28transmute_2ways_arr_to_tuple
```
(Struct-generic transmute - represents generic type complexity)

---

## PRIORITY MATRIX FOR INVESTIGATION

### Tier 1 - Highest ROI (affects most tests, likely shared root cause)

| Issue | Tests | Priority | Reason |
|-------|-------|----------|--------|
| **UMBRELLA #1**: Transmute | 200+ | 🔴 P0 | Fundamental unsafe feature, highest test count |
| **UMBRELLA #2**: Arithmetic | 150+ | 🔴 P0 | Common verification goal, large impact |

### Tier 2 - High Impact (medium test count, critical features)

| Issue | Tests | Priority | Reason |
|-------|-------|----------|--------|
| **UMBRELLA #5**: Float→Int | 100+ | 🟠 P1 | Numeric safety, medium test count |
| **UMBRELLA #3**: Pointers | 60+ | 🟠 P1 | Memory safety critical, 60+ tests |

### Tier 3 - Medium Priority (parse succeeds or scope-limited)

| Issue | Tests | Priority | Reason |
|-------|-------|----------|--------|
| **UMBRELLA #7**: Contracts | 50+ | 🟡 P2 | Parse succeeds (different pattern), new info |
| **UMBRELLA #8**: Generics | 50+ | 🟡 P2 | Scope limitation, architectural |

### Tier 4 - Lower Priority (smaller test sets)

| Issue | Tests | Priority | Reason |
|-------|-------|----------|--------|
| **UMBRELLA #4**: Swap | 20+ | 🟡 P3 | Specific operations, lower count |
| **UMBRELLA #6**: Collections | 15+ | 🟡 P3 | Data structures, smallest set |

---

## Recommended Investigation Order

1. **Start with UMBRELLA #1 (Transmute)**:
   - Simplest test: `check_typed_swap_u8`
   - Affects 200+ tests
   - Likely points to fundamental lowering issue

2. **Then UMBRELLA #2 (Arithmetic)**:
   - Simplest test: `widening_mul_u8`
   - Affects 150+ tests
   - May share root cause with #1

3. **Then UMBRELLA #7 (Contracts)**:
   - Different pattern (parse succeeds)
   - Will require different debugging approach
   - May reveal verification-layer issues

4. **Remaining issues** can be addressed after the above are stabilized

---

## Test Execution Notes

All tests are located under the Kani benchmarks output directory, referred to
below as `$KANI_BENCHMARKS_OUT` (e.g. `<kani-benchmarks-root>/out`):
`$KANI_BENCHMARKS_OUT/<test-name>/`

Each test directory contains:
- `parse.log` - output from C generation phase (Kani → ESBMC C)
- `verify.log` - output from symbolic verification phase (ESBMC SMT solving)

**To investigate a specific test**:
```bash
cd "$KANI_BENCHMARKS_OUT/<test-name>"
tail -100 parse.log   # Check for parse phase errors
tail -100 verify.log  # Check for verification phase errors
```

---

## Additional Metadata

**Test Suite Coverage**:
- ✅ core::convert - PARTIAL (type conversions)
- 🔴 core::intrinsics - BROKEN (unsafe ops)
- 🔴 core::mem - BROKEN (memory operations)
- 🔴 core::num - BROKEN (arithmetic)
- 🔴 core::option - BROKEN (option types)
- 🔴 core::ptr - BROKEN (pointer manipulation)
- 🔴 core::slice - BROKEN (slice operations)
- 🟠 core::time - PARTIAL (contracts abort)

**Exit Code Reference**:
- `0` = Success
- `134` = SIGABRT (abort signal - assertion/contract failure)
- `139` = SIGSEGV (segmentation fault - memory access violation)

**Warning Pattern Index**:
- Thread-local loss: "`CBMC adapter: dropping 'thread_local'`" (200+ per test)
- Found in: Umbrella #7 tests (contracts), partially in #8 (generics)
