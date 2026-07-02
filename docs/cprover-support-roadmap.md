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
| Float-classification predicates: `isnan`/`isinf`/`isnormal` operand-wrap crash fix (§4.4, Phase 2, partial) | ✅ (PR TBD) | `cbmc_adapter.cpp` |

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
SUCCESSFUL. **Open follow-up:** selecting a CBMC harness via `--function` still needs
work — today it is consumed by the boilerplate-additions synthesis rather than reaching
the retarget logic; the default `__CPROVER__start` bridge (the common case) is fixed.

### 4.3 Type system: anonymous structs and wide constants (Phase 3)
`cbmc_adapter.cpp::expand_anon_struct` aborts on CBMC's anonymous-aggregate naming
(`tag-#anon#ST[...]`), which the contracts library uses heavily — this is why
`mul_contract.goto` is rejected (identically to the reference). Needs an LL(k) parser for
CBMC's `ST[...]`/`SYM`/`*{...}` type-name grammar (a skeleton exists in the original Rust
`adapter.rs::Anon2Struct`). Separately, the hex→binary constant rewrite goes through
`uint64_t`, so constants wider than 64 bits (e.g. 128-bit) are wrong.

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

Still open: `__CPROVER_assume`/`assert` (only relevant if they surface as expressions
rather than instruction-level ASSUME/ASSERT, unconfirmed), array/quantifier predicates,
IEEE-754 rounding-mode operations, `byte_update`, big-endian byte ops. Needs a systematic
audit of the CBMC `irep_idt` vocabulary against the adapter's wrap-set, not just
gap-by-gap discovery.

**Float-classification predicates, investigated by direct testing against real CBMC
binaries.** `math.h`'s `isnan`/`isinf`/`isnormal` lower (via `__builtin_isnan` etc.) to
CBMC ireps of the same name, which `migrate_expr` already fully supports (unary,
`op0()`) — but none were in the adapter's operand-wrap set, so all three **segfaulted**
(`op0()` on an empty operand list). Fixed by adding `isnan`/`isinf`/`isnormal` (plus
`isfinite`/`nearbyint`/`signbit`, defensive — `migrate_expr` supports all three, but this
corpus never exercises them: `isfinite`/`nearbyint` fall back to an unimplemented-function
nondet return in both CBMC's own model and ESBMC's libm operational model, sidestepping
the exprt path entirely on both sides; `signbit` is CBMC's own naming for `__CPROVER`'s
sign-extraction predicate under the *upstream ESBMC* id — real CBMC binaries use `"sign"`
instead, per §4.4's `isinf` note below, so this is latent-gap protection, not a live fix).
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

**Still open — `isinf`.** No longer segfaults, but still aborts ("migrate expr failed")
because glibc's `isinf` additionally uses CBMC's `"sign"` predicate (a sign-bit
extraction, type `bool`), which has no `migrate_expr` counterpart under that name —
ESBMC's own equivalent is `"signbit"`, typed `int_type()`. A straightforward
id-rename-and-retype was attempted and got past the abort, but surfaced a further,
unresolved SMT-encoding error (Bitwuzla: "expected Boolean term") in
`smt_fp_conv.cpp::convert_signbit`, suggesting the `bool`-vs-`int32` mismatch is not the
only issue. **Next task**: investigate `convert_signbit`'s type handling once the
`sign`→`signbit` rename lands, rather than attempting both at once. Pinned as KNOWNBUG
(`cbmc_isinf`).

### 4.5 Symbol metadata (Phase 2)
The adapter maps a subset of symbol flags (`is_type`, `is_macro`, `is_parameter`, `lvalue`,
`static_lifetime`, `file_local`, `is_extern`). `is_weak`, `is_volatile`, `is_thread_local`,
`is_property`, etc. are dropped. Audit which affect soundness.

### 4.6 Contracts subsystem (Phase 4)
`__CPROVER_contracts_*` (requires/ensures/assigns/frees, `is_fresh`, object/write sets) is a
whole subsystem. ESBMC has its own contracts (`src/goto-programs/contracts/`); the work is
to bridge CBMC's encoding onto it rather than re-implement.

### 4.7 Versioning & robustness (Phase 5)
Only CBMC binary **version 6** is accepted. No graceful handling of other versions, and the
reader `abort()`s on malformed input rather than returning a recoverable error.

### 4.8 Builtin-call rewrites (malloc, libm, ...) never reach CBMC-sourced GOTO (Phase 2) — 🔶 newly diagnosed, not yet fixed
Distinct from §4.4 (expression-id coverage): this is about **instruction-level FUNCTION_CALL
targets**, not expression ireps. Found by direct testing: `sqrtf`/`fabsf`/`ceilf`/`floorf`/
`truncf`/`roundf` all produce `WARNING: no body for function <name>` when loaded via
`--binary`, so symex treats the return value as unconstrained nondet — CBMC reports
SUCCESSFUL on assertions these functions would trivially satisfy (e.g. `sqrtf(x) >= 0`),
ESBMC reports FAILED (a nondet counterexample violates the assertion).

**Root cause, confirmed by inspecting both sides' GOTO output.** ESBMC's own C frontend
never emits a real `sqrtf` function call at all: `clang_c_adjust_expr.cpp` (~line 1278,
`compare_float_suffix(identifier, "sqrt")`) recognises the call **syntactically at parse
time** and rewrites it directly into an `ieee_sqrt` expression — confirmed via
`--goto-functions-only` on native `sqrtf(x)`, which shows `ASSIGN y=ieee_sqrt(...)`, no
`sqrtf` symbol anywhere. CBMC's own goto-binary format has no equivalent expression node:
`goto-instrument --dump-c` / `--show-goto-functions` on a real `sqrtf`-using `.goto` file
both show a genuine `CALL return_value_sqrtf := sqrtf(x)` instruction, resolved only when
`cbmc` itself later runs ("Adding CPROVER library (x86_64)" in its own log) — a library
linked **inside the `cbmc` binary at verification time**, never baked into the `.goto`
file. So `--binary`-loaded CBMC programs and ESBMC's own C-parsed programs take genuinely
different paths to the same operator, and nothing bridges them.

**Ruled out as the fix**: making `esbmc_parseoptions.cpp`'s `synthesize_cprover_additions`
boilerplate *call* `sqrtf` so ESBMC's normal C-frontend linking supplies a body doesn't
work — because there is no body to link (`ieee_sqrt` is an operator, not a library
function), so this produces no observable effect. Tried and reverted.

**Needed fix**: the CBMC adapter needs to recognise `FUNCTION_CALL` instructions whose
callee matches a known libm symbol name (`sqrtf`/`sqrt`/`sqrtl`, `fabsf`/`fabs`/`fabsl`,
etc.) and rewrite them into the equivalent `ieee_sqrt`/`abs`/... expression assigned
directly to the call's return-value target, mirroring `clang_c_adjust_expr.cpp`'s logic
but operating on GOTO-level CALL instructions instead of AST call expressions — a
different code shape than `fix_expression`'s id-based rewriting (which only ever sees
expression ireps, not instruction-level call targets), likely needing its own function in
`cbmc_adapter.cpp` alongside `fix_expression`. Not attempted here — this needs its own
design pass rather than a rushed fix layered onto existing `fix_expression` logic.

**Correction, and a second confirmed instance of the same root cause**: `malloc`/`free`
were first flagged here as a *possibly-different* failure mode ("Incorrect alignment when
accessing data object", assumed to mean `malloc` has a real body). That assumption was
wrong — re-checked with full log output: `malloc`/`free` **also** produce `WARNING: no
body for function malloc`/`free`; the "Incorrect alignment" error is a downstream symptom
of dereferencing the resulting nondet/invalid pointer, not a distinct bug. And `malloc` is
**not** a body-based `libclibs.a` function either — `grep`ing `src/c2goto/library/` for a
real `malloc` definition finds none. Like `sqrtf`, ESBMC recognises `malloc`/`alloca` as a
special case and rewrites the call, but at a *different* stage than `sqrtf`:
`src/goto-programs/builtin_functions.cpp::goto_convertt::do_mem` (`base_name == "malloc"`)
runs during **`goto_convert`** — the AST-code-to-GOTO-instructions lowering pass — turning
a `FUNCTION_CALL` statement into a `side_effect_exprt("malloc", ...)`, which symex knows
how to handle as dynamic allocation. CBMC binaries never go through `goto_convert` at
all: `read_cbmc_goto_object` builds `goto_programt` directly from the already-GOTO-shaped
CBMC irep via `goto_program_irep.cpp::convert()`, a mechanical 1:1 translator with no
builtin-function recognition of any kind. So this is the **same class of gap** as §4.8's
libm functions (a compile-time/convert-time rewrite that CBMC-sourced GOTO instructions
never pass through), not an unrelated bug — but the *fix shape* differs again: `do_mem`
already operates at GOTO-instruction level (unlike `clang_c_adjust_expr.cpp`'s AST-level
`sqrtf` handling), so it may be more directly reusable for a CBMC-adapter equivalent than
the libm case is. Likely not the only such builtin — `goto_convertt`'s other
`do_*`-prefixed special-cases (`free`, `printf`-family, `__ESBMC_*` intrinsics reached via
plain C names) are worth auditing together rather than one at a time.

Unlike `sqrtf`, a clean minimal verdict-mismatch reproducer for `malloc` proved harder to
construct in the time available: `malloc`'s own semantics is inherently nondeterministic
(CBMC's own model allows a may-fail-NULL return, so `assert(p != 0)` alone already reports
`VERIFICATION FAILED` on **both** tools, just for different reasons underneath — CBMC's
deliberate may-fail modelling vs. ESBMC's much wider unconstrained-nondet fallback). No
regression test added for this one; the `sqrtf` KNOWNBUG test and this write-up are enough
to point at the shared root cause.

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
- Recognise known libm `FUNCTION_CALL` targets (`sqrtf`, `fabsf`, ...) and rewrite them to
  their `ieee_*`/intrinsic equivalents, the instruction-level counterpart to §4.4's
  expression-level rewriting (§4.8).
- **Acceptance:** a curated suite of single-feature CBMC binaries (pointer predicates,
  overflow, byte ops, FP rounding, libm calls) all verify to the CBMC verdict.

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
