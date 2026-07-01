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

**Verified today:** every pre-built CBMC binary in the corpus loads to a goto program
**byte-identical** to the goto-transcoder reference (6/7; the 7th, `mul_contract.goto`, is
rejected identically by both — see §4.3). `esbmc --binary hello-gb.goto` verifies end to
end with no manual library step.

---

## 4. Known gaps

These are the items between "loads and trivially verifies" and "fully supports CPROVER".

### 4.1 Instruction-type fidelity (Phase 1)
ESBMC and CBMC share the `goto_program_instruction_typet` heritage, so values **0–18
agree** (`OTHER=4`, `SKIP=5`, `ATOMIC_BEGIN=10`, `ASSIGN=13`, `FUNCTION_CALL=16`,
`THROW=17`, `CATCH=18`). They **diverge at 19**: CBMC `INCOMPLETE_GOTO` vs ESBMC
`THROW_DECL` — hence the adapter's `assert(instr_type != 19)`. `convert()` casts the raw
`typeid` straight to the enum, so any divergence above 18, or any instruction whose
`typeid` is not faithfully carried, surfaces in symex as
`GOTO instruction type <N> not handled in goto_symext::symex_step`. Some harnesses
(e.g. `mul.goto --function mul_harness`) hit this **identically via the reference
converter**, so it is a translation-pipeline gap, not specific to the native reader.
Needs: an explicit CBMC→ESBMC instruction-type mapping table + an audit of which CBMC
instruction kinds reach a final binary.

### 4.2 Entry-point bridging (Phase 1)
CBMC's entry is `__CPROVER__start`; ESBMC's symex looks for `__ESBMC_main`. Today the
auto-synthesised additions provide an `__ESBMC_main` that calls the *boilerplate*
`c:@F@main`, not the CBMC program's `main`/harness — fine for smoke runs, wrong for real
verification. Needs: wire the synthesised `__ESBMC_main` to dispatch into the CBMC binary's
entry (`__CPROVER__start`, or the `--function` harness), reconciling the `main` vs
`c:@F@main` symbol-id conventions.

### 4.3 Type system: anonymous structs and wide constants (Phase 3)
`cbmc_adapter.cpp::expand_anon_struct` aborts on CBMC's anonymous-aggregate naming
(`tag-#anon#ST[...]`), which the contracts library uses heavily — this is why
`mul_contract.goto` is rejected (identically to the reference). Needs an LL(k) parser for
CBMC's `ST[...]`/`SYM`/`*{...}` type-name grammar (a skeleton exists in the original Rust
`adapter.rs::Anon2Struct`). Separately, the hex→binary constant rewrite goes through
`uint64_t`, so constants wider than 64 bits (e.g. 128-bit) are wrong.

### 4.4 Intrinsic & expression coverage (Phase 2)
`fix_expression` recognises a fixed set of ~40 expression ids. CBMC's surface is much
larger: pointer predicates (`__CPROVER_r_ok`/`w_ok`/`same_object`, `POINTER_OFFSET`,
`POINTER_OBJECT`), `__CPROVER_assume`/`assert`, array/quantifier predicates, IEEE-754
rounding-mode operations, `byte_update`, big-endian byte ops, etc. Unmapped ids pass
through unwrapped and may break downstream. Needs a systematic, tested mapping keyed off
the CBMC `irep_idt` vocabulary, extending the PR #2443 intrinsic approach.

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

### 4.8 Math-library intrinsics arrive as bodyless FUNCTION_CALLs (Phase 2) — 🔶 newly diagnosed, not yet fixed
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

**Also observed, not yet root-caused**: `malloc`/`free` also don't match CBMC's verdict on
a minimal alloc/write/free program (ESBMC: "Incorrect alignment when accessing data
object"; unlike the libm case, this is a different failure mode, not "no body for
function" — `malloc` does get a real body, since ESBMC's embedded `libclibs.a` provides
one independent of the C frontend / CBMC binary distinction). Whether this is the same
class of gap or an unrelated pre-existing bug is unconfirmed; flagging so it isn't lost.

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
