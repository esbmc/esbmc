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
| Operand-wrap for unary bit-builtins `popcount`/`bswap` (§4.4, Phase 2) | ✅ (PR #TBD) | `cbmc_adapter.cpp::fix_expression` |
| Width-aware constant rewrite: ≤64-bit wide constants no longer truncated to 32 bits (§4.3, Phase 3) | ✅ (PR #TBD) | `cbmc_adapter.cpp::hex_to_bin` |
| Expression rewrite for `ieee_float_notequal` → `notequal` (float `!=`; §4.4, Phase 2) | ✅ (PR #TBD) | `cbmc_adapter.cpp::fix_expression` |
| Builtin-call rewrite for integer `abs`/`labs`/`llabs`/`imaxabs` (+`__builtin_`) → `abs` expr (§4.8, Phase 2) | ✅ (PR #TBD) | `cbmc_adapter.cpp::fix_builtin_call` |
| Tag-cache keyed by symbol name so **function-local** struct/union tags resolve (§4.3, Phase 3) | ✅ (PR #5925) | `cbmc_adapter.cpp::cbmc_adapt` |
| Type rewrite for `c_bit_field` (bitfield members) → underlying bv narrowed to the bitfield width + `#bitfield`/`subtype` (§4.3, Phase 3) | ✅ (PR #5924) | `cbmc_adapter.cpp::fix_type` |
| Operand-wrap for unary bit-builtins `popcount`/`bswap` (§4.4, Phase 2) | ✅ (PR #5910) | `cbmc_adapter.cpp::fix_expression` |
| Width-aware constant rewrite: ≤64-bit wide constants no longer truncated to 32 bits (§4.3, Phase 3) | ✅ (PR #5916) | `cbmc_adapter.cpp::hex_to_bin` |
| Expression rewrite for `ieee_float_notequal` → `notequal` (float `!=`; §4.4, Phase 2) | ✅ (PR #5909) | `cbmc_adapter.cpp::fix_expression` |
| Builtin-call rewrite for integer `abs`/`labs`/`llabs`/`imaxabs` (+`__builtin_`) → `abs` expr (§4.8, Phase 2) | ✅ (PR #5912) | `cbmc_adapter.cpp::fix_builtin_call` |
| Expression rewrite for `count_leading_zeros`/`count_trailing_zeros` (`__builtin_clz`/`ctz`) → popcount-based bit-count formula (§4.4, Phase 2) | ✅ (PR #5923) | `cbmc_adapter.cpp::fix_expression` |
| Overflow predicates `overflow-+`/`overflow--`/`overflow-*` (+ `/`/`mod`/`shl`/`unary-`) wrapped (`__builtin_{add,sub,mul}_overflow_p`) (§4.4, Phase 2) | ✅ (PR #TBD) | `cbmc_adapter.cpp::fix_expression` |
| 128-bit float constant width: `long double`/`float128` hex value converted to a 128-bit binary string instead of mistaken for an already-binary 32-bit value (§4.3, Phase 3) | ✅ (PR #TBD) | `cbmc_adapter.cpp::hex_to_bin`, `fix_expression` |
| Enum type reference `c_enum_tag` → bare `c_enum` so migrate yields a signed int (§4.3, Phase 3) | ✅ (PR #TBD) | `cbmc_adapter.cpp::fix_type` |
| Quantifier predicates `forall`/`exists` (`__CPROVER_forall`/`__CPROVER_exists`) + `=>` implication wrapped; bound-var `tuple` unwrapped; goto_check skips quantifier bodies (§4.4, Phase 2) | ✅ (PR #TBD) | `cbmc_adapter.cpp::fix_expression`, `goto_check.cpp::check_rec` |
| Rotate expressions `rol`/`ror` (`__builtin_rotateleft`/`rotateright`) → `(x << d) \| (x >> (W − d))` with `d = n mod W` (§4.4, Phase 2) | ✅ (PR #TBD) | `cbmc_adapter.cpp::fix_expression` |
| Builtin-call retarget for `memcpy`/`memset`/`memmove` FUNCTION_CALLs → ESBMC's `c:@F@__ESBMC_*` memory intrinsics (CBMC's ARRAY_COPY/REPLACE/SET body is unexecutable in ESBMC symex); `__*_impl` byte-loop fallbacks linked via the additions (§4.8, Phase 2) | ✅ (PR #TBD) | `cbmc_adapter.cpp::fix_builtin_call`, `parseoptions/goto_program.cpp` |
| Builtin-call rewrite for `__builtin_nan`/`__builtin_nanf` FUNCTION_CALLs → `ieee_div(0.0, 0.0)` (quiet NaN, mirroring CBMC's own `floatbv_div(0,0,rm)` body); `nanl` left bodyless for parity (§4.8, Phase 2) | ✅ (PR #TBD) | `cbmc_adapter.cpp::fix_builtin_call` |
| Find-first-set builtin `find_first_set` (`__builtin_ffs`/`ffsl`/`ffsll`) → `(x==0)?0:popcount(~x&(x-1))+1` (§4.4, Phase 2) | ✅ (PR #TBD) | `cbmc_adapter.cpp::fix_expression` |
| Builtin-call rewrite for `__builtin_huge_val{,f,l}`/`__builtin_inf{,f,l}` FUNCTION_CALLs → +∞ floatbv constant (sign 0, exponent all ones, mantissa 0), width-generic incl. 128-bit long double (§4.8, Phase 2) | ✅ (PR #TBD) | `cbmc_adapter.cpp::fix_builtin_call` |
| Bit-reversal expression `bitreverse` (`__builtin_bitreverse{8,16,32,64}`) → SWAR reversal via `bitand`/`shl`/`lshr`/`bitor` (§4.4, Phase 2) | ✅ (PR #TBD) | `cbmc_adapter.cpp::fix_expression` |

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

**Function-local struct/union tags — ✅ fixed.** A `struct_tag`/`union_tag` reference
resolves to its definition by the type symbol's *name*, which CBMC scope-qualifies:
`tag-S` at file scope but `main::1::tag-S` for a struct declared inside a function body. The
adapter's tag cache was keyed `"tag-" + base_name`, which only matched the file-scope form,
so **any function-local struct or union** went unresolved and aborted with
`struct_tag/union_tag should have been resolved`. Fixed by keying the cache on the symbol
*name* (`type_cache[sym.name]`), which equals the reference identifier at every scope and is
byte-identical to the old key at file scope (`name == "tag-" + base_name` there). This also
correctly distinguishes same-named structs in different scopes (`struct S` with different
layouts in two functions get distinct `f::…::tag-S` / `main::…::tag-S` names). Verdict
parity with CBMC, dual-solver, across local struct/union, nested, pointer-to-local, array-of-
local, and two-scope-same-name cases (`cbmc_local_struct`, `cbmc_local_struct_fail`,
`cbmc_local_union`, `cbmc_local_struct_scopes`). (A function-local struct with a *bitfield*
member additionally needs the §4.3 `c_bit_field` fix to verify end-to-end.)

**Wide-constant truncation (≤64 bits) — ✅ fixed.** `fix_expression`'s constant rewrite
called `hex_to_bin32`, which — mirroring the Rust reference's `format!("{:032b}", …)` —
padded every converted constant to **32** bits regardless of the constant's own type width.
CBMC stores integer constant values as hex strings, so a 64-bit value like `-5000000000LL`
(hex `12A05F200`) was emitted as a ≤33-char binary string and then silently interpreted at
32 bits: `__CPROVER_assume(x == -5000000000LL); assert(x < 0)` returned a **false `FAILED`**
(x truncated to its unsigned low 32 bits, `+3589934592`) where CBMC says `SUCCESSFUL` — a
soundness divergence, worse than a crash. Fixed by making the helper width-aware
(`hex_to_bin(val, width)`, `width` read from the constant's own `signedbv`/`unsignedbv`
type) so the binary string matches the type width. Surgical: the `!= 32` guard that passes
through already-binary 32-char strings is unchanged, and 32-bit constants still pad to 32
(byte-identical to before); only wider types change. Because the byte-identical goto-transcoder
parity reference shares the same 32-bit bug, this fix **intentionally diverges** from that
reference on wide constants — CBMC-**verdict** parity (roadmap §6, the end-state oracle) is the
correctness signal here, and the full `goto-transcoder` verdict suite still passes. Tests
`cbmc_wide_const` (the soundness repro), `cbmc_wide_const_fail`, `cbmc_wide_uconst`, dual-solver.

**128-bit float constants (`long double`/`float128`) — ✅ fixed.** A second, subtler collision
with the same 32-bit heuristic. `fix_expression`'s constant rewrite skipped conversion whenever
the value string was **exactly 32 chars** (`if (val.size() != 32)`), on the assumption that a
32-char value is an already-binary 32-bit string. But CBMC stores *every* constant — integer and
floating-point — as **hex**, and a 128-bit `long double`/`_Float128` value is stored as exactly
**32 hex chars** (e.g. `1.5L` → `3FFF8000000000000000000000000000`, IEEE binary128, `width=128`,
`f=112`). That value hit the 32-char guard, was left as raw hex, and `migrate`'s `ieee_floatt`
misdecoded it: `long double d=1.5L; assert(d>1.0L)` read `d` as ≈0 and returned a **false
`FAILED`** where CBMC says `SUCCESSFUL` (native ESBMC handles the same program correctly — the
bug was purely on the `--binary` path). Fixed by keying the guard on the constant's **own type
width** (`val.size() != width`) rather than a hardcoded 32: for every standard width the hex
length is strictly shorter than the bit width (w=32 → 1–5 chars, w=64 → 1–16, w=128 float → 32),
so the width test never mistakes hex for binary, and the 32-char float128 value is now converted.
`hex_to_bin` was extended past its `uint64_t` ceiling to expand >64-bit hex per digit (4 bits
each) instead of returning it unchanged. Byte-identical for ≤64-bit constants (same `std::stoull`
path); the full `goto-transcoder` verdict suite still passes. Tests `cbmc_long_double` (pins the
decoded value via `d==1.5L`, the `d>1.0L` soundness repro, and `d*2==3.0L`) and
`cbmc_long_double_fail`, dual-solver.

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

**Bitfield type `c_bit_field` — ✅ fixed.** CBMC types a bitfield struct member as a
`c_bit_field` type — `{width: N; sub[0]: <underlying integer bv of width W>}` — which
`migrate_type` has no case for, so any struct with a bitfield member aborted with
`ERROR: c_bit_field`. ESBMC has no distinct bitfield type node: its native C frontend
(`clang_c_convert.cpp::get_bitfield_type`) represents `unsigned a:N` as the *underlying* bv
kind narrowed to width `N`, tagged `#bitfield`, carrying the full underlying type as its
`subtype`; `migrate_type` then yields an `N`-bit bv. `fix_type` now rewrites `c_bit_field`
into exactly that shape. The one subtlety is `_Bool a:1`: `migrate` reads a `#bitfield`
**bool** as an *unsigned* `N`-bit value (`get_uint_type`), but `fix_type` otherwise maps
CBMC's `c_bool` to `signedbv` — and a 1-bit *signed* bv reads value `1` back as `-1`, a
verdict divergence. The rewrite detects a bool underlying before that mapping and keeps the
result `bool`. Verdict parity with CBMC, dual-solver, across unsigned/signed fields,
width-truncation (`s.a = 9` in a 3-bit field ⇒ `1`), signed wrap (`int a:3`, `5 ⇒ -3`),
`_Bool:1`, and multi-field packing (`cbmc_bitfield`, `cbmc_bitfield_signed`,
`cbmc_bitfield_bool`, `cbmc_bitfield_fail`). Bitfield members of a struct **defined inside a
function body** additionally trip the pre-existing `struct_tag/union_tag should have been
resolved` gap (§4.3 anon/tag resolution) — orthogonal to this fix and still open; the tests
use file-scope struct definitions.

**Enum type `c_enum_tag` — ✅ fixed.** CBMC references an enum type via a `c_enum_tag` node —
the tag counterpart of `c_enum`, exactly as `struct_tag`/`union_tag` reference `struct`/`union`.
`migrate_type` maps `c_enum`/`incomplete_c_enum` to a signed int (C99 6.7.2.2.3, "the type of
an enumeration is int") but has **no case for the tag**, so *any* enum-typed object — an enum
variable, an enum struct member, an enum function parameter — aborted with `ERROR: c_enum_tag`.
Unlike `struct_tag`/`union_tag`, resolving the tag through the adapter's cache is neither
possible (the cache holds only `struct`/`union` definitions, `cbmc_adapt`) nor necessary:
`migrate_type` collapses every `c_enum` to `signedbv[int_width]` regardless of the enum's
declared underlying width, so `fix_type` rewrites `c_enum_tag` to a bare `c_enum` and lets
migrate produce the identical int type. Contained to the CBMC-binary path (`fix_type` is never
reached by native frontends). Verdict parity with CBMC, dual-solver (Bitwuzla + Z3), across an
enum variable/comparison, enum arithmetic, explicitly-valued flag enums (`F_A | F_C`), and an
enum struct member (`cbmc_enum`), plus a negative case (`cbmc_enum_fail`: `BLUE == GREEN`
FAILED, confirming the value is really compared, not a vacuous pass). A C23 enum with a *fixed*
non-`int` underlying type would still collapse to `int` here — consistent with `migrate_type`'s
own `c_enum` handling, so not a new divergence — and is a migrate-level concern if it ever
matters.

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

**Unary bit-builtins `popcount`/`bswap` — ✅ landed.** `__builtin_popcount` and
`__builtin_bswap32` lower to CBMC `popcount`/`bswap` ireps, both of which `migrate_expr`
already handles via `op0()` — but neither was in `fix_expression`'s operand-wrap set, so
CBMC's raw operands stayed in `get_sub()`, `op0()` read an empty operand list, and the
verdict was garbage/crash (the exact `isnan`/`pointer_offset` failure shape). Fixed by adding
`popcount`/`bswap` to the wrap-set. Verdict parity both directions, dual-solver
(`cbmc_popcount`/`_fail`, `cbmc_bswap`/`_fail`).

**Bit-count builtins `clz`/`ctz` — ✅ landed.** `__builtin_clz`/`__builtin_ctz` lower to CBMC
`count_leading_zeros`/`count_trailing_zeros` ireps, which `migrate_expr` has *no* handler for
at all (aborts with `migrate expr failed`) — and, unlike `popcount`/`bswap`, ESBMC has no
`clz`/`ctz` irep2 node either: the native frontend resolves `__builtin_clz` at *symex* time
(`run_builtin.cpp`) with a popcount-based bit-count formula, and does not model `__builtin_ctz`
at all. Rather than add a new irep2 node, `fix_expression` reproduces that same formula in terms
of ids `migrate_expr` already lowers — `clz(x) = width − popcount(x smeared down below its MSB)`
(mirroring `run_builtin.cpp` exactly) and `ctz(x) = popcount(~x & (x−1))` — so no new node is
needed and the CBMC path gains `ctz` coverage the native path still lacks. Scoped to the
`--binary` path, so it cannot perturb native handling (which never emits
`count_{leading,trailing}_zeros` as an expression). `clz(0)`/`ctz(0)` is UB; CBMC emits its own
`#bounds_check` zero-argument guard, matched independently. Verdict parity both directions,
dual-solver, across 32-/64-bit widths and a symbolic (no-`assume`) case
(`cbmc_clz`/`_fail`, `cbmc_ctz`/`_fail`).

**Rotate expressions `rol`/`ror` — ✅ landed.** `__builtin_rotateleft{8,16,32,64}` /
`__builtin_rotateright...` lower to CBMC `rol`/`ror` ireps (value, distance), which `migrate_expr`
has *no* handler for (aborts with `migrate expr failed`) — and, like `clz`/`ctz`, ESBMC has no
rotate irep2 node. `fix_expression` reproduces the rotate from ids `migrate_expr` already lowers
(`shl`, `lshr`, `bitor`, `bitand`, `-`): `rol(x, n) = (x << d) | (x >> (W − d))` and
`ror(x, n) = (x >> d) | (x << (W − d))`, where `d = n mod W`. CBMC takes the distance **mod the
width** (`rol(x, W) == x`, `rol(x, W + k) == rol(x, k)`); the width is always a power of two, so
`& (W − 1)` is the modulus. The complement `W − d` is likewise masked with `W − 1` so `d == 0`
produces a 0-bit shift rather than an (edge-case) full-width shift — `rol(x, 0)` then reduces to
`(x << 0) | (x >> 0) == x`. Scoped to the `--binary` path, so it never perturbs native handling
(which never emits `rol`/`ror`). Verdict parity with CBMC, dual-solver (Bitwuzla + Z3), across
`rol`/`ror` at 32- and 64-bit widths and the mod-width edge (`rol(x, 32) == x`,
`rol(x, 36) == rol(x, 4)`): `cbmc_rotate` (SUCCESSFUL) and `cbmc_rotate_fail` (a wrong expected
value → FAILED, confirming the rotation is really computed).

**Find-first-set builtin `find_first_set` — ✅ landed.** `__builtin_ffs`/`ffsl`/`ffsll`
lower to a CBMC `find_first_set` irep — the 1-based index of the least-significant set bit,
or 0 when the argument is zero — which `migrate_expr` has *no* handler for (aborts with
`migrate expr failed`), and, like `clz`/`ctz`, ESBMC has no `ffs` irep2 node and its native
path does not model `__builtin_ffs` at all. Rather than add a node, `fix_expression`
reproduces it from ids `migrate_expr` already lowers: `ffs(x) = (x == 0) ? 0 :
popcount(~x & (x − 1)) + 1`, where the `popcount(~x & (x − 1))` term is exactly the `ctz(x)`
formula reused from the `count_trailing_zeros` rewrite above. The `x == 0` guard is
load-bearing — `~0 & (0 − 1)` is all-ones, whose popcount is the type *width*, not 0 — so
without it `ffs(0)` would wrongly be `width + 1`. Scoped to the `--binary` path, so it cannot
perturb native handling (which never emits `find_first_set`). Verdict parity with CBMC,
dual-solver (Bitwuzla + Z3), across a 32-bit value, the zero-input guard, and a 64-bit
`ffsll` operand (`cbmc_ffs` SUCCESSFUL), plus an off-by-one negative that confirms the value
is really computed, not vacuously passed (`cbmc_ffs_fail`: `ffs(0x100) == 8` FAILED where the
true answer is 9).

**Bit-reversal expression `bitreverse` — ✅ landed.** `__builtin_bitreverse{8,16,32,64}` lower to a
CBMC `bitreverse` irep (reverse the bit order: bit `i` ↔ bit `W−1−i`), which `migrate_expr` has *no*
handler for (aborts with `migrate expr failed`) — and, like `clz`/`ctz`, ESBMC has no `bitreverse`
irep2 node. `fix_expression` reproduces it from ids `migrate_expr` already lowers (`bitand`, `shl`,
`lshr`, `bitor`) via the standard SWAR reversal — swap adjacent bits, then 2-bit groups, then 4-bit,
… doubling the group size each step: `acc = ((acc & mask_k) << k) | ((acc >> k) & mask_k)`, where
`mask_k` selects the low `k` bits of every `2k`-bit block (`0x5555…`, `0x3333…`, `0x0F0F…`, …). The
loop is width-generic (⌈log₂W⌉ steps for W = 8/16/32/64) and self-referential like the `clz` smear.
Scoped to the `--binary` path, so it never perturbs native handling (which never emits `bitreverse`).
Verdict parity with CBMC, dual-solver (Bitwuzla + Z3): `cbmc_bitreverse` (32-bit alternating-bit
swap `0xAAAAAAAA → 0x55555555`, low-nibble-to-high `0x0F → 0xF0000000`, and a 64-bit MSB case) and
`cbmc_bitreverse_fail` (a wrong expected value → FAILED, confirming the reversal is really computed).

**Overflow predicates `overflow-<op>` — ✅ landed.** `__builtin_add_overflow_p`/
`__builtin_sub_overflow_p`/`__builtin_mul_overflow_p` lower to CBMC's bool-typed `overflow-+`/
`overflow--`/`overflow-*` predicate ireps (distinct from `overflow_result-<op>`, which returns
the value+flag pair and was already handled). `migrate_expr` fully supports the whole
`overflow-<op>` family (`convert_operand_pair` → `overflow2tc`), but none were in
`fix_expression`'s operand-wrap set, so their operands stayed in `get_sub()`, `convert_operand_pair`
read an empty operand list, and the verdict **segfaulted** (the exact `isnan`/`popcount` failure
shape). Fixed by adding the family to the wrap-set: `+`/`-`/`*` are exercised by the builtins,
and `/`/`mod`/`shl`/`unary-` share the identical single-mechanism wrap requirement (added for
completeness). Verdict parity with CBMC, dual-solver (Bitwuzla + Z3), across genuine
add/sub/mul overflow at `INT_MAX`/`INT_MIN` and a no-overflow negative (`cbmc_overflow_p`), plus
a false-overflow claim (`cbmc_overflow_p_fail`: `2 + 3` reported as overflowing ⇒ FAILED).

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

**Quantifier predicates `forall`/`exists` — ✅ landed.** `__CPROVER_forall`/`__CPROVER_exists`
lower to CBMC `forall`/`exists` ireps, which `migrate_expr` already handles via `op0()`/`op1()`
(bound symbol, predicate) — but neither was in the operand-wrap set, so `op0()` read an empty
operand list and **segfaulted** (the exact `isnan`/`popcount` failure shape). Three coupled
fixes were needed for verdict parity, not just crash-avoidance:
- **Wrap `forall`/`exists`** so their operands reach `migrate_expr`.
- **Unwrap the bound-variable `tuple`.** CBMC binds the quantifier variable(s) inside a `tuple`
  node in the first operand; ESBMC's `forall2t`/`exists2t` (and `smt_solver.cpp`) expect
  `side_1` to be the bound *symbol* itself, so `migrate` aborted with `ERROR: tuple`. The
  adapter unwraps a single-symbol tuple to the bare symbol; a multi-binder tuple is left
  untouched (ESBMC binds exactly one symbol) so it aborts cleanly rather than silently dropping
  binders.
- **Wrap `=>` implication.** Quantifier guards use `guard ==> body`; `migrate_expr` lowers `=>`
  to `implies2t` via a wrapped operand pair, but `=>` was missing from the wrap-set, so a
  guarded `forall` segfaulted after the first two fixes. (Valid in any boolean context, not just
  quantifiers.)
- **`goto_check` skips quantifier bodies.** A universally/existentially bound variable ranges
  over its whole type; its body is a pure predicate, not executed code, so `goto_check`'s bounds/
  overflow/div-by-zero instrumentation over the bound variable is meaningless — `a[i]` inside
  `forall i . (0<=i<n) ==> a[i]==0` is **not** a real out-of-bounds access. Previously
  `check_rec` descended into the body with an empty guard and emitted a spurious bounds check
  (false **FAILED** where CBMC says SUCCESSFUL — a soundness divergence, worse than a crash).
  A new `forall_id`/`exists_id` case returns without recursing, matching CBMC (which emits no
  such checks inside a quantifier) and letting the SMT array theory model the body. This also
  fixes the identical pre-existing gap on ESBMC's own native `__ESBMC_forall` (a bare
  `forall(i, a[i]==0)` spuriously bounds-failed before this change); the existing native
  quantifier suites, which dodge it via a `(guard)?body:1` ternary, are unaffected.
Verdict parity with CBMC, dual-solver (Bitwuzla + Z3), across a holding `forall`
(`cbmc_forall` SUCCESSFUL), a violated `forall` (`cbmc_forall_fail` FAILED — confirms the body
is really evaluated, not vacuously skipped), and an `exists` witness (`cbmc_exists` SUCCESSFUL).

Still open: `__CPROVER_assume`/`assert` (only relevant if they surface as expressions
rather than instruction-level ASSUME/ASSERT, unconfirmed),
IEEE-754 rounding-mode operations, `byte_update`, big-endian byte ops (`byte_extract_big_endian`,
`byte_update_little_endian`/`_big_endian` are absent from the wrap-set and have `migrate_expr`
support, but goto-cc/goto-instrument never persist them into a `.goto` — they are introduced by
CBMC's own symex flattening, so a Kani-derived binary is needed to reproduce and test). Needs a
systematic audit of the CBMC `irep_idt` vocabulary against the adapter's wrap-set, not just
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

**Integer abs family `abs`/`labs`/`llabs`/`imaxabs` (+`__builtin_` spellings) — ✅ landed.**
The integer counterpart of `fabsf`: CBMC emits these as bodyless `FUNCTION_CALL` externals
too, so ESBMC returned nondet and a valid `abs(-7)==7` reported `FAILED` where CBMC says
`SUCCESSFUL`. The native `abs` expr is type-agnostic (`build_unary_fp_rhs` takes the lhs
type), so the same rewrite the float family uses covers integer abs unchanged — just extend
the callee match. Tests `cbmc_abs`/`_fail` (int) and `cbmc_llabs` (64-bit-typed), dual-solver.
While building these, discovered a **pre-existing, abs-independent 64-bit-constant truncation
bug**: `__CPROVER_assume(x == -5000000000LL); assert(x < 0)` returns `FAILED` under ESBMC vs
`SUCCESSFUL` under CBMC — the 64-bit constant is truncated to its low 32 bits and
zero-extended (`fix_expression`'s constant rewrite / `hex_to_bin32` is 32-bit-only; roadmap
§4.3, §7). The abs tests deliberately use values inside 2^31 to avoid conflating the two;
the constant bug is tracked as a separate follow-up.

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

**`memcpy` / `memset` / `memmove` — ✅ landed.** Distinct from the value-returning builtins above:
CBMC inlines a `<builtin-library-memcpy>` (etc.) body that performs the copy via `ARRAY_COPY` /
`ARRAY_REPLACE` (memset: `ARRAY_SET`) — CBMC `OTHER` instructions carrying `array_copy` /
`array_replace` / `array_set` codet statements that **ESBMC's symex has no handler for and silently
skips**, so the copy never happens and any read of the destination after the call reports a false
`FAILED` (`char d[4]; memcpy(d, "abc", 4); assert(d[0]=='a')` failed where CBMC says `SUCCESSFUL`).
Rather than teach symex those array ops, `fix_builtin_call` **retargets** the call to ESBMC's own
well-tested memory intrinsic — `memcpy` → `c:@F@__ESBMC_memcpy`, `memset` → `c:@F@__ESBMC_memset`,
`memmove` → `c:@F@__ESBMC_memmove` — by rewriting only the callee symbol's identifier: symex
dispatches any `c:@F@__ESBMC*` call to `run_intrinsic` purely by name (`symex_main.cpp`), the
3-argument signature already matches, and a nil lhs (discarded return) is fine, so the instruction
stays a `FUNCTION_CALL` (the adapter returns `false`, leaving CBMC's original instruction type).
`intrinsic_memcpy`/`memmove`/`memset` compute a constant-size copy directly but **bump to the
`__memcpy_impl`/`__memmove_impl`/`__memset_impl` byte-loop bodies** when the size or pointers are
symbolic (and, for memmove, when the regions overlap); those bodies live in ESBMC's `string.c`
operational model and are **not** in a CBMC binary, so the additions boilerplate
(`synthesize_cprover_additions`) now also references `memcpy`/`memmove`/`memset`, force-linking
them. Verdict parity with CBMC, dual-solver (Bitwuzla + Z3): `cbmc_memcpy` (full 4-byte copy),
`cbmc_memset` (fill), `cbmc_memmove` (overlapping `memmove(a+1, a, 4)`), all `SUCCESSFUL`, and
`cbmc_memcpy_fail` (`dst[0] == 'z'` after copying `'a'`) `FAILED`. A symbolic-size copy verifies to
CBMC's verdict too, but takes the byte-loop path and so needs an `--unwind` bound like any other
symbolic-length loop — unchanged from ESBMC's native `memcpy` semantics.
**`__builtin_nan` / `__builtin_nanf` — ✅ landed.** CBMC provides a bodied
`<builtin-library-__builtin_nan>` model that returns `floatbv_div(0, 0, __CPROVER_rounding_mode)`
(i.e. `0.0/0.0`, a quiet NaN), but that body does not survive the reader/adapter (its
`floatbv_div` flattened-IEEE node has no `migrate_expr` handler), so the function reaches symex
as a **bodyless external returning nondet** — and `double d = __builtin_nan(""); assert(d != d)`
reported a false `FAILED` (nondet `d` may equal itself) where CBMC says `SUCCESSFUL`. `fix_builtin_call`
now rewrites the call into `lhs = ieee_div(0.0, 0.0)` — mirroring CBMC's own body exactly, using an
id already in the operand-wrap set that migrates and defaults its rounding mode like the rest of the
`ieee_*` family. The NaN-payload string argument is ignored (it does not affect NaN-ness, and ESBMC's
own C frontend likewise folds `__builtin_nan` to a constant NaN without dereferencing it).
Deliberately restricted to `double`/`float`: **CBMC 6.5.0 does not model `__builtin_nanl` as a NaN**
(its result compares equal to itself, so `x != x` is `FALSE` — CBMC itself reports `FAILED`), so
`nanl` is left as a bodyless external, whose nondet return already yields the same `FAILED` verdict,
preserving parity rather than manufacturing a divergence. Verdict parity with CBMC, dual-solver
(Bitwuzla + Z3): `cbmc_nan` (`d != d` and `f != f` both SUCCESSFUL) and `cbmc_nan_fail`
(`d == 0.0` FAILED — a NaN equals nothing, confirming a real NaN, not just an unconstrained value).

**`__builtin_huge_val{,f,l}` / `__builtin_inf{,f,l}` — ✅ landed.** CBMC provides bodied
`<builtin-library-*>` models for these positive-infinity builtins, but the bodies do not survive
the reader/adapter (their flattened `floatbv` nodes have no `migrate_expr` handler), so each reaches
symex as a **bodyless external returning nondet** — and `double d = __builtin_huge_val(); assert(d > 1e30)`
reported a false `FAILED` where CBMC says `SUCCESSFUL`. `fix_builtin_call` now rewrites the call into a
direct +∞ `floatbv` **constant** — sign 0, exponent all ones, mantissa 0 — emitted as the full-width
binary bit pattern (`fix_expression`'s constant branch leaves an already-width-length string
unchanged), so it is correct for **every** width including the 128-bit `long double` (binary128),
unlike a 64-bit literal. Unlike `nanl` (§4.4, which CBMC does *not* model as a NaN), CBMC models the
long-double `huge_vall`/`infl` as genuine +∞, so those are handled too. `__builtin_inf` (double, no
suffix) is folded to a constant by CBMC and never reaches the adapter as a call — it is matched for
uniformity and is harmless. Verdict parity with CBMC, dual-solver (Bitwuzla + Z3): `cbmc_inf` pins the
result to +∞ precisely via `x > 0 && x == x + 1` (the only finite-rejecting, sign-checking identity)
across `double`/`float`/`long double`, and `cbmc_inf_fail` (`d == 0.0` ⇒ FAILED).

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

**Status**: 🟢 CRASH CHAIN RESOLVED + CBMC-verdict parity on the minimal reproducer (`check_typed_swap_u8`); other candidates pending
**Impact**: ~200+ tests
**Severity**: 🔴 CRITICAL
**Exit Codes**: rc=139 (both parse & verify) — **no longer reproduces** on `check_typed_swap_u8`

**Description**:
ESBMC crashes with segmentation fault when processing `transmute` operations and type-casting intrinsics. Both safe `transmute` and unsafe `transmute_unchecked` variants fail during both parse and verify phases. The lowering layer cannot properly represent bit-level type reinterpretation in the C intermediate representation.

**Root cause & fix (minimal reproducer `check_typed_swap_u8`) — ✅ crash chain resolved.**
The single rc=139 was not one bug but a chain of four, each masking the next; fixing all
four takes the reproducer from an immediate stack-overflow segfault to a clean end-to-end
verification verdict. Discovered and fixed by direct testing against the real Kani
CBMC binary (`--binary --function …check_typed_swap_u8`), dual-solver (Bitwuzla + Z3):

1. **Recursive-aggregate stack overflow → segfault** (`cbmc_adapter.cpp::fix_type`). Rust
   `core` aggregates are pervasively self-referential (a struct holding a pointer back to
   its own tag). `fix_type` inlined a tag's definition out of the struct-tag cache
   unconditionally, so a self-reference recursed forever and blew the stack — the literal
   rc=139. Fixed by threading an `expanding` set of the tag identifiers currently on the
   recursion stack; a tag already being inlined is left as a `symbol` type instead, which
   `migrate_type` resolves lazily via the symbol table — the same way ESBMC's own frontends
   encode recursive aggregates. A fresh stack is seeded per top-level type.
2. **`union` missing from the operand-wrap set → empty `constant_union2t` abort**
   (`cbmc_adapter.cpp::fix_expression`, §4.4 shape). Exactly the `pointer_offset`/`popcount`
   failure mode: a `union` constant carries its single initialiser as a raw operand, but
   `union` was absent from `fix_expression`'s wrap-set, so the operand never moved into the
   `"operands"` named-sub `migrate_expr`'s `forall_operands` reads. Migrate then built a
   `constant_union2t` with **no** `datatype_members`, which every consumer that indexes
   `datatype_members[0]` (value_set, dereference, symex_assign) aborts on. Rust's
   `MaybeUninit` makes this ubiquitous — its active variant's value is a nondet sideeffect
   that must survive migration, so the union stays soundly nondeterministic. Fixed by adding
   `"union"` to the wrap-set (symmetric to the pre-existing `"struct"`).
3. **Zero-sized union member → 65-bit-vs-64-bit sort mismatch** (`smt_solver.cpp`, union
   `with` encoding). A Rust enum payload union `{ Ok:(i64)=64b, Err:(())=0b }` written at
   its zero-sized `Err(())` variant took `mem_bits = 0`, and the encoding built a degenerate
   `get_uint_type(0)` bitvector; the solver widened the 0-width operand to 1 bit, so the
   `concat` produced a value **one bit wider** than the union sort (Z3/Bitwuzla both reject
   it). Fixed by recognising that a zero-sized member occupies no storage — writing it leaves
   the union's bit representation unchanged — and short-circuiting to the source value.
4. **Wide `byte_extract` → pointer/non-bv target yielded an 8-bit bitvector**
   (`smt_byteops.cpp::convert_byte_extract_bv_mode`). This is the **bit-level type
   reinterpretation** at the heart of the umbrella. ESBMC's own `byte_extract2t` is a
   byte-granular primitive — every internal construction uses an 8-bit type and the SMT
   conversion always extracted exactly one byte, ignoring the result type — but
   `migrate.cpp` maps CBMC's `byte_extract_little_endian` 1:1 with its (possibly wide)
   result type. Rust's `without_provenance_mut::<()>(0)` reinterprets integer `0` as a
   `*mut ()`, i.e. `byte_extract(0_u64, offset 0, *mut ())`, which produced an 8-bit
   bitvector assigned to a pointer-struct LHS. Fixed by generalising the extraction to
   `data.type`'s width and reinterpreting the extracted bits into the target sort via a
   `bitcast` when the target isn't a plain bitvector (pointer/float/aggregate; the
   int→pointer bitcast falls through to ESBMC's pointer typecast, giving a null pointer for
   `0`). The 8-bit path is byte-identical (`out_width == 8` collapses to the original
   single-byte extraction), so native frontends are unaffected.

**Verified — CBMC-verdict parity (roadmap §6 oracle).** On the same binary and function,
**CBMC 6.10.0 also reports `VERIFICATION FAILED`** (11 of 3036 checks), so ESBMC matches
CBMC's top-level verdict. Moreover the failures agree at the *property* level: all 11 of
CBMC's failures are Kani `reachability_check` coverage markers (`KANI_CHECK_ID_…`, which are
designed to "fail" when the code is reachable — not real assertion violations; the harness's
substantive checks all pass in both tools), and ESBMC's first-found failure —
`num::<impl usize>::abs_diff`, `KANI_CHECK_ID_core…::core_11`, line 1278 — is an **exact
match** to one of CBMC's 11 (ESBMC stops at the first failing property by default, CBMC
enumerates all). Separately: all 421 ESBMC unit tests pass; end-to-end C sanity across the
four reinterpret shapes (8-bit memcpy, union type-pun, float bit-reinterpret, int→pointer)
verifies SUCCESSFUL on both solvers, with a negative variant (wrong float bit-pattern)
correctly FAILED to prove the values are genuinely constrained.

**Still open (this umbrella):** (a) the other candidates below (explicit `transmute`/
`transmute_unchecked`, `transmute_ptr_address`, arr↔tuple) are not yet exercised — they may
share this crash chain or surface new reinterpret shapes; (b) ESBMC stops at the first
failing property — a `--multi-property` run to confirm it reproduces all 11 of CBMC's
reachability markers (not just the first) would tighten parity from verdict-level to
full property-set; (c) CBMC-binary regression fixtures (`cbmc_transmute*`) are not yet
added — the fixes were validated against the live Kani binary and native C reproducers,
since this build has no regression suite configured.

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
