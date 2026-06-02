# Python Issues — Sweep & Triage Report (2026-06-02)

Companion to `docs/python-issues-triage-plan.md` (2026-05-31). This report records an
**independent, empirical re-test of every open Python-labeled issue against current
`master`** (ESBMC 8.3.0, commit `26958d9fd9` + PR #5040), classifying each by *actual*
present-day failure mode and disposition. Every verdict below was reproduced locally, not
inferred from the prior plan.

Guiding policy (unchanged): **an unsound or heuristic fix is worse than an honest
KNOWNBUG.** Timeouts must not be "fixed" by loosening unwinding (false `SUCCESSFUL` is
banned). Crashes are bugs, but converting a crash to a clean diagnostic does not flip a
KNOWNBUG verdict and is tracked separately as robustness work.

---

## 1. PRs opened this sweep

| PR | Issue(s) | Summary | Validation |
|----|-------|---------|------------|
| **#5040** | **#5036** | Segfault converting `return (g(a), g(a))` — a tuple literal whose elements are function calls. Root cause: `tuple_handler::get_tuple_expr` embedded a `code_function_callt` (a *statement*) directly as a `struct_exprt` operand, which then flowed into the GOTO `RETURN` and was dereferenced on a null operand. Fix materialises non-symbol elements into temporaries (DECL+ASSIGN), mirroring the proven list-literal fix for #4699. | New regression pair `github_5036{,_fail}`; dual-solver Z3+Bitwuzla agreement; CPython sanity; all 67 existing python tuple/unpack tests pass; CI style checks green. |
| **#5042** | **#4782, #4804, #4805** | Crash in `goto_symext::intrinsic_get_object_size` (`__ESBMC_get_object_size`): the bare `assert(is_array_type(...))` plus `internal_deref_items.front()` was UB (SIGSEGV in release) / assertion failure when the deref yielded an empty/non-array object — which the Python set/graph OMs route here via `len()`/membership. Fix replaces the assert with an explicit guard emitting a clean `log_error`+`abort` diagnostic; the array success path is byte-for-byte unchanged. | New tests `github_4782_object_size` (clean error) + `github_4782_object_size_ok` (array path still verifies); **all 50 object_size C regression tests pass before & after** (no C/C++ impact); CPython sanity; dual-solver/pre-solver determinism. |

These are the two **isolated, soundly-fixable** defects in the current open set. #5040 fixes a
genuine corner-case crash in a working feature; #5042 is a robustness hardening that converts a
crash into an honest diagnostic without changing any valid C/C++ behaviour. The remaining issues
do not share this profile (see §3).

### Robustness crashes deliberately NOT masked
- **#4796** (reverse_linked_list) aborts in the SMT backend `mk_eq` on
  `assert(get_data_width(a)==get_data_width(b))` when comparing a `Node` to `None`. This is a
  **generic, all-frontend solver invariant** — equality operands must share a bit-width — whose
  violation signals a frontend *codegen* bug (the Python object/Optional comparison emitting
  mismatched widths). Guarding it in `mk_eq` would suppress ESBMC's ability to catch the same
  class of codegen bug in C/C++/all languages, so the sound fix belongs in the Python
  object/Optional model (design-level, #4653/#3067), **not** the solver. Left as KNOWNBUG.

---

## 2. Method

For each KNOWNBUG test and standalone issue, ran `esbmc <src> <test.desc flags>` with a
25–30 s wall cap and classified the outcome: `SEGFAULT` / internal-`ABORT` / clean
`ERROR:` (unsupported feature) / `TIMEOUT` / wrong-verdict (`FAILED`/`UNKNOWN` vs expected).
Crashes were inspected for whether they bottom out in a deep design area or an isolated bug.
A scripted "does it now match the expected regex?" scan flagged four python/ tests as
possibly-fixed; **all four were manually disproven** (the scan over-matched) — a reminder
that scan output must be verified before trust.

---

## 3. Disposition of every open Python issue

### 3a. Fixed / shipped
- **#5036** — fixed by PR #5040 (above).

### 3b. Design-level blockers — no sound point fix exists
These bottom out in one of ~5 architecture gaps. Reproduced root cause in brackets.

| Issue(s) | Test(s) | Reproduced blocker | Underlying design item |
|---|---|---|---|
| #4771, #4772, #4773 | github_4117_attr_conflict / _attr_shadow / _function_internal | wrong verdict / `Cannot resolve nested attribute` | **Flow-sensitive class tracking** (replace lexical scan); umbrella #3067 |
| #4784 | detect_cycle | `ERROR: Cannot resolve nested attribute: successor` | same (#4117 family) |
| #4782 | depth_first_search | internal `ABORT: is_array_type(internal_deref_items...)` | object/deref model on class graphs |
| #4796 | reverse_linked_list | internal `ABORT: get_data_width()==...` in `mk_eq` (comparing `Node` vs `None`/Optional) | **Object-model / Optional rework** (#4653, #3067) |
| #4804, #4805 | topological_ordering(_fail) | `SEGFAULT` (class attrs + `set().issuperset` + `not in`) | class tracking + set ops |
| #4775 | rover | `ERROR: Variable 'Twist' is not defined` (typed `Dict[str,T]`, nested attr writes) | class/import resolution + #4117 |
| #4791 | minimum_spanning_tree_fail | `ERROR: Cannot unpack signedbv` | **Tuple-unpack type inference** |
| #4794, #4795 | powerset(_fail) | `ERROR: Cannot unpack pointer` | tuple-unpack inference |
| #4802, #4803 | shortest_paths(_fail) | `ERROR: Only simple targets are supported in DictComp` | dict-comp tuple targets + class |
| #4799, #4800 | shortest_path_length(_fail) | `ERROR: Cannot unpack empty` + `heappush/heappop` undefined | tuple-unpack + missing `heapq` OM |
| #4776 | shedskin | `ERROR: 'int' object is not subscriptable` (line 25 = `dict.items()` list-of-tuples equality) | dict/tuple-equality modeling |
| #4806 | wrap | wrong verdict (string slicing/equality) | **String/list-equality soundness** |
| #4797, #4798 | rpn_eval(_fail) | wrong verdict / arg-arity TypeError on `**`/dict dispatch | mixed (equality + builtins) |
| #4642, #4653 | — | arbitrary-precision int; OM rework to unblock bignum | **bignum + OM rework** |
| #4579, #4584, #4566 | concurrency_fail | `ERROR: Thread subclass must be constructed at module scope (MVP)` | **GIL-aware threading model** |
| #2848 | — | `typing.Any` at symex level | **`any`-type inference** (root of the str()/sorted() clusters below) |
| #3067 | — | refactor class handling into a dedicated class | enabler for #4117 family |
| #3541 | — | math dataset/benchmark suite | infrastructure task, not a bug |

**humaneval umbrella #4807** sub-tests reproduce as the same clusters:
- `any`-typing (#2848): `str(<any>)` → `TypeError: str() expects a string argument` (humaneval_108/145/151); `sorted()` mixed-type (123); "Unhandled symbol format in string extraction" (93/125).
- string/list-equality soundness: humaneval_1/1-1/33/37/62/67/78/86/95/103/137 report `FAILED` vs expected `SUCCESSFUL`.
- specific feature gaps: tuple slice non-constant lower (148); `split()` non-constant maxsplit (91); numpy `fmod` (2-1); dict→list retyping (126, = #4774 function-level).
- **infeasible to model soundly**: humaneval_162 `hashlib.md5` with an exact-digest assertion (would require a full MD5 implementation; symbolic verification of a crypto digest is out of scope).

### 3c. Perf / timeout — policy-banned from "machine fixing"
Forcing these to pass requires loosening unwinding (false `SUCCESSFUL`, banned) or
k-induction convergence work, not a point fix:
- #4780, #4781 breadth_first_search(_fail); #4786 flatten_fail; #4789 knapsack_fail;
  #4792/#4793 next_permutation(_fail); #4802 shortest_path_lengths; humaneval_39, 90, 158.

### 3d. Questionable expected-verdict (not a code bug)
- #4779 bitcount_fail: the buggy variant `n ^= n-1` is **non-terminating** for the test
  inputs; incremental-BMC correctly returns `UNKNOWN` (cannot falsify an unbounded
  non-terminating program). The test's expected `FAILED` is the doubtful part. Recommend
  re-tagging or bounding rather than a frontend change.

### 3e. Already addressed since the 2026-05-31 plan (no action)
- #4774 module-level int/float/str retyping: `regression/python/mixed-types` is now **CORE
  and passing** (fixed by #4988). Only the *function-level* retyping variant remains open;
  that is the #2848/#4117 cluster.

---

## 4. Validation performed
- Re-ran the full quixbugs (25) and humaneval (25) KNOWNBUG sets plus python/ standalone
  KNOWNBUGs individually with their own `test.desc` flags.
- For PR #5040: dual-solver agreement, CPython sanity (`check_python_tests.sh`), and the
  complete `*tuple*`/`*unpack*` python regression subset (67 tests) — all green; no
  regressions. Solidity `tuple_*` failures are the known macOS-only environmental issue
  (empty sol64 models), unrelated.

## 5. Remaining work (recommended, in priority order)
1. **Flow-sensitive class tracking** (#3067 → unblocks #4771-3, #4775, #4782, #4784, #4796,
   #4804/5). Largest single unlock.
2. **String/list/tuple-equality soundness** (unblocks the humaneval `FAILED` cluster, #4806,
   #4776). Soundness-critical — must be solver-validated, never heuristic.
3. **Tuple-unpack type inference** (#4791, #4794/5, #4799/4800).
4. **`any`-type inference at symex** (#2848 → str()/sorted() clusters).
5. **Robustness**: convert the internal aborts/segfaults (#4782, #4796, #4804/5) into clean
   "unsupported feature" diagnostics. Sound and independently shippable; does **not** flip
   the KNOWNBUG verdicts but removes crashes.
6. Stdlib OMs where tractable: `heapq` (#4799/4800). Skip `hashlib` (infeasible).

**Bottom line:** the 2026-05-31 plan's classification holds under re-test. Two isolated,
soundly-fixable defects were found and fixed this sweep: a tuple-return segfault (#5036 →
PR #5040) and a crash in the `__ESBMC_get_object_size` intrinsic (#4782/#4804/#4805 →
PR #5042, a crash→diagnostic robustness fix with zero C/C++ impact). Everything else open is
a design-level blocker, a policy-banned timeout, a questionable test expectation, or — like
#4796 — a crash whose only sound fix is the architectural work in §5 (masking it in shared
backend code would degrade ESBMC's bug-detection for all frontends). No further isolated,
sound PR is available without that architectural work.
