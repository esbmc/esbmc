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

---

## 6. 2026-06-10 re-validation & fix

Independent re-run of **all 47 KNOWNBUG python/quixbugs/humaneval/python-intensive tests**
against `master` as of commit `9d13245e4e` (ESBMC 8.3.0, Z3 + Bitwuzla), with each test's
own `test.desc` flags. **None now produce their expected verdict** — no zero-risk
KNOWNBUG→CORE flip is available; the design-level classifications in §3 still hold after the
8 days of fixes that closed the *separately-tracked* issues (#5085/#5096/#5110/#5114 string
soundness, #5102–#5129 matmul/FP, #5015 flow-sensitive class tracking, #5274 None-handle
width, …).

### 6a. New isolated, soundly-fixable defect found & fixed
**SMT-backend abort on a short-circuited `is None` before an attribute dereference.**

A guard such as `a is None or a.b is None` (QuixBugs `detect_cycle`, `depth_first_search`,
`topological_ordering`; and `reverse_linked_list`) dereferences `a.b` under the short-circuit
guard `not isnone(a)`. The short-circuit lowers to a GOTO branch
`IF !(ISNONE(cur, 0) || ISNONE(cur->nxt, 0))`, so the attribute deref lives **inside the GOTO
guard**. In the GOTO case of `symex_step` (`symex_main.cpp`), `dereference()` ran **before**
`simplify_python_builtins()`, so the NULL-pointer dereference-safety assertion was recorded —
guarded by the short-circuit `not isnone(cur)` — with the still-live `isnone` in its
condition. That `isnone` then reached `smt_convt::convert_ast`, which has no rule for it, and
`abort()`ed (`ERROR: Couldn't convert expression in unrecognised format`; SIGABRT / core
dump).

**Fix:** in the GOTO case, lower Python predicates *before* dereferencing the guard.
`simplify_python_builtins` is a pure no-op on non-Python expressions (no
`isnone`/`isinstance`/`hasattr` exist in C/C++ IR), and `dereference` never introduces an
`isnone`, so the reorder is strictly safe and has **zero effect on non-Python frontends**.
New regression pair `github_4784_isnone_short_circuit{,_fail}`.

The other two symex sites that interleave `dereference()` with `simplify_python_builtins()` —
the ASSERT case (`symex_main.cpp`, via `claim`) and assignment RHS (`symex_assign.cpp`) — host
the *same* `ISNONE(p, 0) || ISNONE(p->attr, 0)` pattern when the short-circuit appears in an
`assert`/assignment over a pointer owner. They were re-tested with the unchanged ordering and
do **not** hit the SMT-convert abort (they reach `Converting` and yield a real verdict), so the
fix is confined to the GOTO site that actually leaks the predicate — no speculative reorder is
added where it changes nothing.

This **removes the crash** but, like #5042/#4796, **does not flip the KNOWNBUG verdicts**: the
affected tests still hit the *pre-existing* unsound `isnone(<struct-ptr>, None) → false`
lowering ("a `Node` pointer is never `None`"), which now surfaces as a spurious NULL-deref
(`VERIFICATION FAILED`) instead of an abort. Closing those tests still needs the §5 object /
Optional-model rework (#4653/#3067). This is the §5-item-5 robustness work, shipped.

### 6b. Everything else: unchanged disposition
Re-confirmed failure modes (today's master): perf/timeout (breadth_first_search, knapsack,
flatten_fail, reverse_linked_list, shortest_path_lengths, humaneval_39/90/93/158 — §3c,
policy-banned); clean "unsupported feature" errors (list-slice-assign for next_permutation /
humaneval_33; tuple-unpack for minimum_spanning_tree / powerset / shortest_path_length;
DictComp tuple targets for shortest_paths; mixed-type `sorted()` / non-const tuple slice for
humaneval — §3b feature gaps); and wrong-verdict soundness clusters (§3b). Each remains a
design-level blocker or a substantial sound feature, not an isolated point fix.

---

## 7. 2026-06-11 re-validation & reversed() fix

Independent re-run of **all 47 KNOWNBUG python/quixbugs/humaneval tests** against `master`
(commit `71d0d97983`, ESBMC 8.3.0, Bitwuzla), each with its own `test.desc` flags, after the
merges that landed since §6 (`#5302` list slice-assign, `#5307` tuple-as-shallow-copy, `#5268`
k-induction phase 2). **Zero KNOWNBUG→CORE flips** — the §3 classification still holds. Every
crash/error reproduced is already documented: the `__ESBMC_get_object_size` diagnostic
(`depth_first_search`, by design since #5042), `rover`'s `Variable 'Twist' is not defined`
(#4775 class/import resolution), and `wrap` (its `--z3` pin is a local Bitwuzla-only artefact;
under Bitwuzla it is the §3b wrong-verdict, not a crash).

### 7a. New isolated, soundly-fixable defect found & fixed
**`reversed()` was unmodelled, blocking list-slice-assignment RHS.** `#5302` advanced
QuixBugs `next_permutation` from a slice-assign feature gap to a fresh narrow error,
`List slice assignment requires a list right-hand side`, on
`next_perm[i+1:] = reversed(next_perm[i+1:])`: `reversed()` had no operational model and its
`builtin_functions()` entry mapped to the invalid type tag `"reversed"` (vs `sorted` →
`"list"`), so even a bare `r = reversed(xs)` assignment raised
`NameError: name 'reversed' is not defined` during the annotation pass.

**Fix** (commit `8504e4f64d`): add `reversed`/`reversed_float`/`reversed_str` operational
models returning a freshly reversed list (mirroring `sorted`); wire `reversed` into the
element-type monomorph dispatch in `function_call/expr.cpp`; correct the `builtin_functions()`
tag to `"list"` and drop `reversed` from the `iterable_builtins` self-mapping set. Modelling
`reversed()` as an eager list is sound in every context ESBMC routes through the model (slice
assignment, `list(reversed(...))`); `for`/`range` iteration is rewritten earlier in the
preprocessor and is unaffected (existing `reversed1`/`reversed_loop` unchanged). New regression
pair `regression/python/reversed_builtin{,_fail}`; CPython sanity + 53 sibling
reversed/sorted/slice/list tests green; code-reviewed (0 critical/major).

Like #5042/#4796/§6a, this **does not flip a KNOWNBUG verdict**: with `reversed()` modelled,
`next_permutation` now does genuine BMC and hits the unwinding wall (policy-banned timeout,
§3c) instead of erroring — an honest perf bound, not a feature gap. The fix is independently
useful for any program using `reversed()` in a value context.

### 7b. Everything else: unchanged disposition
No further isolated, sound point fix is available on current master without the §5
architectural work (flow-sensitive class tracking, tuple-unpack inference, any-typing,
string/tuple-equality soundness). The §5 priority order stands.

---

## 8. 2026-06-16 re-validation — signatures shifted, no new isolated fix

Independent re-run of **all 46 KNOWNBUG python/quixbugs/humaneval tests** against current
`master` (tip `bd2e4a7d79`), each with its own `test.desc` flags, after the **63 commits**
that landed since §7. That set includes the largest Python push of any inter-sweep window —
`#5329` (model a class used as a first-class value; flip `rover` to CORE), `#5324`/`#5336`
(tuple targets / `d.items()` in dict comprehensions), `#5333` (`@property` getters),
`#5338`/`#5347`/`#5353`/`#5356` (lexicographic tuple ordering, sort literal/symbolic lists of
tuples, resolve `list[Tuple[...]]` element type), `#5306` (`issuperset` lowering), `#5319`
(forward-ref class returns), `#5331`/`#5332` (while-else, complex inference).

**Result: zero KNOWNBUG→CORE flips.** Every test still misses its expected verdict; the §3
classification holds. A pass that checks each KNOWNBUG test against its expected regex flagged
**no** candidate (the test harness `sys.exit(77)`s any KNOWNBUG that matches — none did).

### 8a. The Python push shifted error signatures *deeper into the same clusters*
The class/tuple/dict-comprehension work did not open a new isolated point fix; it advanced
several tests from one design-cluster failure to a *later* one in the same cluster:

| Test | §7 failure mode | Today's failure mode | Still bottoms out in |
|---|---|---|---|
| `depth_first_search` | `ABORT`→diagnostic `__ESBMC_get_object_size: …non-array object` (#5042) | `ERROR: function call: argument "…@search_from@node" type mismatch: got pointer, expected struct` | object pointer-vs-value model (#3067): the nested `search_from`'s param is typed as a `Node` struct by value, but `startnode`/`nextnode` arrive as pointers |
| `github_4782_object_size` | fast clean diagnostic (`…non-array object`) | BMC **timeout** — the set/graph is now modelled far enough that it no longer reaches the non-array `object_size` guard, and instead hits the unwinding wall | object/set model (#3067) → §3c perf |
| `shortest_paths(_fail)` | `ERROR: Only simple targets are supported in DictComp` | `ERROR: DictComp tuple target requires iterating a list of tuples` (after `#5324`) | dict-key-tuple modelling + any-typed param (`{v: … for u, v in weight_by_edge}` iterates a **dict**'s tuple keys, and `weight_by_edge` is an unannotated parameter — #2848) |

The `depth_first_search` and `github_4782_object_size` cases are the *same* DFS-over-graph
program (set membership + nested closure + generator expression); both are #3067 object-model
work, not isolated bugs. `shortest_paths` needs dict-comprehension iteration over a dict's
tuple keys plus `.items()` tuple unpacking, `min`, and dict `==` — the §3b dict/tuple-equality
cluster, gated additionally by any-typing on the parameter.

### 8b. Everything else: unchanged disposition
Re-confirmed failure modes on today's master: perf/timeout (§3c, policy-banned) for
`breadth_first_search(_fail)`, `knapsack(_fail)`, `next_permutation(_fail)`,
`reverse_linked_list`, `shortest_path_lengths(_fail)`, `topological_ordering(_fail)`, and
humaneval_37/86/90/93/158; clean "unsupported feature" errors for the tuple-unpack gaps
(`minimum_spanning_tree` "Cannot unpack signedbv", `powerset` "Cannot unpack pointer",
`shortest_path_length` "Cannot unpack empty"), mixed-type `sorted()` (humaneval_123) and
non-constant tuple slice (humaneval_148); wrong-verdict soundness clusters (humaneval_1/1-1/
67/91/95/145, `detect_cycle`, `wrap`, `bare_raise_nested`, `github_4117_function_internal`);
`concurrency_fail` (threading MVP); `bitcount_fail` UNKNOWN (§3d); humaneval_162 `hashlib`
(infeasible). Each remains a design-level blocker, a policy-banned timeout, a substantial
sound feature, or a questionable test expectation — not an isolated point fix.

**Build note.** This sweep ran a Z3-only `esbmc`. The four `--bitwuzla`-pinned tests
(humaneval_39, `flatten_fail`, `rpn_eval`, `rpn_eval_fail`) were re-run under `--z3` and
reproduce the §3c perf/timeout disposition; the solver pin does not change their
classification.

**Bottom line.** §7b's conclusion is reaffirmed five days and 63 commits later: no new
isolated, soundly-fixable point fix is available on current `master` without the §5
architectural work. The recent class/tuple/dict-comprehension landings are real progress on
the §5 roadmap (they move tests *through* the cluster), but the remaining KNOWNBUGs are still
gated on flow-sensitive class tracking, the object/pointer-vs-value model, tuple-unpack
inference, any-typing, and string/tuple-equality soundness. The §5 priority order stands.

---

## 9. 2026-06-17 re-validation — one-day window, no flips, no signature shift

Independent re-run of **all 46 KNOWNBUG python/quixbugs/humaneval tests** against current
`master` (tip `74da7c0400`), each with its own `test.desc` flags. Only **4 functional commits**
landed since §8's tip `bd2e4a7d79`: `#5381` (multi-element list repetition `[a,b]*n` with a
runtime count), `#5382` (don't model non-scalar uninterpreted functions as native SMT UFs —
fixes an `smt_sort.h:220` abort on pointer/aggregate UF args), `#5384` (fma interval lifting),
and `#5368` (witness 2.1 branching waypoints). `#5383` was the §8 report itself.

**Result: zero KNOWNBUG→CORE flips.** The full set was driven through `ctest` (which exits
non-zero on any KNOWNBUG that now matches its expected regex): `100% tests passed, 0 tests
failed out of 46` — every test still misses its expected verdict. The §3 classification holds.

### 9a. None of the four commits touches a Python KNOWNBUG path
The two Python-relevant commits were checked against the actual test set rather than assumed:

- **`#5381` (multi-element `[a,b]*n`)** — a scan of all 46 KNOWNBUG sources for multi-element
  list-repetition shows **no** test uses it (the only `*`-with-list hit, humaneval_145
  `n[0] * neg`, is a scalar multiply). So the fix cannot shift any KNOWNBUG signature.
- **`#5382` (non-scalar UF abort)** — targets C/SV-COMP harnesses that declare pointer/aggregate
  `__ESBMC_uninterpreted_*` callbacks (the `aws-c-common` hash-table reproducer, #5145/#5287).
  No Python KNOWNBUG routes a pointer/aggregate through an uninterpreted function, so it is
  out of scope here. (Distinct from the separately-tracked cmath-intensive `smt_sort.h:123`
  crash, which is pre-existing and unrelated.)

### 9b. The §8 signature-shifted tests re-confirmed identical
The three tests §8 called out as having moved deeper into their clusters reproduce **the same
signatures** today:

| Test | Today's signature (tip `74da7c0400`) | Bottoms out in |
|---|---|---|
| `depth_first_search` | `ERROR: function call: argument "…@search_from@node" type mismatch: got pointer, expected struct` (SIGABRT) | object pointer-vs-value model (#3067) |
| `github_4782_object_size` | BMC **timeout** (no longer reaches the non-array `object_size` guard) | object/set model (#3067) → §3c perf |
| `shortest_paths` | `ERROR: DictComp tuple target requires iterating a list of tuples` (after `#5324`) | dict-key-tuple modelling + any-typing (#2848/#3067) |

### 9c. Everything else: unchanged disposition
Re-confirmed on today's master: perf/timeout (§3c, policy-banned) for `breadth_first_search(_fail)`,
`knapsack(_fail)`, `next_permutation(_fail)`, `reverse_linked_list`, `shortest_path_lengths(_fail)`,
`flatten_fail`, `topological_ordering(_fail)`, humaneval_33/37/90/93/158 — several of these
genuinely ran 240–360 s against their own TIMEOUT property, i.e. the honest unwinding wall;
clean "unsupported feature" errors for the tuple-unpack gaps and mixed-type `sorted()` /
non-constant tuple slice; wrong-verdict soundness clusters; `concurrency_fail` (threading MVP);
`bitcount_fail` UNKNOWN (§3d); humaneval_162 `hashlib` (infeasible). Each remains a design-level
blocker, a policy-banned timeout, a substantial sound feature, or a questionable test
expectation — not an isolated point fix.

**Bottom line.** One day and four functional commits after §8, the conclusion is reaffirmed:
no new isolated, soundly-fixable point fix is available on current `master` without the §5
architectural work. The two Python-touching landings (`#5381`, `#5382`) do not intersect the
open KNOWNBUG set. The §5 priority order stands.

---

## 10. 2026-06-19 re-validation & cmath inverse-trig crash fix

Re-test against current `master` (tip `79c8b93eb0`), after the **17 commits** since §9's tip
`74da7c0400` (notably `#5403` numpy float/complex literal folding, `#5407` np.copysign/fmax/fmin
scalar-constant crash, `#5404` calloc zero-size, and the V.3 IREP2 frontend landings
`#5391/#5392/#5401/#5402/#5406/#5408/#5412`). The §3 KNOWNBUG classification is unchanged —
none of these commits touches a cmath inverse-trig path, and the cmath fix below lives entirely
in the `caller == "cmath"` dispatch entry, so it cannot move any non-cmath KNOWNBUG.

### 10a. New isolated, soundly-fixable defect found & fixed
**`cmath.acos`/`cmath.acosh` crashed on pure-imaginary inputs (PR #5415).**

`cmath.acos(0.5j)` and `cmath.acosh(0.5j)` aborted with
`ERROR: function call: argument "…models/cmath.py@F@acos@z" type mismatch: got pointer,
expected struct` (SIGABRT / core dump). The "cmath inverse functions" dispatch entry in
`function_call/expr.cpp` matched only `asin`/`atan`/`asinh`/`atanh`, and `acos` was likewise
absent from `python_math::is_unary_dispatch_function`. So `cmath.acos`/`cmath.acosh` matched no
cmath/math handler and fell through to a generic call path that passed the complex literal as a
**pointer** to the model function's by-value `struct` parameter — the same `got pointer, expected
struct` argument-binding fault family seen in §8/§9's `depth_first_search`, but here from a
missing dispatch case rather than the object/pointer-vs-value model.

`acos`/`acosh` were excluded from the original fast path on purpose: that path returns
`complex(0, ·)`, and their pure-imaginary result has a **nonzero** real part. But the exclusion
left no correct route at all.

**Fix:** add `acos`/`acosh` to the dispatch entry with their own closed-form pure-imaginary fast
path, valid for **all** real `y` (guard is just `z.real == 0`):
`acos(i*y) = (pi/2, -asinh(y))`, `acosh(i*y) = (asinh(|y|), copysign(pi/2, y))`. Every other
input still routes to the `cmath.py` model unchanged. The fast path matches CPython bit-for-bit
and is strictly more accurate than the model on the imaginary axis (the model returns `0`
instead of `pi/2` for `acosh(0j)` and is 1 ULP off elsewhere). New regression pair
`regression/python/cmath_inverse_pure_imag{,_fail}`; dual-solver Bitwuzla+Z3 agreement; all 133
`python/(cmath|complex)` regression tests green; code-reviewed (0 critical/major/minor blocking).

Unlike §6a/§7a, this fix **does** restore a working feature (acos/acosh on the imaginary axis now
verify with exact values) rather than only converting a crash to a diagnostic — it is the §5-item-5
robustness category, but with a sound value model rather than an "unsupported feature" stub.

### 10b. Everything else: unchanged disposition
The §3 design-level blockers, §3c policy-banned timeouts, §3d questionable expectation, and the
infeasible `hashlib` case all stand. No further isolated, soundly-fixable point fix is available
on current `master` without the §5 architectural work; the §5 priority order stands.

---

> **Note on numbering.** §11 (PR #5510), §12 (PR #5513), and §13 (PR #5515) are all in flight and
> not yet on `master`; this section is appended as §14 of the fourth 2026-06-21 sweep so the
> four same-day PRs do not collide on the section number. When they land, the maintainer orders
> §11 → §12 → §13 → §14.

## 14. 2026-06-21 re-validation (fourth sweep) & inverted constant-int soundness fix

Re-test against current `master` (tip `4a5b002c26`, still unchanged; PRs #5510/#5513/#5515 OPEN,
awaiting review). KNOWNBUG classification unchanged. With the §11 crash backlog drained, this
sweep probed string-method idioms and found a **soundness** defect (false verdicts), not just a
crash — the highest-value class.

### 14a. New isolated, soundly-fixable defect found & fixed
**An inverted constant-int guard silently broke `str.center`/`ljust`/`rjust`/`zfill` and
`expandtabs`, proving false assertions.**

`"42".zfill(5) == "00042"`, `"hi".center(6,"*") == "**hi**"`, `"hi".ljust(5,".")`,
`"hi".rjust(5,".")`, and `"a\tb".expandtabs(4)` all gave wrong verdicts: a *correct* assertion on
the result reported `VERIFICATION FAILED`, and the result's value/length was effectively nondet
(`expandtabs(4)` silently used the default tabsize 8, giving length 9 instead of 5). `bytes`,
`upper`, `replace`, and `find` were unaffected.

**Root cause** (`src/python-frontend/string/string_method_handler.cpp`, `get_constant_int`). The
helper guarded on `if (!to_integer(expr, tmp)) return false;` — but `to_integer()` returns
**false on success** (CBMC convention). The `!` inverted it: every valid integer constant was
rejected (treated as non-constant), and non-constants were *accepted* with an unset value. Each
width/fill method then hit its `get_constant_int(width_arg, …)` check, concluded the width was
non-constant, and returned `build_nondet_string_fallback()` — so the assignment target became a
nondet string. `find`/`upper` were spared because they are folded by `python_consteval` and never
reach this helper; `bytes`/`replace` take string (not int) arguments.

**Fix:** drop the inverted `!` (`if (to_integer(expr, tmp)) return false;`). One token. All five
methods now compute exact CPython-matching values (verified `"00042"`, `"**hi**"`, `"hi..."`,
`"...hi"`, `"a   b"`), and a wrong-value assertion now correctly `FAILED`. This is a condition
correction (literal change), not a structural branch change; the new regression pair
`regression/python/str_width_methods{,_fail}` pins the corrected value/length contract. A broad
`regression/python/` sweep shows **zero new failures** — the only failures are the pre-existing
Bitwuzla-only environmental set (`--z3`/`--ir`-pinned), none touching these methods.

Unlike §13's crash→diagnostic, this is a true **soundness** fix: ESBMC was proving false
assertions about string-method results, which is worse than a crash. Bitwuzla-only build; the fix
is a frontend constant-folding guard with no SMT encoding, so the verdict is solver-agnostic.

A separate, narrower issue remains visible but out of scope here: `len()` of a space-padded
method result (`len("x".center(7))`) still mis-verifies even though the value (`"   x   "`)
compares equal — a distinct `len`/`strlen` concern on the padded array, tracked for a later sweep.

### 14b. Everything else: unchanged disposition
> **Note on numbering.** §11–§14 (PRs #5510/#5513/#5515/#5518) are all in flight and not yet on
> `master`; this section is appended as §15 of the fifth 2026-06-21 sweep. When they land, the
> maintainer orders §11 → §12 → §13 → §14 → §15.

## 15. 2026-06-21 re-validation (fifth sweep) & dict-union SIGSEGV→diagnostic

Re-test against current `master` (tip `4a5b002c26`, still unchanged; PRs #5510/#5513/#5515/#5518
OPEN, awaiting review). KNOWNBUG classification unchanged. This sweep probed dict/set/int/float/
list method idioms (fresh ground — the §11 string/cmath/numpy battery is drained).

### 15a. New isolated, soundly-fixable defect found & fixed
**Python dict union (`d1 | d2`) SIGSEGV'd in the SMT backend.**

`{"a":1} | {"b":2}` (PEP 584 dict merge, Python 3.9+) crashed with `EXC_BAD_ACCESS` inside
`bitwuzla_mk_term2` — the faulting address decoded to ASCII (`...tuple_`), i.e. a struct/type
irep was handed to the solver as a term pointer. The crash reproduced for every key type; set
union (`{1,2}|{3,4}`) and integer bitwise-or (`6|1`) were unaffected.

**Root cause** (`src/python-frontend/converter/converter_binop.cpp`,
`build_binary_operator_expr`). Set union and the difference/intersection ops are intercepted for
list-typed (set) operands; integer `|` takes the proper bitvector path. But **two dict operands
matched neither** — dict union is unmodeled — so they fell through to `build_binary_expression`,
whose bitwise branch (`is_bitwise_op`) only type-adjusts bv/bool operands and then emitted a raw
`BitOr` over the two dict structs. That malformed term crashed the Bitwuzla encoder.

**Fix:** add a guard beside the existing set-operation handling that rejects bitwise ops
(`BitOr`/`BitAnd`/`BitXor`) on dict operands with a clean
`ERROR: dict union '|' and bitwise operations on dict are not supported`. Modelling dict merge
properly (copy-then-update semantics with right-wins key precedence) is a feature for a dedicated
change; this is the §5-item-5 crash→diagnostic robustness step, bounded and sound. Set/int
bitwise ops and dict `==`/`.update()` are untouched. New regression pair
`regression/python/dict_union_unsupported` (clean error — the **C-Live** liveness witness for the
added guard) + `set_union_after_dict_fix` (set/int `|` still verify). Broad `regression/python/`
sweep: **zero new failures** (only the pre-existing `--z3`/`--ir` environmental set).

Bitwuzla-only build; the fix is a frontend dispatch guard with no SMT encoding, so the verdict is
solver-agnostic.

### 15b. Other fresh defects observed this sweep (catalogued, not yet fixed)
The dict/int/float probe also surfaced (deferred — several are unmodeled-method feature gaps, and
one is blocked behind an in-flight PR):
- `int.bit_count()`, `int.from_bytes()`, `int.to_bytes()`, `float.is_integer()`, `float.hex()` —
  **unmodeled methods**; ESBMC replaces the undefined function with `assert(false)`, so any program
  calling them reports a spurious `FAILED`. Each is an isolated "add the model" candidate.
- `set.isdisjoint()` — `ERROR: Object "" not found` (dispatch/model gap).
- `len()` of a space-padded `str.center()` result still mis-verifies (noted in §14) — blocked
  behind PR #5518, which must land first to make `center` return a concrete value.

### 15c. Everything else: unchanged disposition
The §3 design-level blockers, §3c policy-banned timeouts, §3d questionable expectation, and the
infeasible `hashlib` case all stand. The §5 priority order stands.
> **Note on numbering.** §11 (the first 2026-06-21 sweep — the bare imaginary-literal
> complex-argument crash) is in flight as PR #5510 and not yet on `master`; this section is
> appended as §12 of the same day's second sweep so the two PRs do not collide on the section
> number. When both land, the maintainer orders §11 before §12.

## 12. 2026-06-21 re-validation (second sweep) & numpy round(x, decimals) crash fix

Re-test against current `master` (tip `4a5b002c26`, unchanged since §11's sweep — zero new
commits). The open KNOWNBUG classification is therefore identical to §11's: zero
KNOWNBUG→CORE flips, §3 holds. With no master movement, this sweep instead drained the
isolated-crash backlog surfaced by §11's idiom-probing battery.

### 12a. New isolated, soundly-fixable defect found & fixed
**`np.round(x, decimals)` — the 2-argument form — aborted on scalar-constant operands.**

`np.round(2.567, 1)` crashed (`WARNING: Unknown operator: round`, then SIGABRT). The 1-argument
form (`np.round(2.567)`) and `np.around` (clean "unsupported" error) were fine; only the 2-arg
`round` crashed.

**Root cause** (`src/python-frontend/numpy_call_expr.cpp`). `round` is registered in
`is_math_function()`, so a 2-arg scalar-constant call enters the scalar-fold block — but `round`
was **absent from the `compute_scalar_result` table and has no `operator_map()` entry**, so it
fell through to the generic `create_binary_op("round", …)` BinOp path, which has no rule for
`round` → "Unknown operator" → `migrate_expr` abort. This is the **exact** trap the code already
special-cased two lines above for `copysign`/`fmax`/`fmin` ("have no operator_map() entry and no
handler, so the BinOp path below crashes migrate_expr").

**Fix:** add `round` to the scalar-fold table and to the special-case fold guard, folding the
scalar-constant case as `std::nearbyint(x * 10^d) / 10^d`. Under the default FP rounding mode
this is round-half-to-even, so it matches numpy bit-for-bit — verified for positive decimals
(`round(2.567,1)==2.6`, `round(2.567,2)==2.57`), zero decimals, **negative** decimals
(`round(12345,-2)==12300`), and banker's rounding (`round(2.5,0)==2.0`, `round(3.5,0)==4.0`).
The symbolic-operand case already degrades to a clean `Unsupported Numpy call: round` diagnostic
(same as `copysign`/`fmax`/`fmin`, which also fold only constants), so no crash path remains.

New regression pair `regression/numpy/round_decimals{,_fail}` (CORE, `--incremental-bmc`); the
positive test is the **C-Live** liveness witness for the added `round` branch (it SIGABRT'd
pre-fix). Full `regression/numpy/` suite **368/368 green**; the change mirrors an established,
already-shipped precedent in the same function. Like §10a/§11a this **restores a working feature**
(2-arg `np.round` now verifies with exact values) rather than only converting a crash to a
diagnostic. Bitwuzla-only build (`ENABLE_Z3=OFF`), as §11; the fix is compile-time scalar
constant-folding in the frontend with no SMT encoding, so the verdict is solver-agnostic.

### 12b. Everything else: unchanged disposition
## 11. 2026-06-21 re-validation & bare-imaginary-literal argument crash fix

Re-test against current `master` (tip `4a5b002c26`), after the **83 commits** since §10's tip
`79c8b93eb0` — predominantly the V.3 IREP2 frontend rebuild (`#5454`/`#5459`/`#5472`–`#5493`
build comparison constant-folds, list/tuple/set/slice/string-index guards, any()/all()
reductions, complex div-by-zero and numpy-overflow asserts, pointer is-None checks, etc.) plus
`#5502` (reject str-variable `%` formatting instead of crashing). A `ctest` flip-check over the
39 KNOWNBUG `humaneval`/`quixbugs` tests reproduced **zero KNOWNBUG→CORE flips** — every test
that completed (including the design-cluster cases `depth_first_search`, `detect_cycle`,
`minimum_spanning_tree(_fail)`, and the perf-timeout cases `humaneval_33`/`37`/`39`/`93`/`158`,
which genuinely ran 120–240 s against their own TIMEOUT property) still misses its expected
verdict; the §3 classification holds. None of the 83 commits touches the bare-imaginary-literal
argument path, and the fix below lives entirely in the general call-argument lowering in
`function_call/expr.cpp`, so it cannot move any KNOWNBUG verdict.

### 11a. New isolated, soundly-fixable defect found & fixed
**A bare pure-imaginary literal passed directly as a by-value `complex` argument crashed
(SIGABRT, "got pointer, expected struct").**

`f(0.5j)` — where `f` takes a `complex` parameter — aborted with

```
ERROR: function call: argument "…@F@f@z" type mismatch: got pointer, expected struct
```

This is the **root** of the `got pointer, expected struct` argument-binding family the prior
sweeps repeatedly hit (§8/§9 `depth_first_search`, §10 `cmath.acos`/`acosh`): §10 (PR #5415)
only *worked around* it for `acos`/`acosh` via a pure-imaginary fast path that bypasses the
model call. Probing the rest of the surface showed it is **general** — every cmath model
function crashes on a bare imaginary literal (`cmath.exp(0.5j)`, `sqrt`, `sin`, `cos`, `tan`,
`sinh`, `cosh`, `tanh`, `phase`, `polar`), and so does any plain user function
`def f(z: complex)`. The crash fires **only** for a bare imaginary `ast.Constant` used
*directly* as an argument; every other form already worked: `f(1+1j)`, `f(0+0.5j)` (BinOp),
`w = 0.5j; f(w)` (variable), `f(complex(0, 0.5))` (ctor) — none crashed.

**Root cause** (`src/python-frontend/function_call/expr.cpp`,
`function_call_expr::handle_general_function_call`). `ast2json` serialises a Python complex
value as its `str()` representation (`"0.5j"`), so a complex `Constant`'s JSON `value` field is
a **string**. `get_literal` correctly reads the `esbmc_type_annotation == "complex"` /
`real_value` / `imag_value` fields and returns a proper `complex` struct. But a post-`get_expr`
override unconditionally replaced *any* `Constant` whose JSON `value` is a string with a string
literal — clobbering the complex struct, which then fell into `build_address_of`, yielding
`&"0.5j"` (a `pointer`) bound to the by-value `complex` (`struct`) parameter → the symex abort.
A BinOp/Name/Call argument has a different `_type`, so the override never fired — hence those
forms worked.

**Fix:** guard the string-literal override to skip complex-annotated constants
(`!arg_is_complex_literal && _type == "Constant" && value.is_string()`). Minimal, general (no
per-function workaround), and confined to the Python frontend — it only narrows a frontend
override that was already wrong for complex constants. New regression pair
`regression/python/imag_literal_arg{,_fail}` (`imag_literal_arg` is the **C-Live** liveness
witness for the added guard: without the fix its `f(0.5j)` call took the clobbering path and
SIGABRT'd; with the fix it takes the new guarded path and verifies). After the fix, the full
user-function + cmath repro set returns `VERIFICATION SUCCESSFUL`; the six already-working forms
are unchanged; the cmath/complex regression subset (174 tests) is 100% green; CPython sanity
(`check_python_tests.sh imag_literal_arg`) passes.

Unlike §6a/§7a, and like §10a, this fix **restores working behaviour** (every cmath function and
every user function now accepts a bare imaginary literal) rather than only converting a crash to
a diagnostic — §5-item-5 robustness, but with a sound value model. This sweep ran a
Bitwuzla-only `esbmc` (this build has `ENABLE_Z3=OFF`, as the §7/§8 sweeps also ran
single-solver); the fix is a purely syntactic JSON-field guard with no SMT encoding, so the
verdict is solver-agnostic.

### 11b. Everything else: unchanged disposition
The §3 design-level blockers, §3c policy-banned timeouts, §3d questionable expectation, and the
infeasible `hashlib` case all stand. No further isolated, soundly-fixable point fix is available
on current `master` without the §5 architectural work; the §5 priority order stands.

---

> **Note on numbering.** §11 (PR #5510, bare imaginary-literal complex-argument crash) has landed
> on `master`; §12 (PR #5513, `np.round(x, decimals)` crash) is still in flight and not yet on
> `master`. This section is appended as §13 of the third 2026-06-21 sweep so the same-day PRs do
> not collide on the section number. The maintainer orders §11 → §12 → §13.

## 13. 2026-06-21 re-validation (third sweep) & bytearray crash→diagnostic

Re-test against current `master` (tip `4a5b002c26`, still unchanged — zero new commits since §11;
PRs #5510 and #5513 OPEN, awaiting review). KNOWNBUG classification identical to §11/§12: zero
flips, §3 holds. This sweep drained the last isolated crash from §11's idiom-probing battery.

### 13a. New isolated, soundly-fixable defect found & fixed
**`bytearray` crashed ESBMC with an uncaught C++ exception (SIGABRT).**

`bytearray([1,2,3]); b[0]=9` and `bytearray(3)` aborted with `terminating due to uncaught
exception type2t::symbolic_type_excp`; `bytearray([1,2,3])` followed by a read returned a bogus
verdict. `bytes(...)` (the immutable form) was and remains fine.

**Root cause** (`src/python-frontend/type_handler.cpp::get_typet`). `bytes` is modeled (array of
int8); `bytearray` (its mutable counterpart) is not, so it reached the unsupported-but-defined
fall-through that merely `log_warning`s and returns `empty_typet()`. That empty type then
propagated into symex and was migrated to IREP2, where the empty type id raised
`type2t::symbolic_type_excp` — uncaught → SIGABRT (on `bytearray(n)` / item assignment) or a
silently wrong verdict (on plain construction+read).

**Fix:** add an explicit `bytearray` case beside the `bytes` handler that throws a clean
`std::runtime_error` — ESBMC's established mechanism for "unsupported feature," reported as
`ERROR: bytearray is not supported; use bytes for an immutable byte sequence` (clean exit 254).
All three crashing/wrong-verdict cases now produce the same deterministic diagnostic; `bytes` is
untouched. This is §5-item-5 robustness (crash → clean diagnostic) and, like #5042/§6a, **does not
flip a KNOWNBUG** — it removes a crash on an unsupported feature rather than enabling one.

New regression pair: `regression/python/bytearray_unsupported` (asserts the clean diagnostic) and
`regression/python/bytes_after_bytearray_fix` (guards that the adjacent `bytes` path still
verifies). Both green via `ctest`; CPython sanity passes; a broad `regression/python/` sweep shows
**zero new failures** attributable to the change (the only failures are the pre-existing
Bitwuzla-only environmental set: `--z3`-pinned and `--ir`-pinned tests, none touching `bytearray`).
Bitwuzla-only build (`ENABLE_Z3=OFF`); the fix is a frontend type-resolution guard with no SMT
encoding, so the verdict is solver-agnostic.

### 13b. Everything else: unchanged disposition
The §3 design-level blockers, §3c policy-banned timeouts, §3d questionable expectation, and the
infeasible `hashlib` case all stand. No further isolated, soundly-fixable point fix is available
on current `master` without the §5 architectural work; the §5 priority order stands.

---

> **Note on numbering.** §16–§27 (PRs #5526, #5531, #5532, #5536, #5537, #5540, #5543, #5548,
> #5554, #5555, #5556, #5558) are in flight and not yet on `master`; this section is appended as
> §28.

## 28. 2026-06-22 re-validation (eighteenth sweep) & startswith/endswith position args

Re-test against current `master` (tip `9d429fb23b`). KNOWNBUG classification unchanged — §3 holds.
This sweep added the optional position arguments catalogued in §27b.

### 28a. New isolated, soundly-fixable defect found & fixed
**`str.startswith`/`endswith` rejected the optional start/end position arguments.**

`"abcabc".startswith("bc", 1)` reported `ERROR: startswith() requires one argument` — the
two/three-argument forms `s.startswith(prefix, start[, end])` were unimplemented (a clean error,
not a wrong verdict, hence catalogued as a feature gap).

**Fix** (`string_method_handler.cpp`): before the single-argument dispatch table, route
`startswith`/`endswith` with ≥2 args to a new `handle_startswith_endswith_with_pos`, which
evaluates the call as `s[start:end].startswith(prefix)` — it extracts the constant receiver
string, reads constant `start`/`end` (default `len`), applies Python slice clamping (negative
`+= len`, clamp to `[0, len]`), builds the `s[start:end]` substring literal, and delegates to the
base `handle_string_startswith`/`endswith`. A non-constant receiver/`start`/`end` raises a clean
unsupported-feature error (sound, matching the constant-only support of the other string methods);
`>3` args raises "takes at most 3 arguments".

**Code-review caught a soundness bug before commit:** an empty affix with a *raw* `start > len`
(e.g. `"abc".startswith("", 4)`) must be `False` in CPython, but the over-clamped empty slice plus
the base handler's empty-affix short-circuit would wrongly report `True` — a false `SUCCESSFUL`.
Added a `if (start > len) return False` guard (strict, so `start == len` still matches an empty
affix). Verified bit-for-bit against CPython for start, start+end, negative indices, `start > end`
(empty window), `start > len`, the empty-affix edge, and both methods. New regression pair
`regression/python/str_startswith_pos{,_fail}` (CORE), including the empty-affix `start > len` edge
that the fix pins; the positive test is the liveness witness. CPython sanity passes; the focused
`regression/python/{str,string,startswith,endswith}*` ctest subset (436 tests) shows zero new
failures (the 12 failing are `--z3`/`--ir`/`--boolector` environmental); single-argument
`startswith`/`endswith` and the other one-arg methods are untouched. Solver-agnostic (constant-fold
in the frontend, no SMT-encoding change).

### 28b. Everything else: unchanged disposition
Deferred candidates stand: `zip()` (unmodelled), `list("abc")` (string→list, a "requires constant"
error), symbolic/user-function `max`/`min(key=)`, `list.index()`-in-`try/except`, `int.to_bytes()`,
the bytes/encoding family, `float.hex()` (infeasible), `str.isascii()` (string-soundness), and the
numeric-tower *properties*. The §3 design-level blockers, §3c timeouts, §3d questionable
expectation, and the infeasible `hashlib` case all stand; the §5 priority order stands.

> **Note on numbering.** §16 (PR #5526, `int.bit_count()`), §17 (PR #5531, `set.isdisjoint()`),
> §18 (PR #5532, `float.is_integer()`), and §19 (PR #5536, `int.conjugate()`) are in flight and
> not yet on `master`; this section is appended as §20 so the in-flight PRs do not collide on the
> section number. The maintainer orders §16 → §17 → §18 → §19 → §20.

## 20. 2026-06-22 re-validation (tenth sweep) & set union/intersection/difference methods

Re-test against current `master` (tip `44b2605c1c`). KNOWNBUG classification unchanged — §3 holds.
A fresh idiom battery (dict/list/set/str/numeric methods) found most already modelled; the
isolated unmodelled one fixed below.

### 20a. New isolated, soundly-fixable defect found & fixed
**`set.union()` / `set.intersection()` / `set.difference()` (method forms) were unmodeled.**

The *operator* forms `a | b`, `a & b`, `a - b` already verify (via
`python_set::build_set_{union,intersection,difference}_call`), but the equivalent *method* forms
reported `Unsupported function 'union'/'intersection'/'difference' is reached → VERIFICATION
FAILED`. Only the method-call dispatch was missing — the builders exist and are the ones the
operators already use.

**Fix:** add the three names to `is_set_method_call` and route them in
`python_set::build_set_method_call` to the **same builders** the `|`/`&`/`-` operators call
(`build_symbol(self)` as lhs, the method argument as rhs). Each returns a fresh set and leaves the
receiver unchanged. The set/set equivalence to the trusted operator path is the correctness
guarantee; verified against CPython for union/intersection/difference membership, the
non-mutating property (`a` unchanged), a `frozenset` receiver, and a list-iterable argument
(`a.union([3,4])`, which Python permits and which works because sets are modelled as lists).

**Documented limitations (sound — clean errors, never wrong verdicts):** the existing
single-argument arity guard in `handle_set_method` means the zero-arg copy form `a.union()` and
the variadic form `a.union(b, c)` produce a clean `ERROR: union() takes exactly one argument`
rather than a verdict (Python allows both; supporting them is a follow-up). A non-iterable scalar
argument `a.union(5)` — itself a `TypeError` in CPython — yields a clean `address_of` ERROR, a
pre-existing exposure shared by the already-shipped `update`/`symmetric_difference` methods, not
introduced here.

Like §16a–§19a this **restores a working feature**. New regression pair
`regression/python/set_union_methods{,_fail}` (CORE); the positive test is the liveness witness for
the three added dispatch branches (pre-fix they hit the unsupported-function stub). Focused
`regression/python/set*` ctest subset shows zero new failures (the two failing — `set_difference`,
`set_intersection` — are `--ir`-pinned, environmental on this Bitwuzla-only `ENABLE_Z3=OFF`
build). CPython sanity passes; code-reviewed (0 critical/major). The change is frontend dispatch
reusing proven builders with no SMT-encoding change, so the verdict is solver-agnostic.

### 20b. Everything else: unchanged disposition
Deferred candidates from earlier sweeps stand: `int.to_bytes()` (args + variable-length bytes
return), `float.hex()` (infeasible exact repr, like `hashlib`), `str.isascii()` (string-soundness,
§5-#2), `str.encode()` (bytes return), and the numeric-tower *properties* `int.numerator`/
`denominator`, `float.real`/`imag` (attribute access, not method calls). The §3 design-level
blockers, §3c policy-banned timeouts, §3d questionable expectation, and the infeasible `hashlib`
case all stand; the §5 priority order stands.
> **Note on numbering.** §16–§23 (PRs #5526, #5531, #5532, #5536, #5537, #5540, #5543, #5548) are
> in flight and not yet on `master`; this section is appended as §24.

## 24. 2026-06-22 re-validation (fourteenth sweep) & str.rsplit()

Re-test against current `master` (tip `f23d79805d`). KNOWNBUG classification unchanged — §3 holds.
This sweep took on the largest of the deferred candidates, `str.rsplit()`.

### 24a. New isolated, soundly-fixable defect found & fixed
**`str.rsplit()` was unmodeled, producing a spurious `VERIFICATION FAILED`.**

`str.split()` already verifies; `rsplit()` differs only in the direction `maxsplit` counts from
(the rightmost separators rather than the leftmost) and was unsupported. **Fix:** thread a
`bool from_right` flag through both `python_list::build_split_list` overloads (constant-fold and
symbolic-runtime) and `dispatch_split_method`.

- **Constant separator case:** for `from_right && maxsplit >= 1`, compute the *full* split, then
  **merge the surplus leftmost parts** back together with the separator — this yields exactly
  Python's rsplit result (the rightmost `maxsplit` splits survive) and avoids error-prone
  backward-scan boundary logic. `maxsplit == 0` returns `[input]` (shared with split); `maxsplit < 0`
  (unlimited) falls through to the existing forward loop, which produces an identical token
  sequence in identical order. Verified bit-for-bit against CPython for full split, `maxsplit`
  1/2/≥count/0, separator-absent, multi-character separators, the empty string, and
  leading/trailing/consecutive separators (`"a..b".rsplit(".",1) == ["a.","b"]`).
- **Sound clean-error limitations (never wrong verdicts):** whitespace `rsplit(None, maxsplit)` —
  whose leading/trailing-whitespace asymmetry vs `split()` is subtle — and `rsplit(sep, maxsplit)`
  on a *non-constant* string both raise a clean unsupported-feature error. Without a `maxsplit`,
  `rsplit(None)` == `split(None)`, so that path is handled exactly. The split-specific BinOp
  fast-path (which splits at the *first* separator) is skipped for rsplit.

The `split` (`from_right=false`) path is byte-for-byte behaviour-preserving (the original loop,
relocated into an `else`). `rsplit` is mapped to the `"list"` return type at **both**
type-inference sites (string-literal/BinOp receivers *and* typed-variable receivers — the second
site was caught in code review). Like the other §16a–§23a fixes this **restores a working
feature**. New regression pair `regression/python/str_rsplit{,_fail}` (CORE) covering no-maxsplit,
right-anchored `maxsplit`, `maxsplit == 0`, a consecutive-separator edge, and a typed-variable
receiver; the positive test is the liveness witness. CPython sanity passes; the focused
`regression/python/{str,string,split}*` ctest subset (436 tests) shows zero new failures (the 12
failing are `--z3`/`--ir`/`--boolector` environmental); the existing `split` tests are unaffected.
Code-reviewed (0 critical/major; one minor type-inference parity gap found and fixed before
commit). Solver-agnostic (constant-fold in the frontend, no SMT-encoding change).

### 24b. Everything else: unchanged disposition
Deferred candidates stand: `max`/`min` with a `key=` function and `list.index()`-in-`try/except`
(both wrong-verdicts needing function-application / exception-model machinery, not point fixes);
`int.to_bytes()`, `bytes.hex()`/`decode()`/`str.encode()`, `str.maketrans`/`translate`,
`float.hex()` (infeasible), `str.isascii()` (string-soundness), and the numeric-tower *properties*.
The §3 design-level blockers, §3c timeouts, §3d questionable expectation, and the infeasible
`hashlib` case all stand; the §5 priority order stands.
> **Note on numbering.** §16 (PR #5526, `int.bit_count()`), §17 (PR #5531, `set.isdisjoint()`),
> §18 (PR #5532, `float.is_integer()`), §19 (PR #5536, `int.conjugate()`), and §20 (PR #5537,
> `set` union/intersection/difference methods) are in flight and not yet on `master`; this section
> is appended as §21 so the in-flight PRs do not collide on the section number.

## 21. 2026-06-22 re-validation (eleventh sweep) & dict.clear() wrong-verdict fix

Re-test against current `master` (tip `44b2605c1c`). KNOWNBUG classification unchanged — §3 holds.
A fresh idiom battery (tuple/dict/str/bytes/builtin) surfaced a **wrong-verdict** bug — higher
severity than the prior unmodelled-method additions.

### 21a. New isolated, soundly-fixable defect found & fixed
**`dict.clear()` reported a spurious `VERIFICATION FAILED` (out-of-bounds dereference).**

`d = {"a": 1}; d.clear()` reported a dereference-failure at `list.c:632` (`__ESBMC_list_clear`)
— a wrong verdict on a common, valid operation. **Root cause:** the method dispatch order is
set → list → dict, and `is_list_method_call` ends in a catch-all `return true` that claimed
`clear` (and the other mutators) for *any* receiver — including a dict. `handle_list_clear` then
passed the dict **struct** (a `{keys, values}` pair, tag `__python_dict__`) to
`__ESBMC_list_clear`, which dereferenced it as a `PyListObject` (`l->size = 0`) at the wrong
offset → out-of-bounds.

**Fix:** add a `clear` guard in `is_list_method_call` mirroring the existing `pop`/`copy`/`count`/
`index` pattern — claim `clear` as `list.clear()` only when the receiver resolves to a
`list_type` symbol (or is a BinOp), otherwise fall through to the dict path. Allow `clear` through
the `is_dict_method_call` gate, route it in `handle_dict_method`, and add
`python_dict_handler::handle_dict_clear`, which empties the dict in place by calling
`__ESBMC_list_clear` on both backing lists (`keys`, `values`). The members are `PyListObject*`
(`get_list_type()` is a pointer), so they pass directly; the dict has no separate size field
(`len()` reads list size), so zeroing both lists fully empties it and the dict stays usable
(verified: `d["c"]=3` after clear gives `len==1`).

This **fixes a wrong verdict / memory-corruption**, the highest-value class in this report. Sets
are unaffected (set receivers are `list_type` symbols → still routed to the list path; `clear`
is not in `is_set_method_call`). `list.clear()` is unchanged (still claimed when the receiver is
a list). New regression pair `regression/python/dict_clear{,_fail}` (CORE); the positive test —
which exercises empty-after-clear, key-absence, and re-use — is the liveness witness for the added
`clear` dispatch branch (pre-fix it produced the OOB `FAILED`). CPython sanity passes; the focused
`regression/python/{dict,list}*` ctest subset (449 tests) shows zero new failures (the 47 failing
are all `--z3`/`--ir`/`--boolector` environmental on this Bitwuzla-only `ENABLE_Z3=OFF` build).
Code-reviewed (0 critical/major/minor). Discriminator purity preserved (the guard only reads the
AST / symbol table). Solver-agnostic (frontend dispatch + a void list-clear call, no SMT-encoding
change).

### 21b. Everything else: unchanged disposition
Deferred candidates stand: `int.to_bytes()` (args + bytes-array return), `bytes.hex()`/
`bytes.decode()`/`str.encode()` (bytes/encoding models), `str.maketrans`/`str.translate`,
`float.hex()` (infeasible, like `hashlib`), `str.isascii()` (string-soundness, §5-#2), and the
numeric-tower *properties* (`int.numerator`/`denominator`, `float.real`/`imag` — attribute access,
not methods). The §3 design-level blockers, §3c policy-banned timeouts, §3d questionable
> **Note on numbering.** §16–§25 (PRs #5526, #5531, #5532, #5536, #5537, #5540, #5543, #5548,
> #5554, #5555) are in flight and not yet on `master`; this section is appended as §26.

## 26. 2026-06-22 re-validation (sixteenth sweep) & list.extend(tuple)

Re-test against current `master` (tip `d23bfa2728`). KNOWNBUG classification unchanged — §3 holds.
A fresh idiom battery surfaced another **wrong-verdict / memory-corruption** bug.

### 26a. New isolated, soundly-fixable defect found & fixed
**`list.extend(tuple)` crashed with an out-of-bounds dereference.**

`a = [1]; a.extend((2, 3))` reported `dereference failure: Access to object out of bounds` at
`__ESBMC_list_extend` (`list.c:610`). Python's `list.extend()` accepts any iterable, but the
frontend passed the **tuple struct** straight to the list-extend model, which dereferenced it as
a `PyListObject` — the same struct-as-list confusion class as the §21 `dict.clear()` bug. Extend
already special-cased function-call results and strings, but not tuples.

**Fix** (`python_list::build_extend_list_call`): add a tuple branch — when the operand is a tuple
(detected via the canonical `tuple_handler::is_tuple_type`), materialise its components into a
fresh list (`create_list` + a per-component `build_push_list_call` + `add_type_info`), then extend
from that real `PyListObject*`. A small `build_member` IREP2 helper (non-deref struct member
access, the rvalue counterpart of the existing `build_deref_member`) reads each component. The
branch is gated on tuple-struct type and placed after the function-call/string branches (which
rewrite the operand to a list pointer), so list/string/function-call extend are untouched.

This **fixes a wrong verdict / memory-corruption**, the highest-value class. Both tuple literals
(`a.extend((2, 3))`) and tuple variables (`t = (2, 3); a.extend(t)`) work, since both carry tuple
struct type after `get_expr`. New regression pair `regression/python/list_extend_tuple{,_fail}`
(CORE) covering a two-element literal, a single-element tuple, a tuple variable, a mixed-type
tuple, and the unaffected list-extend path; the positive test is the liveness witness (pre-fix the
tuple cases OOB-crashed). Verified against CPython; CPython sanity passes; the focused
`regression/python/{list,extend,tuple}*` ctest subset (287 tests) shows zero new failures (the 38
failing are `--z3`/`--ir`/`--boolector` environmental on this Bitwuzla-only build); list/str/
function-call extend unaffected. Code-reviewed (0 critical/major; applied the suggestion to reuse
`tuple_handler::is_tuple_type` instead of an inlined tag check). Solver-agnostic (frontend
materialisation + the existing list-extend model, no SMT-encoding change).

### 26b. Everything else: unchanged disposition
A sibling defect was catalogued but **not** fixed here: `list(tuple)` (the `list()` constructor over
a tuple) is broken on a *different* code path and remains open. Other deferred candidates stand:
`str.startswith`/`endswith` with start/end position arguments (clean "requires one argument" error
today), `zip()` (unmodelled), symbolic/user-function `max`/`min(key=)`, `list.index()`-in-
`try/except`, `int.to_bytes()`, the bytes/encoding family, `float.hex()` (infeasible), `str.isascii()`
(string-soundness), and the numeric-tower *properties*. The §3 design-level blockers, §3c timeouts,
§3d questionable expectation, and the infeasible `hashlib` case all stand; the §5 priority order
stands.
> **Note on numbering.** §16–§28 (PRs #5526, #5531, #5532, #5536, #5537, #5540, #5543, #5548,
> #5554, #5555, #5556, #5558, #5562) are in flight and not yet on `master`; this section is
> appended as §29.

## 29. 2026-06-22 re-validation (nineteenth sweep) & tuple(str)

Re-test against current `master` (tip `b2f1ff2beb`). KNOWNBUG classification unchanged — §3 holds.
This sweep implemented the `tuple(str)` conversion catalogued in §27b/§28b.

### 29a. New isolated, soundly-fixable defect found & fixed
**`tuple(str)` errored "only supported over list and tuple arguments".**

`tuple("ab")` (and `tuple(s)` for a constant-string variable) was rejected; CPython yields
`('a', 'b')`. **Fix** (`function_call/expr.cpp`): in the `tuple()` handler, before the final
throw, fold a constant string operand to a `Tuple` AST of single-character string `Constant`s and
recurse through the proven tuple-literal path (new file-local `build_char_sequence_node` helper).
Non-constant strings keep the clean error (sound).

**Code review caught a critical soundness regression before commit.** `ast2json` decodes a bytes
literal to a JSON string identical to a str Constant, so the naive `extract_constant_string` gate
folded `tuple(b"ab")` to `('a', 'b')` — but CPython gives `(97, 98)` (a tuple of ints). That turned
a previously-clean error into a **wrong verdict**. The operand's *type* still distinguishes them
(bytes is an `int` array, str a `char` array), so the fold is now gated on
`et.subtype() == char_type()`; `tuple(b"ab")` correctly falls through to the clean error again.

This **restores a working feature** (`tuple(str)` verifies with the right elements) while keeping
`tuple(bytes)` a sound clean error. New regression tests `regression/python/tuple_from_str{,_fail}`
(CORE) and `tuple_from_bytes_unsupported` (pins the bytes boundary the review exposed); the
pre-existing `tuple-arg-unsupported` test — which asserted the now-implemented `tuple("ab")` error
and whose own comment admitted the program is valid CPython — was repurposed to verify the correct
result. CPython sanity passes; the focused `regression/python/{tuple,list,str,string,bytes}*` ctest
subset (735 tests) shows zero new failures (the 50 failing are `--z3`/`--ir`/`--boolector`
environmental on this Bitwuzla-only build); `tuple(list)`/`tuple(tuple)`/`tuple(int)` are
unaffected. Solver-agnostic (constant-fold in the frontend, no SMT-encoding change).

### 29b. Everything else: unchanged disposition
The sibling `list("abc")` (string→list, a "requires constant" error today) is the obvious next
candidate — the same char-sequence lowering applies on the `list()` constructor path. Other
deferred candidates stand: `zip()` (unmodelled), symbolic/user-function `max`/`min(key=)`,
`list.index()`-in-`try/except`, `int.to_bytes()`, the bytes/encoding family, `float.hex()`
(infeasible), `str.isascii()` (string-soundness), and the numeric-tower *properties*. The §3
design-level blockers, §3c timeouts, §3d questionable expectation, and the infeasible `hashlib`
case all stand; the §5 priority order stands.
> **Note on numbering.** §16–§24 (PRs #5526, #5531, #5532, #5536, #5537, #5540, #5543, #5548,
> #5554) are in flight and not yet on `master`; this section is appended as §25.

## 25. 2026-06-22 re-validation (fifteenth sweep) & min/max key=abs/len

Re-test against current `master` (tip `d23bfa2728`). KNOWNBUG classification unchanged — §3 holds.
This sweep took on one of the catalogued **wrong-verdict** candidates that needs key-function
handling.

### 25a. New isolated, soundly-fixable defect found & fixed
**`max`/`min` with a `key=` builtin returned a spurious `VERIFICATION FAILED`.**

`max([1, -5, 3], key=abs)` returned the plain maximum `3` (key `=` silently dropped) instead of
the key-maximum `-5`. The Python preprocessor already lowered `min`/`max(list, key=lambda x: x[K])`
over tuple literals (`_lower_min_max_with_key_call`), but a builtin key such as `abs`/`len` fell
through and the keyword was ignored.

**Fix** (`src/python-frontend/preprocessor/generator_mixin.py`, run by `python3` at parse time and
FLAIL-mangled into the binary): factor the per-element key extraction into
`_eval_min_max_key_values`, which now supports both the existing `lambda x: x[K]` form **and**
`key=abs`/`key=len` over **constant** elements — the key is computed with Python's own `abs`/`len`
at preprocess time, so it matches CPython by construction. A new `_const_scalar_value` helper reads
a constant through a unary `+`/`-` wrapper (Python parses `-5` as `UnaryOp(USub, Constant(5))`,
which is why negative ints initially failed). The winning **element** (not its key) is returned,
ties breaking toward the first occurrence (CPython semantics: strict `<`/`>` in
`_select_min_max_index`).

**Soundness:** the fold fires only when every key is a provable constant; any other form (symbolic
elements, a user function/complex lambda, a non-constant key) returns `None` and the caller defers
to the regular dispatch — the pre-existing "key dropped" behaviour, never a *wrong* substituted
element. Verified bit-for-bit against CPython for `max`/`min` × `key=abs` (incl. negatives and
abs-ties) and `key=len` over strings. The existing `lambda x: x[K]` path and plain `min`/`max`
(no key) are unchanged. New regression pair `regression/python/min_max_key{,_fail}` (CORE); the
positive test is the liveness witness (pre-fix the `key=abs` assertions produced the wrong
`FAILED`). CPython sanity passes; the focused `regression/python/{min,max,sorted,list,builtin}*`
ctest subset (304 tests) shows zero new failures (the 38 failing are `--z3`/`--ir`/`--boolector`
environmental on this Bitwuzla-only build). Code-reviewed (0 critical/major/minor). Solver-agnostic
(constant-fold in the preprocessor, no SMT-encoding change).

### 25b. Everything else: unchanged disposition
Deferred candidates stand: `max`/`min(key=...)` with **symbolic** elements or a user-defined key
(needs general function application), `list.index()`-in-`try/except` (exception model), `int.to_bytes()`,
`bytes.hex()`/`decode()`/`str.encode()`, `str.maketrans`/`translate`, `float.hex()` (infeasible),
`str.isascii()` (string-soundness), and the numeric-tower *properties*. The §3 design-level
blockers, §3c timeouts, §3d questionable expectation, and the infeasible `hashlib` case all stand;
the §5 priority order stands.
> **Note on numbering.** §16–§26 (PRs #5526, #5531, #5532, #5536, #5537, #5540, #5543, #5548,
> #5554, #5555, #5556) are in flight and not yet on `master`; this section is appended as §27.

## 27. 2026-06-22 re-validation (seventeenth sweep) & list(tuple)

Re-test against current `master` (tip `d23bfa2728`). KNOWNBUG classification unchanged — §3 holds.
This sweep fixed the `list(tuple)` sibling defect catalogued in §26b — a second **wrong-verdict /
crash** on the same tuple-as-list confusion, on a *different* code path than `list.extend`.

### 27a. New isolated, soundly-fixable defect found & fixed
**`list(tuple)` (the `list()` constructor over a tuple) gave a wrong list / OOB crash.**

`list((2, 3))` produced a wrong list (incorrect elements) and `t = (2, 3); list(t)` crashed with a
dereference failure. The `list()` constructor handled empty, `range`, and list-typed arguments,
but a tuple argument fell through to the generic function-call builder, which mistyped the tuple
struct as a list.

**Fix** (two complementary branches in `converter_funcall.cpp`'s `list(...)` block, plus a reusable
helper):
- **Tuple literal** `list((a, b, ...))`: lower the `Tuple` AST node to a `List` node and recurse,
  routing through the well-tested `[]` list-literal path with full element-type tracking.
- **Tuple variable** `list(t)`: after the existing list-typed check, when the evaluated argument is
  a tuple struct (`tuple_handler::is_tuple_type`), call the new
  `python_list::build_list_from_tuple`, which materialises the tuple's components into a fresh list
  (delegating to the existing `build_list_from_exprs`, with a new non-deref `build_member` IREP2
  helper reading each component). The variable branch generalises to any non-literal whose
  `get_expr` yields a tuple struct (`Name`, `Subscript`, a `Call` returning a tuple).

The two branches are cleanly partitioned (the literal branch intercepts `_type == "Tuple"` before
`get_expr`; the variable branch only sees non-`Tuple` nodes), with no overlap or gap, and
`list(list)`/`list(range)`/`list()`/`list("abc")` are untouched. This **fixes a wrong verdict /
memory-corruption**. New regression pair `regression/python/list_from_tuple{,_fail}` (CORE)
covering a two-element literal, a tuple variable, a single-element tuple, mutate-after-construction,
an empty tuple, and the unaffected `list(list)` path; the positive test is the liveness witness.
Verified against CPython; CPython sanity passes; the focused `regression/python/{list,tuple}*`
ctest subset (287 tests) shows zero new failures (the 38 failing are `--z3`/`--ir`/`--boolector`
environmental on this Bitwuzla-only build). Code-reviewed (0 critical/major; consolidated
`build_list_from_tuple` onto the existing `build_list_from_exprs` per the review). Solver-agnostic
(frontend materialisation + the existing list-literal path, no SMT-encoding change).

### 27b. Everything else: unchanged disposition
Deferred candidates stand: `str.startswith`/`endswith` with start/end position arguments (clean
"requires one argument" error today), `zip()` (unmodelled), `list("abc")` (string→list, a separate
"requires constant" error), symbolic/user-function `max`/`min(key=)`, `list.index()`-in-`try/except`,
`int.to_bytes()`, the bytes/encoding family, `float.hex()` (infeasible), `str.isascii()`
(string-soundness), and the numeric-tower *properties*. The §3 design-level blockers, §3c timeouts,
§3d questionable expectation, and the infeasible `hashlib` case all stand; the §5 priority order
stands.
> **Note on numbering.** §16–§22 (PRs #5526, #5531, #5532, #5536, #5537, #5540, #5543) are in
> flight and not yet on `master`; this section is appended as §23.

## 23. 2026-06-22 re-validation (thirteenth sweep) & str.startswith/endswith tuple support

Re-test against current `master` (tip `f23d79805d`). KNOWNBUG classification unchanged — §3 holds.
A fresh idiom battery surfaced a **wrong-verdict** bug (the highest-value class).

### 23a. New isolated, soundly-fixable defect found & fixed
**`str.startswith(tuple)` / `str.endswith(tuple)` reported a spurious `VERIFICATION FAILED`.**

Python's `startswith`/`endswith` accept a *tuple of affixes* and return `True` iff the string
matches **any** element (`"abc".startswith(("x", "ab"))` is `True`). ESBMC's handlers took a single
affix and fed the tuple **struct** straight into `ensure_null_terminated_string`, mis-evaluating it
as a string → a silently wrong `False`.

**Fix:** guard the top of `handle_string_startswith`/`handle_string_endswith` — when the affix
argument is a tuple type, delegate to a new `build_affix_tuple_match`, which ORs the per-element
single-affix match (recursively reusing the scalar handlers over `affix_tuple.operands()`, the
correct way to read an inline tuple literal's elements — the same direct-operand strategy
`tuple_handler::get_tuple_element` uses to avoid the non-addressable-struct-member pitfall #5185).
A tuple passed *by symbol* (no inline operands) throws a clean `runtime_error` rather than silently
returning `False` — sound: an honest unsupported-error, never a wrong verdict. The single-string
path is a byte-for-byte unchanged suffix of the guard, so there is no scalar-path regression.

This **fixes a wrong verdict**, the highest-value class in this report. New regression pair
`regression/python/str_startswith_tuple{,_fail}` (CORE) covering tuple match / no-match / both
`startswith` and `endswith` / single-string forms; the positive test is the liveness witness for
the added guard (pre-fix it produced the wrong `FAILED`). Verified bit-for-bit against CPython;
CPython sanity passes; the focused `regression/python/{str,string,startswith,endswith}*` ctest
subset (436 tests) shows zero new failures (the 12 failing are `--z3`/`--ir`/`--boolector`
environmental on this Bitwuzla-only build). Code-reviewed (0 critical/major/minor; scalar-path
non-regression diff-confirmed). Solver-agnostic (frontend disjunction over existing strncmp/strlen
matches, no SMT-encoding change).

### 23b. Everything else: unchanged disposition
Other fresh wrong-verdicts catalogued this sweep but **not** isolated point fixes: `max`/`min` with
a `key=` function (ignores the key — needs function application over each element), and
`list.index()` of an absent value inside `try/except ValueError` (exception-model interaction).
Deferred method candidates stand: `str.rsplit()` (right-side `split` — two `build_split_list`
overloads + maxsplit-from-right), `int.to_bytes()`, `bytes.hex()`/`decode()`/`str.encode()`,
`str.maketrans`/`translate`, `float.hex()` (infeasible), `str.isascii()` (string-soundness), and the
numeric-tower *properties*. The §3 design-level blockers, §3c timeouts, §3d questionable
expectation, and the infeasible `hashlib` case all stand; the §5 priority order stands.
> **Note on numbering.** §16 (PR #5526, `int.bit_count()`), §17 (PR #5531, `set.isdisjoint()`),
> and §18 (PR #5532, `float.is_integer()`) have landed on `master`; this section is §19.

## 19. 2026-06-22 re-validation (ninth sweep) & int.conjugate() model

Re-test against current `master` (tip `38fd6daaa1`). KNOWNBUG classification unchanged — §3 holds.
This sweep probed the remaining §15b candidates and shipped `int.conjugate()`.

### 19a. Candidates triaged this sweep
A fresh idiom battery over int/float/str/list/dict methods found most already modelled. The
genuinely unmodelled, isolated ones:
- **`int.conjugate()` / `float.conjugate()`** — returns the value unchanged (the complex
  conjugate of a real number is itself). Cleanest sound fix; shipped below (int only — `float`
  has no OM on `master` yet; the `float` OM lands with §18/PR #5532, after which `float.conjugate`
  is a one-line follow-up).
- **`str.isascii()`** — *attempted and withdrawn this sweep.* It is **not** an isolated fix: the
  constant-string `consteval` path drops non-ASCII bytes (so a literal `"café"` is seen as `"caf"`)
  while the symbolic path uses a C runtime model, and the two interact context-dependently (bare
  `assert s.isascii()` vs `== True`/`not`), risking an *unsound* `True` on non-ASCII literals.
  Doing it correctly needs the constant-string representation to preserve raw bytes — a
  string-soundness item (§5-#2), not a method-add. Reverted in full; deferred.
- **`int.to_bytes()`** — needs receiver-passing for an int method that *takes arguments* plus a
  variable-length bytes-array return model; deferred as a larger feature.
- **`float.hex()`** — reproduces CPython's exact hex-float repr; like `hashlib` (§3b), effectively
  infeasible to model soundly. Deferred/declined.

### 19b. New isolated, soundly-fixable defect found & fixed
**`int.conjugate()` was unmodeled, producing a spurious `VERIFICATION FAILED`.**

`(5).conjugate()` reported `Unsupported function 'conjugate' is reached → VERIFICATION FAILED`
even though `int.conjugate()` returns the integer unchanged. **Fix:** add a `conjugate(cls, n)`
classmethod to `models/int.py` returning `n` (mirroring the zero-arg `bit_length` dispatch), and —
critically — register `conjugate` in `type_utils::is_python_model_func`. The latter was found by
code review: without it, verifying a function that calls `x.conjugate()` under `--function` tries
to resolve `conjugate` as a *user* function and aborts with an uncaught `nlohmann::json` type_error
(reproduced; the empty `find_function` result is dereferenced in `converter_funcall.cpp`). The
complex-number `conjugate` handler already guards on `is_complex_type`, so int receivers fall
through cleanly with no collision.

Like §16a/§17a/§18a this **restores a working feature**. New regression tests
`regression/python/int_conjugate{,_fail}` (CORE) plus `int_conjugate_func` (the `--function`
crash-regression witness for the `is_python_model_func` registration — it aborted pre-fix).
Verified for positive/negative/zero and a BinOp receiver against CPython; CPython sanity passes; a
19-test int-method ctest subset is 100% green; code-reviewed (the `--function` MAJOR finding was
caught and fixed before commit). Bitwuzla-only build (`ENABLE_Z3=OFF`); the change is a value model
plus a frontend allowlist entry with no SMT-encoding change, so the verdict is solver-agnostic.

### 19c. Everything else: unchanged disposition
The deferred §19a candidates (`str.isascii()` string-soundness, `int.to_bytes()`, `float.hex()`),
the §3 design-level blockers, §3c policy-banned timeouts, §3d questionable expectation, and the
infeasible `hashlib` case all stand; the §5 priority order stands.
---

> §18 (PR #5532, `float.is_integer()`), §19 (PR #5536, `int.conjugate()`), §20 (PR #5537, `set`
> union/intersection/difference methods), and §21 (PR #5540, `dict.clear()`) are in flight and not
> yet on `master`; this section is appended as §22.

## 22. 2026-06-22 re-validation (twelfth sweep) & str.rpartition()

Re-test against current `master` (tip `44b2605c1c`). KNOWNBUG classification unchanged — §3 holds.
A fresh idiom battery confirmed the list/set/dict mutators (remove/insert/reverse/sort/pop/del,
set.clear/remove/pop/add) all verify; the isolated unmodelled method fixed below.

### 22a. New isolated, soundly-fixable defect found & fixed
**`str.rpartition()` was unmodeled, producing a spurious `VERIFICATION FAILED`.**

`str.partition()` already verifies, but its right-hand mirror `str.rpartition()` reported
`Unsupported function 'rpartition' is reached`. **Fix:** refactor `handle_string_partition`'s body
into a shared `build_partition_tuple(string_obj, sep_arg, location, bool from_right)`;
`handle_string_partition` delegates with `from_right=false`, the new `handle_string_rpartition`
with `from_right=true`. The shared helper searches with `rfind` instead of `find` when
`from_right`, and on a missing separator returns `("", "", input)` (rpartition's shape — the
unmatched receiver lands in the *last* element) versus partition's `(input, "", "")`. Register
`rpartition` in the one-arg string-method dispatch table and map it (like `partition`) to the
`"tuple"` return type in `annotation_conversion.inl` so the 3-tuple is typed correctly (the
`#5114` mistyping guard). Constant-receiver/separator only, matching `partition`; the non-constant
path keeps partition's `("", "", "")` GOTO-no-abort fallback (#4807).

The `from_right=false` path is byte-identical to the prior `partition` implementation (verified by
diff and by the unchanged partition regression behaviour). Like §16a–§20a this **restores a working
feature**. New regression pair `regression/python/str_rpartition{,_fail}` (CORE) covering
last-occurrence split, the not-found `("", "", input)` shape, a **multi-character separator**
(`"xxabxxcd".rpartition("xx")` — the subtlest `pos + sep.size()` path), an empty receiver, and
`len == 3`; the positive test is the liveness witness for the added dispatch entry. Verified
bit-for-bit against CPython for every case; CPython sanity passes; the focused
`regression/python/{str,string,partition}*` ctest subset (436 tests) shows zero new failures (the
12 failing are all `--z3`/`--ir`/`--boolector` environmental on this Bitwuzla-only build).
Code-reviewed (0 critical/major/minor; partition-preservation diff-confirmed). Solver-agnostic
(constant-fold in the frontend, no SMT-encoding change).

### 22b. Everything else: unchanged disposition
Deferred candidates stand: `str.rsplit()` (right-side `split`), `int.to_bytes()` (args +
bytes-array return), `bytes.hex()`/`bytes.decode()`/`str.encode()` (bytes/encoding models),
`str.maketrans`/`str.translate`, `float.hex()` (infeasible, like `hashlib`), `str.isascii()`
(string-soundness, §5-#2), and the numeric-tower *properties* (`int.numerator`/`denominator`,
`float.real`/`imag`). The §3 design-level blockers, §3c policy-banned timeouts, §3d questionable
expectation, and the infeasible `hashlib` case all stand; the §5 priority order stands.
> **Note on numbering.** §16 (PR #5526, `int.bit_count()` model) and §17 (PR #5531,
> `set.isdisjoint()` model) are in flight and not yet on `master`; this section is appended as
> §18 of the next sweep so the in-flight PRs do not collide on the section number. The maintainer
> orders §16 → §17 → §18.

## 18. 2026-06-22 re-validation (eighth sweep) & float.is_integer() model

Re-test against current `master` (tip `38fd6daaa1`). KNOWNBUG classification unchanged — §3
holds. This sweep drained the `float.is_integer()` entry from the §15b backlog.

### 18a. New isolated, soundly-fixable defect found & fixed
**`float.is_integer()` was unmodeled, producing a spurious `VERIFICATION FAILED`.**

`(5.0).is_integer()` reported `Unsupported function 'is_integer' is reached → VERIFICATION
FAILED`. Unlike `int`, the Python frontend had **no `float` operational model at all**, and the
int instance-method receiver-passing dispatch (`x.bit_length()` → `bit_length(x)`) only fired for
`int` receivers, so a float method call could never reach a model even if one existed.

**Fix** (three coordinated, minimal parts):
1. New `src/python-frontend/models/float.py` — a `float` class with `is_integer(cls, x)` returning
   `x == int(x)`, mirroring `models/int.py`'s classmethod-with-value-parameter convention. `int(x)`
   truncates toward zero, so a finite float equals its truncation exactly iff it is integral —
   sound for positive, negative, zero, and non-integral values across the range the bounded checks
   explore.
2. `src/python-frontend/python_converter.cpp` — register `"float"` in the hardcoded model-loader
   list so `float.json` is generated and loaded alongside `int`.
3. `src/python-frontend/function_call/expr.cpp` — broaden the two zero-arg receiver-passing
   dispatch sites from `int`-only to `int`-or-`float`, so `x.is_integer()` lowers to
   `is_integer(x)`. Purely additive; int methods are untouched, and the gate stays zero-arg-only so
   no other float call is mis-routed.

Like §16a/§17a, this **restores a working feature** rather than converting a crash to a
diagnostic. Verified against CPython for `5.0`/`5.5`/`-2.0`/`-2.5`/`0.0` and an expression
receiver (`(4.0+0.5).is_integer()`). New regression pair `regression/python/float_is_integer{,_fail}`
(CORE); the positive test is the liveness witness for the added dispatch branch — without it the
call lowers to the unsupported-function stub and the test would report `FAILED`. A broad
int/float/math/builtin regression subset (247 tests) is 100% green — the shared dispatch broadening
introduces no int-method regressions; CPython sanity (`scripts/check_python_tests.sh
float_is_integer`) passes; error-level pylint on the model is clean. Code-reviewed: 0
critical/major findings. Bitwuzla-only build (`ENABLE_Z3=OFF`); the change is frontend lowering
plus a value model with no SMT-encoding change, so the verdict is solver-agnostic. **Build note:**
adding a new `models/*.py` or `regression/python/*` directory requires re-running `cmake` (the
FLAIL `WILDCARD *.py` glob and ctest discovery are evaluated at configure time) before the model
or test is visible.

### 18b. Everything else: unchanged disposition
The remaining §15b candidates — `int.to_bytes()` (needs receiver-passing for an int method that
*takes arguments*, plus a bytes-array return model) and `float.hex()` (a string-formatting model
on the now-established `float` OM) — stand as the next "add the model" entries. The §3
design-level blockers, §3c policy-banned timeouts, §3d questionable expectation, and the infeasible
`hashlib` case all stand; the §5 priority order stands.
> **Note on numbering.** §16 (PR #5526, `int.bit_count()` model) is in flight and not yet on
> `master`; this section is appended as §17 of the next sweep so the two PRs do not collide on
> the section number. The maintainer orders §16 before §17.

## 17. 2026-06-22 re-validation (seventh sweep) & set.isdisjoint() model

Re-test against current `master` (tip `38fd6daaa1`). KNOWNBUG classification unchanged — §3
holds. This sweep drained the `set.isdisjoint()` entry from the §15b backlog.

### 17a. New isolated, soundly-fixable defect found & fixed
**`set.isdisjoint()` was unmodeled, producing a spurious `VERIFICATION FAILED`.**

`{1,2,3}.isdisjoint({4,5,6})` reported `Unsupported function 'isdisjoint' is reached →
VERIFICATION FAILED` even though the two sets share no elements. `isdisjoint` had no dispatch
entry, so ESBMC lowered the call to the unsupported-function stub. The sibling relations
`issubset`/`issuperset` were already modeled by `python_set::build_set_relation_call`, which
iterates one list and clears a `true`-initialised bool when the per-element predicate fails.

**Fix:** `A.isdisjoint(B)` is the exact dual of `A.issubset(B)` — iterate `A`, but clear the
result when an element **is** present in `B` (a shared element means the sets are not disjoint)
rather than when it is absent. Extend `build_set_relation_call` with a `disjoint` flag selecting
the `in` trigger over `not in`, and add `isdisjoint` to the three dispatch sites
(`is_set_method_call`, the `set(<iterable>)` constructor fast path in `handle_set_method`, and
`build_set_method_call`). The empty-set and any-iterable cases fall out of the existing
list-based model for free (empty receiver → loop body never runs → stays disjoint; the argument
is consumed as a list via `__ESBMC_list_contains`). Verified against CPython for disjoint,
overlapping, empty-receiver, `frozenset`, and `set(<list>)`-receiver cases.

Like §16a, this **restores a working feature** (any program calling `set.isdisjoint()` now
verifies with the correct boolean) rather than converting a crash to a diagnostic. New regression
pair `regression/python/set-isdisjoint{,_fail}` (CORE); the positive test is the **C-Live**
liveness witness for the added branches — without the dispatch branch the call lowers to the
unsupported-function stub and `set-isdisjoint` would report `FAILED`, so its `SUCCESSFUL` verdict
proves the new path executes (the per-report convention used for §11a/§12a/§15a/§16a). The full
`regression/python/` suite shows zero new failures (only the pre-existing Bitwuzla-only
`--z3`/`--ir` environmental set). Code-reviewed: 0 critical/major/medium findings. Bitwuzla-only
build (`ENABLE_Z3=OFF`); the change is frontend IR lowering with no SMT-encoding change, so the
verdict is solver-agnostic.

### 17b. Everything else: unchanged disposition
The remaining §15b unmodeled-method candidates (`int.from_bytes`/`to_bytes` instance-method
dispatch — note the `int.from_bytes` *classmethod* form already verifies — and `float.is_integer()`
/ `float.hex()`, which both need a `float` operational model plus float instance-method
receiver-passing wiring) stand as the next "add the model" entries. The §3 design-level blockers,
§3c policy-banned timeouts, §3d questionable expectation, and the infeasible `hashlib` case all
stand; the §5 priority order stands.
## 16. 2026-06-22 re-validation (sixth sweep) & int.bit_count() model

Re-test against current `master` (tip `38fd6daaa1`, with §11–§15's PRs now landed:
`#5518` width string methods, `#5519` chained-comparison fold, `#5520` dict-union diagnostic,
`#5516` strcmp normalisation, `#5521` startswith/endswith). KNOWNBUG classification unchanged —
§3 holds. This sweep drained the next entry from the §15b "unmodeled method" backlog.

### 16a. New isolated, soundly-fixable defect found & fixed
**`int.bit_count()` was unmodeled, producing a spurious `VERIFICATION FAILED`.**

`x = 13; x.bit_count()` (Python 3.10+ population count) reported
`Unsupported function 'bit_count' is reached → VERIFICATION FAILED` even though `13 == 0b1101`
has three ones. `bit_count` had no operational model, so ESBMC lowered the call to the
unsupported-function stub. The no-argument int instance-method dispatch already works
(`expr.cpp` passes the receiver as the value argument, exactly as for `bit_length`); only the
model body was missing.

**Fix** (`src/python-frontend/models/int.py`): add a `bit_count` classmethod mirroring the
existing `bit_length` template — fold negatives to their magnitude (`bit_count` operates on the
absolute value), then accumulate `n & 1` while right-shifting, bounded by a literal 512-shift
counter (the `--ir` bignum `IntWide` width) so the unwinder has a termination bound and narrow
callsites exit at `n == 0` well before it. Modelling `bit_count` as an eager popcount is sound
in every context (it has no side effects and depends only on the receiver's value), verified
bit-for-bit against CPython for `0`, `255`, `1024`, a negative (`-3 → 2`), an expression
receiver (`(4-1) → 2`), and `13 → 3`.

Unlike the §6a/§7a/§13a crash→diagnostic fixes, this **restores a working feature** (any program
calling `int.bit_count()` now verifies with the exact count) — like §10a/§11a/§12a it adds a
sound value model. New regression pair `regression/python/int_bit_count{,_fail}` (CORE); the
positive test is the **Py-Live** liveness witness for the new model branch (it reported the
unsupported-function `FAILED` pre-fix and `SUCCESSFUL` after). The full `regression/python/`
suite shows zero new failures (only the pre-existing Bitwuzla-only `--z3`/`--ir` environmental
set, e.g. `github_1964_bit_length_bignum` which is `--ir`-pinned and needs Z3). The fix is
FLAIL-mangled into the binary, so the OM rebuild requirement was honoured before testing.
Bitwuzla-only build (`ENABLE_Z3=OFF`); the model is a frontend lowering with no SMT encoding, so
the verdict is solver-agnostic.

### 16b. Everything else: unchanged disposition
The remaining §15b unmodeled-method candidates (`int.from_bytes`/`to_bytes` instance-method
dispatch, `float.is_integer()`/`float.hex()`, `set.isdisjoint()`) stand as the next "add the
model" entries. The §3 design-level blockers, §3c policy-banned timeouts, §3d questionable
expectation, and the infeasible `hashlib` case all stand. No further isolated, soundly-fixable
point fix beyond those candidates is available on current `master` without the §5 architectural
work; the §5 priority order stands.

---

> **Note on numbering.** §30 (PR #5577, `list(str)`), §31 (PR #5579, numeric-tower properties),
> and §32 (PR #5585, `bytes.hex()`) are in flight and not yet on `master`; this sweep is appended
> as §33. Its fix is in flight as PR #5588. When all land, the maintainer orders §30 → §31 → §32 →
> §33.

## 33. 2026-06-24 re-validation (twenty-third sweep) & str.encode()/bytes.decode()

Re-test against current `master` (tip `8567a816e2`). KNOWNBUG classification unchanged — §3 holds.
This sweep continued the bytes/encoding family from §32b with the str↔bytes codec round-trip.

### 33a. New isolated, soundly-fixable defect found & fixed
**Standalone `str.encode()` and `bytes.decode()` reported a spurious `VERIFICATION FAILED`.**

Only the round-trip `s.encode().decode()` was modelled (it returns the original string expr);
`"abc".encode()` and `b"abc".decode()` used standalone were unmodelled → lowered to the
unsupported-function `assert(false)` → `FAILED` (wrong-verdict class).

**Fix** (`string/string_method_handler.cpp`, in `dispatch_decode_join_method`): fold a constant str
to its UTF-8/ASCII bytes (`encode`) and a constant bytes object to its str (`decode`), gated on
**ASCII** (`< 0x80`) where the byte sequence equals the characters in both directions. Non-ASCII /
non-UTF-8 falls through to the existing clean error — CPython raises `UnicodeDecodeError` / needs
multi-byte encoding there too, so no wrong value is produced. A new `extract_constant_bytes` helper
resolves a bytes *variable* to its literal and **rejects an unresolved symbol** (a `bytes`
parameter, or a value not stored as a constant) so it cannot silently mis-fold to `""` — a
soundness gap caught in code review; a genuine `b""` literal is a constant array (not a symbol), so
empty-bytes folding is preserved. `get_type_from_method` maps `str.encode → bytes` and
`bytes.decode → str` at all three receiver sites (Constant, BinOp via `get_string_method_return_type`,
builtin/variable) so the assignment target is typed correctly (the `#5114` mistyping guard).

Like §16a–§20a/§30a/§32a this **restores a working feature**. The existing `s.encode().decode()`
round-trip is unchanged (its block returns before the new standalone code). New regression pair
`regression/python/bytes_encode_decode{,_fail}` (CORE); the positive test is the liveness witness
(FAILED pre-fix) and covers literal/variable receivers, `len`/index, the explicit `utf-8` argument,
and the round-trip. Verified bit-for-bit against CPython; a `bytes` parameter receiver correctly
falls through (no silent `""`); CPython sanity passes; the focused
`regression/python/(str|string|bytes|encode|decode)` ctest subset (457 tests) shows zero new
failures (the 12 failing are the pre-existing `--z3`/`--ir`/`--boolector` environmental set on this
Bitwuzla-only build). Code-reviewed (1 medium soundness finding fixed before commit; 0 remaining).
Solver-agnostic (a frontend constant-fold, no SMT encoding).

### 33b. Next candidate & everything else: unchanged disposition
The str↔bytes ASCII codec is now complete (`hex`/`encode`/`decode`). The obvious continuations are
**`bytes.hex(sep)`** (the optional separator/`bytes_per_sep` arguments) and **`int.to_bytes()` /
`int.from_bytes()`** (variable-length byte arrays — the **next candidate**, the last sizeable
bytes-family entry). Multi-byte (non-ASCII) UTF-8 encode/decode is a larger follow-up. A
separately-tracked, out-of-scope `len`/`strlen` concern remains: `len()` of an *inline* string
method result (`len(b"..".decode())`, `len("..".replace(...))`) mis-measures even though the value
is correct (§14b). Other deferred candidates stand: `zip()`, symbolic/user-function `max`/`min(key=)`,
`list.index()`-in-`try/except`, `str.maketrans`/`translate`, `float.hex()` (infeasible), and
`str.isascii()` (string-soundness). The §3 design-level blockers, §3c timeouts, §3d questionable
expectation, and the infeasible `hashlib` case all stand; the §5 priority order stands.
