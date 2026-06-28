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

> **Note on numbering.** §30 (PR #5577, `list(str)`) and §31 (PR #5579, numeric-tower properties)
> are in flight and not yet on `master`; this sweep is appended as §32. Its fix is in flight as
> PR #5585. When all land, the maintainer orders §30 → §31 → §32.

## 32. 2026-06-24 re-validation (twenty-second sweep) & bytes.hex()

Re-test against current `master` (tip `8567a816e2`). KNOWNBUG classification unchanged — §3 holds.
This sweep took the bytes/encoding family named in §31b, starting with its most isolated entry,
`bytes.hex()`.

### 32a. New isolated, soundly-fixable defect found & fixed
**`bytes.hex()` was unmodelled, producing a spurious `VERIFICATION FAILED`.**

`b"\x01\xab".hex()` should give `"01ab"` (CPython), but `hex` had no handler, so the call lowered
to the unsupported-function `assert(false)` and any program using it reported `FAILED` — the
highest-value (wrong-verdict) class.

**Fix** (`string/string_handler.cpp`): in `handle_string_attribute_call`, intercept `.hex()` with no
args on a bytes receiver and fold a constant bytes object to its lowercase hex string. `str` has no
`.hex()` method, so the attribute is unambiguously a bytes method; bytes are modelled as an int
array (64-bit elements) while `str` is a char array, so the `subtype() != char_type()` check
excludes strings (the same boundary used by the §29/§30 `tuple`/`list`-over-bytes guards). A variable
receiver is resolved to its symbol value; a non-constant byte (where `to_integer` fails) raises a
clean unsupported-feature error, never a wrong verdict. `get_type_from_method`
(`annotation_conversion.inl`) is extended at both receiver sites (Constant literal and
builtin-type/variable) to map `bytes.hex → str`, so the assignment target is typed `str` and
`len`/indexing/concatenation on the result work (the `#5114` mistyping guard, mirrored from
`split`→`list` / `partition`→`tuple`).

Like §16a–§20a/§30a this **restores a working feature**. New regression pair
`regression/python/bytes_hex{,_fail}` (CORE); the positive test is the liveness witness (FAILED
pre-fix) and covers literal/variable receivers, ASCII, empty bytes, indexing, `len` via a variable,
and string composition. Verified bit-for-bit against CPython; `bytes(3).hex()` and sliced-bytes
receivers resolve without crashing; CPython sanity passes; the focused
`regression/python/(str|string|bytes|hex|encode|decode)` ctest subset (457 tests) shows zero new
failures (the 12 failing are the pre-existing `--z3`/`--ir`/`--boolector` environmental set on this
Bitwuzla-only build). Code-reviewed (0 critical/major). Solver-agnostic (a frontend constant-fold,
no SMT encoding).

### 32b. Next candidate & everything else: unchanged disposition
The rest of the bytes/encoding family is the obvious continuation: **`bytes.decode()`** and
**`str.encode()`** (round-trip between a constant str and bytes — `decode` reuses the char-sequence
lowering, `encode` builds the byte array; the **next candidate**), then `bytes.hex(sep)` with the
optional separator argument, and `int.to_bytes()`/`int.from_bytes()` (variable-length byte arrays).
A separately-tracked, out-of-scope `len`/`strlen` concern remains: `len()` of an *inline* string
method result (`len(b"..".hex())`, `len("..".replace(...))`) mis-measures even though the value is
correct — the same item flagged in §14b. Other deferred candidates stand: `zip()`, symbolic/
user-function `max`/`min(key=)`, `list.index()`-in-`try/except`, `str.maketrans`/`translate`,
`float.hex()` (infeasible), and `str.isascii()` (string-soundness). The §3 design-level blockers,
§3c timeouts, §3d questionable expectation, and the infeasible `hashlib` case all stand; the §5
priority order stands.

---

> **Note on numbering.** §30–§35 (PRs #5577/#5579/#5585/#5588/#5592/#5597) are in flight and not yet
> on `master`; this finding is appended as §36.

## 36. 2026-06-24 (twenty-sixth sweep) — int.from_bytes literal-arg root cause (characterized, deferred)

Re-test against current `master` (tip `6d79c6204c`). KNOWNBUG classification unchanged — §3 holds.
This sweep investigated the §35b next candidate — a bytes **literal passed directly as the
`int.from_bytes` argument** — and root-caused it, but it is **not** the isolated one-liner the
catalogue implied: it is a two-layer frontend bytes-literal-argument lowering bug. No fix is shipped;
the precise diagnosis is recorded here for a future focused effort, and the partial patch was reverted
(it converted a detected out-of-bounds into a silent wrong value — strictly worse).

### 36a. Root cause (GOTO evidence)
`int.from_bytes(b"\x01\x02", "big")` reports `dereference failure: array bounds violated` at the
model's `bytes_data[index]`. The **variable** form `c = b"\x01\x02"; int.from_bytes(c, ...)` verifies.
The GOTO shows why:

- **Variable form:** `c = { 1, 2 }` — a 2-element array of 64-bit ints (`build_raw_byte_array`,
  element type `long_long_int`) — passed as `from_bytes(0, &c[0], 1, 0)`. The model's
  `len(bytes_data)`/indexing are correct.
- **Literal form:** `from_bytes(0, (signed long int *)(&"\001\002"[0]), 1, 0)` — the bytes literal
  is lowered as a **char-array string constant** `"\001\002"` (1 byte/element, null-terminated) and
  its address is cast to the model's 64-bit-element bytes pointer. The model reads 8 bytes per
  element over a 3-byte buffer → out of bounds.

Two layers conspire:
1. **`get_literal`/`is_bytes_literal` (`converter/converter_expr.cpp`):** a bare bytes-literal
   *argument* — with no bytes-typed LHS/annotation context — is not recognised as bytes
   (`current_element_type` is not `"bytes"`, and the node reaches `get_literal` without the
   `encoded_bytes` marker that fires `is_bytes_literal`), so it falls through to the string path and
   is built as a char array. The assignment-RHS context works only because the inferred `bytes` LHS
   sets `current_element_type`.
2. **Argument clobber (`function_call/expr.cpp`, ~L4410):** the model-call arg path then rebuilds any
   `Constant` whose JSON `value` is a string into a `string_literal` — the same clobber the adjacent
   complex-literal guard already prevents. Guarding bytes there (mirroring the complex guard, on a
   non-char array subtype) removes the size-mismatch cast but **not** the wrong value, because
   layer 1 already produced a char array.

The complete fix must make layer 1 lower a bare bytes-literal argument as a byte array (preserve/honour
`encoded_bytes` independent of LHS context), after which the layer-2 guard keeps the arg from being
re-clobbered. That is a frontend bytes-literal-lowering change touching `get_literal`'s context
handling — a focused but non-trivial change, deferred. **Sound workaround:** bind the literal to a
variable first (`c = b"\x01\x02"; int.from_bytes(c, ...)`), which verifies today.

### 36b. Next candidate & everything else: unchanged disposition
A **cleaner next candidate** (avoiding the layer-1 lowering work) is the `signed=True` two's-complement
negative path of `int.from_bytes` (a model-logic case), or `bytes.hex(sep)` separator arguments
(extends §32's `bytes.hex`). The bytes-literal-arg lowering (§36a) and the `byteorder=` keyword form
(model parameter named `big_endian`) remain catalogued. The separately-tracked inline-`len`/strlen
concern (§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`, symbolic/user-function
`max`/`min(key=)`, `list.index()`-in-`try/except`, `str.maketrans`/`translate`, `float.hex()`
(infeasible), and `str.isascii()` (string-soundness). The §3 design-level blockers, §3c timeouts, §3d
questionable expectation, and the infeasible `hashlib` case all stand; the §5 priority order stands.
> **Note on numbering.** §30 (PR #5577, `list(str)`), §31 (PR #5579, numeric-tower properties),
> §32 (PR #5585, `bytes.hex()`), §33 (PR #5588, `str.encode()`/`bytes.decode()`), and §34 (PR
> #5592, `int.to_bytes()` receiver forms) are in flight and not yet on `master`; this sweep is
> appended as §35. Its fix is in flight as PR #5597. When all land, the maintainer orders
> §30 → §31 → §32 → §33 → §34 → §35.

## 35. 2026-06-24 re-validation (twenty-fifth sweep) & int.from_bytes default signed

Re-test against current `master` (tip `8567a816e2`). KNOWNBUG classification unchanged — §3 holds.
This sweep took the `int.from_bytes()` defect catalogued in §34b.

### 35a. New isolated, soundly-fixable defect found & fixed
**`int.from_bytes(b, "big")` (the common 2-argument form) raised an uncaught `TypeError`.**

The counterexample was an uncaught exception, not a wrong value: the `int.py` operational model's
`from_bytes(cls, bytes_data, big_endian, signed)` had no default for `signed`, so the preprocessor's
missing-argument check rejected the 2-argument call before the model ran. CPython's signature is
`int.from_bytes(bytes, byteorder, *, signed=False)`; the model's arithmetic was already correct (the
*explicit*-signed form `int.from_bytes(b, "big", False)`, exercised by the pre-existing
`int_from_bytes` regression test, verified).

**Fix** (`models/int.py`): give the model's `signed` parameter the matching `False` default. One
line; the preprocessor's `functionDefaults` mechanism then supplies it for the 2-argument form. No
arithmetic change — a trivial literal (default-value) change, so no Mode C is required. The model is
FLAIL-mangled, so the binary was rebuilt before testing.

New regression pair `regression/python/int_from_bytes_default_signed{,_fail}` (CORE) covering the
2-argument default form, the `signed=False` keyword, big/little endian, a leading zero, and a single
byte; the positive test is the liveness witness (uncaught-exception `FAILED` pre-fix). Verified
bit-for-bit against CPython; the existing `int_from_bytes` (explicit-signed) test still passes;
pylint clean on the model; the focused `regression/python/(int_|bytes|from_bytes|to_bytes)` ctest
subset (31 tests) shows zero new failures.

(Methodology note: the V.3 `python_expr` IREP2 expr-builder module was considered for this fix, but
the defect turned out to be a missing argument default — an uncaught `TypeError` — not an
expr-construction problem, so no new exprs are built and the builder does not apply here.)

### 35b. Next candidate & everything else: unchanged disposition
Two `from_bytes` sub-defects surfaced once the exception was removed and are the obvious **next
candidates**: (1) a bytes **literal passed directly as the argument** — `int.from_bytes(b"\x00\xff",
"big")` — still fails because the constant byte-array argument is not materialized for the model
call (the variable form works); this is the constant-array-arg materialization path, where the
`python_expr` IREP2 builders (`build_symbol`/`build_address_of`) *would* apply. (2) The `signed=True`
two's-complement negative path and the `byteorder=` keyword form (the model parameter is named
`big_endian`). Smaller adjacent entries: `bytes.hex(sep)` separator arguments, multi-byte (non-ASCII)
UTF-8 encode/decode. The separately-tracked inline-`len`/strlen concern (§14b/§32b/§33b) still
stands. Other deferred candidates stand: `zip()`, symbolic/user-function `max`/`min(key=)`,
`list.index()`-in-`try/except`, `str.maketrans`/`translate`, `float.hex()` (infeasible), and
`str.isascii()` (string-soundness). The §3 design-level blockers, §3c timeouts, §3d questionable
expectation, and the infeasible `hashlib` case all stand; the §5 priority order stands.
> **Note on numbering.** §30 (PR #5577, `list(str)` constructor) is in flight and not yet on
> `master`; this sweep is appended as §31. Its fix is in flight as PR #5579. When both land, the
> maintainer orders §30 → §31.

## 31. 2026-06-24 re-validation (twenty-first sweep) & int/float numeric-tower properties

Re-test against current `master` (tip `8567a816e2`). KNOWNBUG classification unchanged — §3 holds.
This sweep implemented the numeric-tower *properties* deferred since §22b — the attribute-access
(not method-call) members of the int/float numeric tower.

### 31a. New isolated, soundly-fixable defect found & fixed
**`int.numerator`/`denominator`/`real`/`imag` and `float.real`/`imag` raised a spurious
`AttributeError`.**

`(5).numerator`, `(5).denominator`, `(5).real`, `(5).imag`, `(2.5).real`, `(2.5).imag` all failed
with `AttributeError: '<x>' has no attribute '<attr>'`: int/float symbol types are not class
structs, so attribute access fell through to the not-a-class error path (`converter_expr.cpp`). A
program reading any of these properties errored out rather than verifying.

**Fix** (`converter/converter_expr.cpp`): intercept in the instance-attribute branch, beside the
existing complex `.real`/`.imag` delegation. For an int/float-typed symbol, return the known
constant — `real`/`numerator` give the value unchanged (a real number is its own real part; an int
is the ratio n/1), `imag` is `0` (`0.0` for float), `denominator` is `1` (int only). `float.numera
tor`/`denominator` deliberately fall through to a clean `AttributeError` — `float` is not a
`Rational` in CPython. `bool` (type id `bool`, not `signedbv`) is intentionally excluded and stays
a clean `AttributeError`.

Like §16a–§20a/§30a this **restores a working feature** (the values are trivially sound — self/0/1)
rather than only converting a crash to a diagnostic. New regression tests
`regression/python/numeric_tower_properties{,_fail}` (CORE; the positive test is the liveness
witness — it errored pre-fix) and `numeric_tower_float_no_numerator_fail` (pins the float
boundary). Verified bit-for-bit against CPython for int/float, negatives, and the wrong-value
FAILED case; expression/chained receivers (`(4-1).numerator`, `x.real.numerator`) remain
pre-existing clean errors (general attribute-on-BinOp / nested-attribute limitations, out of
scope). CPython sanity passes; the focused
`regression/python/(int|float|complex|numeric|conjugate|attr|class|real|imag|bit_)` ctest subset
(194 tests) is 100% green — zero regressions in the attribute-access path. Code-reviewed (0
critical/major). Solver-agnostic (a frontend constant-fold in attribute resolution, no SMT
encoding).

### 31b. Next candidate & everything else: unchanged disposition
The numeric-tower **properties** are now complete; the remaining numeric-tower work is the
bytes/encoding family — `str.encode()` / `bytes.decode()` / `bytes.hex()` — the obvious **next
candidate** (a wrong-verdict cluster needing a bytes-iteration model; `bytes.hex()` over a constant
bytes literal is the most isolated entry, foldable to a constant hex string). Other deferred
candidates stand: `zip()` (unmodelled — confirmed still `SUCCESSFUL` only on simple forms),
symbolic/user-function `max`/`min(key=)`, `list.index()`-in-`try/except`, `int.to_bytes()`,
`str.maketrans`/`translate`, `float.hex()` (infeasible), and `str.isascii()` (string-soundness).
The §3 design-level blockers, §3c timeouts, §3d questionable expectation, and the infeasible
`hashlib` case all stand; the §5 priority order stands.

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
> **Note on numbering.** §30–§36 (PRs #5577/#5579/#5585/#5588/#5592/#5597 and the #5599 triage
> finding) are in flight and not yet on `master`; this sweep is appended as §37. Its fix is in
> flight as PR #5601.

## 37. 2026-06-24 (twenty-seventh sweep) & int.from_bytes signed=True sign-byte fix

Re-test against current `master` (tip `6d79c6204c`). KNOWNBUG classification unchanged — §3 holds.
This sweep took the cleaner `signed=True` candidate named in §36b (avoiding the §36a literal-arg
lowering work by using variable receivers).

### 37a. New isolated, soundly-fixable defect found & fixed
**`int.from_bytes(b, "big", signed=True)` crashed and was logically wrong for big-endian.**

The two's-complement path reported `dereference failure: array bounds violated`, and even the cases
that did not crash were wrong for big-endian. The model located the sign bit with `bytes_data[-1]`:
(1) Python **negative indexing** is not supported by the OM — `bytes_data[-1]` read out of bounds
(the crash); (2) the most-significant byte, which carries the sign, is **byte 0** for big-endian, not
the last byte, so `bytes([255, 0])` big-endian (CPython `-256`) was mis-evaluated independently of the
crash.

**Fix** (`models/int.py`): index the sign byte directly — `0` for big-endian, `bytes_len - 1` for
little-endian — via a ternary (no new branch, so Mode C is not engaged), and reuse the already-computed
`bytes_len`. This **restores a working feature** (signed `from_bytes` now matches CPython) and removes
the crash. New regression pair `regression/python/int_from_bytes_signed{,_fail}` (CORE) covering
big/little endian, positive/negative, single/multi-byte, the big-endian sign-in-first-byte case, and
the unaffected `signed=False` path; the positive test is the liveness witness. Verified bit-for-bit
against CPython for all seven cases; CPython sanity passes; pylint clean on the model; the focused
`regression/python/(int_|from_bytes|bytes)` ctest subset (30 tests) shows zero new failures, and the
existing `int_from_bytes` (signed=False) test still passes. Independent of the in-flight `from_bytes`
PRs (#5597 default-signed, #5599 literal-arg triage): the tests use the explicit `signed=` keyword and
variable receivers, so it verifies on `master` as-is.

### 37b. Next candidate & everything else: unchanged disposition
With `to_bytes` (§34), the `signed` default (§35) and now signed `from_bytes` (§37) done, the
remaining `from_bytes` work is the §36a bytes-**literal-argument** lowering (deferred; needs the
frontend `get_literal` context fix) and the `byteorder=` keyword form (model parameter named
`big_endian`). A cleaner adjacent candidate is **`bytes.hex(sep)`** (the optional separator /
`bytes_per_sep` arguments, extending §32's `bytes.hex`, stacked on PR #5585). The separately-tracked
inline-`len`/strlen concern (§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`,
symbolic/user-function `max`/`min(key=)`, `list.index()`-in-`try/except`, `str.maketrans`/`translate`,
`float.hex()` (infeasible), and `str.isascii()` (string-soundness). The §3 design-level blockers, §3c
> **Note on numbering.** §30–§37 (PRs #5577/#5579/#5585/#5588/#5592/#5597/#5601 and the #5599 triage
> finding) are in flight and not yet on `master`; this sweep is appended as §38. Its fix is in
> flight as PR #5604.

## 38. 2026-06-24 (twenty-eighth sweep) & str.translate(str.maketrans(...))

Re-test against current `master` (tip `6d79c6204c`). KNOWNBUG classification unchanged — §3 holds.
An idiom battery confirmed `swapcase`/`title`/`zfill`/`removeprefix`/`removesuffix` already verify;
`str.translate` (and `str.isascii`, withdrawn as unsound in §19a) were the unmodelled ones. This sweep
took `str.translate` — an independent candidate on `master` (no PR stacking, unlike `bytes.hex(sep)`).

### 38a. New isolated, soundly-fixable defect found & fixed
**`str.translate` was unmodelled, producing a spurious `VERIFICATION FAILED`.**

`"hello".translate(str.maketrans("el", "ip"))` (CPython `"hippo"`) reported `Unsupported function
'translate'` → `FAILED`.

**Fix** (`string/string_handler.cpp`): in `handle_string_attribute_call`, constant-fold the common
idiom `s.translate(str.maketrans(x, y[, z]))` over constant operands — pattern-match the inline
`str.maketrans(...)` call, extract the constant strings, then map each receiver char `x[i] → y[i]`,
delete the characters in `z`, and keep the rest (delete takes precedence, matching CPython); the
`len(x) != len(y)` case raises the CPython `ValueError`. The byte-wise fold is **gated on ASCII
operands** so a multi-byte UTF-8 sequence cannot be remapped/deleted one byte at a time (which would
corrupt the string — a soundness concern caught in code review); non-ASCII operands, a dict table, and
non-constant operands fall through to the existing (unsupported) dispatch — sound, never a wrong
verdict. `translate` already infers a `str` return type, so no type-inference change is needed.

Like §16a–§20a/§32a this **restores a working feature**. New regression pair
`regression/python/str_translate{,_fail}` (CORE) covering the 2-arg map, 3-arg delete, a variable
receiver, absent-char passthrough, and `len`/index on the result; the positive test is the liveness
witness. Verified bit-for-bit against CPython; CPython sanity passes; the focused
`regression/python/(str|string)` ctest subset (444 tests) shows zero new failures (12 solver-pinned
`--z3`/`--ir`/`--boolector` excepted); the ASCII gate confirmed (non-ASCII `translate` falls through).
Code-reviewed (0 critical/major; the one medium finding — non-ASCII byte corruption — closed by the
ASCII gate). Solver-agnostic (frontend constant-fold, no SMT encoding).

### 38b. Next candidate & everything else: unchanged disposition
Remaining catalogued candidates: `bytes.hex(sep)` separator arguments (extends §32, stacks on
PR #5585); the §36a bytes-**literal-argument** lowering (deferred — needs the frontend `get_literal`
context fix); the `int.from_bytes(byteorder=)` keyword form (model parameter named `big_endian`); and
the `str.maketrans`/`translate` dict-table and non-constant forms (fall through cleanly today). The
separately-tracked inline-`len`/strlen concern (§14b/§32b/§33b) still stands. Other deferred candidates
stand: `zip()`, symbolic/user-function `max`/`min(key=)`, `list.index()`-in-`try/except`, `float.hex()`
(infeasible), and `str.isascii()` (string-soundness, §5-#2). The §3 design-level blockers, §3c
timeouts, §3d questionable expectation, and the infeasible `hashlib` case all stand; the §5 priority
order stands.

---

> **Note on numbering.** §16–§29 (PRs #5526, #5531, #5532, #5536, #5537, #5540, #5543, #5548,
> #5554, #5555, #5556, #5558, #5562, #5563) precede this section; this sweep is appended as §30.
> Its fix is in flight as PR #5577 and not yet on `master`.

## 30. 2026-06-24 re-validation (twentieth sweep) & list(str) constructor

Re-test against current `master` (tip `8567a816e2`). KNOWNBUG classification unchanged — §3 holds.
This sweep implemented the `list("abc")` candidate named in §29b — the direct sibling of §29's
`tuple(str)`, on the `list()` constructor path.

### 30a. New isolated, soundly-fixable defect found & fixed
**`list("abc")` (the `list()` constructor over a constant string) crashed with `migrate expr
failed`.**

CPython yields `list("ab") == ['a', 'b']`, but ESBMC crashed: the string array fell through the
`list()` operand dispatch (which handled empty / `range` / list-typed / tuple operands) into the
generic call builder, which mistyped the `char` array as a `PyListObject` pointer.

**Fix** (`converter/converter_funcall.cpp`): in the `list(...)` handler, fold a constant-string
operand to a `List` AST of single-character `Constant`s and recurse through the proven `[]`
list-literal path (reusing `build_char_sequence_node`, hoisted from §29's `tuple(str)` fix in
`function_call/expr.cpp` into the dependency-free `json_utils.h` as a template — a net
de-duplication). The fold is gated on a char-element operand (`subtype() == char_type()`), so a
`bytes` literal — modelled as a 64-bit int array (`type_handler.cpp` `build_array(long_long_int
_type())`) — is excluded: CPython `list(b"ab") == [97, 98]` (ints), not chars.

**Code review (the same boundary §29 flagged) recommended pinning the bytes case.** `list(b"ab")`
previously fell through to the generic builder and *crashed* (`migrate expr failed`); it is now
rejected with a clean `ERROR: list() over a bytes object is not supported; bytes elements are
integers, not characters` (exit 254) — a §5-item-5 crash→diagnostic that also pins the str-only
soundness boundary of the fold, mirroring §29's `tuple_from_bytes_unsupported`.

This **restores a working feature** (`list(str)` now verifies with the right elements) and removes
an adjacent crash. New regression tests `regression/python/list_from_str{,_fail}` (CORE) and
`list_from_bytes_unsupported`; the positive `list_from_str` test is the **Py-Live** liveness
witness for both added branches (pre-fix it crashed). Verified bit-for-bit against CPython for
literal/variable/empty-string and the `list(list)`/`list(tuple)` non-regression; CPython sanity
passes; the focused `regression/python/{list,tuple,str,string}*` ctest subset (739 tests) shows
zero new failures (the 50 failing are the pre-existing `--z3`/`--ir`/`--boolector` environmental
set on this Bitwuzla-only `ENABLE_Z3=OFF` build). Code-reviewed (0 critical/major). Solver-agnostic
(constant-fold + dispatch guard in the frontend, no SMT-encoding change).

### 30b. Next candidate & everything else: unchanged disposition
The `list(str)`/`tuple(str)` char-sequence lowering is now complete on both constructor paths. The
obvious **next candidate** is `str.encode()` / `bytes.decode()` / `bytes.hex()` — the bytes/encoding
family deferred since §22b–§29b (the now-clean `list(bytes)`/`tuple(bytes)` boundaries make a proper
bytes-iteration model the natural next step). Other deferred candidates stand: `zip()` (unmodelled),
symbolic/user-function `max`/`min(key=)`, `list.index()`-in-`try/except`, `int.to_bytes()`,
`str.maketrans`/`translate`, `float.hex()` (infeasible), `str.isascii()` (string-soundness), and the
numeric-tower *properties*. The §3 design-level blockers, §3c timeouts, §3d questionable
expectation, and the infeasible `hashlib` case all stand; the §5 priority order stands.

---

> **Note on numbering.** §30 (PR #5577, `list(str)`), §31 (PR #5579, numeric-tower properties),
> §32 (PR #5585, `bytes.hex()`), and §33 (PR #5588, `str.encode()`/`bytes.decode()`) are in flight
> and not yet on `master`; this sweep is appended as §34. Its fix is in flight as PR #5592. When
> all land, the maintainer orders §30 → §31 → §32 → §33 → §34.

## 34. 2026-06-24 re-validation (twenty-fourth sweep) & int.to_bytes() receiver forms

Re-test against current `master` (tip `8567a816e2`). KNOWNBUG classification unchanged — §3 holds.
This sweep took the `int.to_bytes()`/`int.from_bytes()` candidate from §33b and found `to_bytes`
already partly implemented but broken outside one receiver form.

### 34a. New isolated, soundly-fixable defect found & fixed
**`int.to_bytes()` gave wrong results for the instance form and the literal-receiver form.**

The value handler `handle_int_to_bytes` (`function_call/str_conv.cpp`) builds a correct bytes array,
and the `int.to_bytes(x, ...)` *class* form verified. But (a) the dispatch predicate
(`function_call/expr.cpp`) only fired for a `Name` receiver, so an int *literal* receiver
`(258).to_bytes(2, "big")` fell through to `Unsupported function`; and (b) the *instance* form
`x.to_bytes(...)` inferred the assignment target as `int` (the receiver's type), so the bytes result
was mistyped as a scalar int and `b[0]`/`len(b)` failed with `TypeError: 'int' object is not
subscriptable`.

**Fix:** extend the dispatch predicate to accept a constant-integer literal receiver
(`is_number_integer()`, which excludes bool/float/str literals — consistent with `get_type_from_constant`
mapping a JSON bool to `"bool"`), and map `int.to_bytes → bytes` in `get_type_from_method` at the
Constant-receiver site (literal form) and the builtin/variable site (instance form). The class form
is mapped earlier in `get_type_from_call`, which is consulted first, so it is untouched (the `#5114`
mistyping-guard pattern). The value handler is unchanged.

Like §30a/§32a this **fixes a wrong verdict / mistyping**. New regression pair
`regression/python/int_to_bytes{,_fail}` (CORE) covering instance big/little-endian, the literal
receiver, the class form, zero-padding, and indexing/len; the positive test is the liveness witness.
Verified bit-for-bit against CPython; CPython sanity passes; focused
`regression/python/(int|bytes|str|annotation)` ctest subsets (37 + 39 tests) show zero new failures
(the `--z3`/`--ir`/`--boolector` environmental set excepted, on this Bitwuzla-only build).
Code-reviewed (0 critical/major/minor). Solver-agnostic (frontend type-inference + dispatch, no SMT
encoding).

### 34b. Next candidate & everything else: unchanged disposition
**`int.from_bytes()` returns a wrong value** and is the obvious **next candidate** — a *separate*
operational-model bug: the `models/int.py` `from_bytes(cls, bytes_data, big_endian, signed)`
classmethod's big-endian loop reads `bytes_data[index]`, and despite logic that looks correct
(`int.from_bytes(bytes([1,2]),"big")` should give 258) it verifies `FAILED`. The fault is in the OM
model's bytes indexing / the `signed` default supplied by `_fill_missing_args_with_defaults`, not in
frontend type inference — a model-level fix, distinct from the `to_bytes` dispatch work here.
Smaller adjacent entries: `bytes.hex(sep)` separator arguments, multi-byte (non-ASCII) UTF-8
encode/decode. The separately-tracked inline-`len`/strlen concern (§14b/§32b/§33b) still stands.
Other deferred candidates stand: `zip()`, symbolic/user-function `max`/`min(key=)`,
`list.index()`-in-`try/except`, `str.maketrans`/`translate`, `float.hex()` (infeasible), and
`str.isascii()` (string-soundness). The §3 design-level blockers, §3c timeouts, §3d questionable
expectation, and the infeasible `hashlib` case all stand; the §5 priority order stands.

---

## 48. 2026-06-28 re-validation (thirty-eighth sweep) & bytes.find()/rfind()

Re-test against current `master` (tip `810d1bc2d5`). KNOWNBUG classification unchanged — §3 holds.
This sweep continued the bytes-affix/search family from §47 with the literal `bytes.find`/`rfind`.

### 48a. New isolated, soundly-fixable defect found & fixed
**`bytes.find(sub)`/`bytes.rfind(sub)` returned the wrong index for a literal bytes object.**

`bytes([1,2,3]).find(bytes([2,3]))` should be `1` (CPython), but ESBMC computed the wrong value: like
the §47 affix methods, bytes search is routed through the str `strncmp`/`strlen` machinery, wrong for
the int-array bytes representation (a NUL byte truncates the length).

**Fix** (`string/string_handler.cpp`): fold `bytes.find`/`bytes.rfind` when the receiver is a literal
`bytes([…])` constructor and the argument is either a literal `bytes([…])` subsequence or a single
integer byte (CPython accepts both), computing the first/last occurrence index (or `-1`) directly and
returning a `long_long_int_type()` constant. As in §47 the match is **purely syntactic** (AST only),
so no symbolic, branch-merged, or partially-evaluated value can reach the fold — the soundness hazard
§47's first draft hit is structurally excluded. A str receiver, a variable/expression receiver, a
`b"…"` literal, and the position-argument (2/3-arg) forms all fall through to the existing dispatch,
sound and unchanged. The reverse `rfind` scan is `m <= n`-guarded so `n - m + 1` never underflows
`size_t`, and `matches_at` indexing stays in bounds.

Like §44a (`str.rindex`) this **restores a working feature** for the literal case. New regression pair
`regression/python/bytes_find{,_fail}` (CORE) covering find/rfind, the not-found `-1`, the single-int
argument, the empty subsequence (find→0, rfind→len), repeated occurrences, embedded NUL bytes,
int-result composition, and `str.find`/`rfind` coexistence; the positive test is the liveness witness
(FAILED pre-fix). Verified bit-for-bit against CPython; CPython sanity passes
(`scripts/check_python_tests.sh bytes_find`); the focused `regression/python/(bytes_find|string-rfind|
str_index|bytes)` ctest subset is green (the 12 failing are the pre-existing `--z3` environmental set
on this Bitwuzla-only build). Code-reviewed: confirmed **sound** (no path admits a non-constant value;
search math verified against CPython incl. the `size_t` bounds; 0 critical/major). Solver-agnostic (a
frontend constant-fold, no SMT encoding).

### 48b. Next candidate & everything else: unchanged disposition
The literal bytes affix/search methods now fold (`startswith`/`endswith` §47, `find`/`rfind` here).
Remaining catalogued candidates: literal `bytes.index`/`rindex` (raising variants — need the exception
machinery), `bytes.count`/`replace`/`split`/`join` over literals (same syntactic-fold pattern); the
**symbolic** bytes affix/search methods (the str `strncmp`/`strlen` path is unsound for the int-array
representation — a pre-existing gap needing a bytes-specific symbolic model, §47b); the list-literal
receiver method gap (§46b); `format()` width specs; `str.maketrans`/`translate` dict-table; and
multi-byte (non-ASCII) UTF-8 encode/decode. The separately-tracked inline-`len`/strlen concern
(§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`, symbolic/user-function
## 47. 2026-06-28 re-validation (thirty-seventh sweep) & bytes.endswith()

Re-test against current `master` (tip `810d1bc2d5`). KNOWNBUG classification unchanged — §3 holds.
An idiom battery over the bytes search/affix methods surfaced `bytes.endswith` as silently wrong while
its sibling `bytes.startswith` worked.

### 47a. New isolated, soundly-fixable defect found & fixed
**`bytes.endswith(suffix)` returned the wrong value for a literal bytes object.**

`bytes([1,2,3]).endswith(bytes([2,3]))` should be `True` (CPython), but ESBMC computed `False`. Bytes
are modelled as an int array (64-bit elements), not a null-terminated char array, and both affix
methods were routed through the str `strncmp`/`strlen` machinery. `startswith` happened to work (it
uses the prefix array's static size), but `endswith` uses `strlen()` to compute the from-the-end
offset — wrong for the int-array representation (a NUL byte truncates it) — so it mis-located the
suffix.

**Fix** (`string/string_handler.cpp`): in `handle_string_attribute_call` (beside the §32 `bytes.hex`
block), fold `bytes.startswith`/`bytes.endswith` when the receiver **and** the single argument are
literal `bytes([const, …])` constructor nodes, comparing the byte vectors directly (startswith at
offset 0, endswith at offset `len - suffixlen`) and returning a Python bool. A first expr-based draft
folded the receiver *value*, which a code review showed to be **unsound** — a symbolic or
branch-merged bytes value reaches the fold as an operand-less array (a bare symbol, or a const-eval
artifact whose array type collapses to size 0) and folded to a wrong constant, a false `SUCCESSFUL`.
The fix was rewritten to inspect **only the AST syntax** (`_type == "Call"`, `func.id == "bytes"`, a
single `List` of `0..255` integer `Constant`s): no symbolic, branch-merged, or partially-evaluated
value can ever reach it. A str receiver, a *variable*/expression receiver, a `b"..."` literal, a tuple
argument, and the position-argument (2/3-arg) form all fall through to the existing dispatch, sound and
unchanged (the variable/symbolic bytes affix case remains on the pre-existing `strncmp` path, an
out-of-scope incompleteness, not a wrong-`SUCCESSFUL`).

Like §16a–§20a this **restores a working feature** for the literal case. New regression pair
`regression/python/bytes_endswith{,_fail}` (CORE) covering true/false/full-match/longer suffixes,
embedded NUL bytes (which `strlen` would truncate at), the unchanged `startswith`, and `str.endswith`
coexistence; the positive test is the liveness witness (FAILED pre-fix). Verified bit-for-bit against
CPython; CPython sanity passes (`scripts/check_python_tests.sh bytes_endswith`); the focused
`regression/python/(endswith|startswith|bytes)` ctest subset (41 tests) is 100% green — `str`
startswith/endswith unaffected. Code-reviewed across three rounds: a critical symbolic-receiver fold
was caught and the handler rewritten to the AST-syntactic form, which a final review confirmed
**sound** (no path admits a non-constant value; 0 critical/major). Solver-agnostic (a frontend
constant-fold, no SMT encoding).

### 47b. Next candidate & everything else: unchanged disposition
The constant `bytes.startswith`/`endswith` are now correct. Remaining catalogued candidates: the
**symbolic** bytes affix/search methods (the str `strncmp`/`strlen` path is unsound for the int-array
bytes representation — a pre-existing gap needing a bytes-specific symbolic model, newly characterised
this sweep); the bytes tuple-of-affixes and position-argument forms (same root); the list-literal
receiver method gap (§46b); `format()` width specs; `str.maketrans`/`translate` dict-table; and
multi-byte (non-ASCII) UTF-8 encode/decode. The separately-tracked inline-`len`/strlen concern
(§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`, symbolic/user-function
## 46. 2026-06-28 re-validation (thirty-sixth sweep) & builtin format()

Re-test against current `master` (tip `810d1bc2d5`). KNOWNBUG classification unchanged — §3 holds.
This sweep first probed the list-literal-receiver method gap (`[1,2].count(x)`) — the §45b list
analogue of §42 — but found it is **not** a clean fix: routing the literal to `handle_list_count`
needs the list-element metadata (`list_type_map`) materialised for the temp symbol, deeper
list-machinery work (recorded as deferred, §46b). The sweep then took the builtin `format()`.

### 46a. New isolated, soundly-fixable defect found & fixed
**The builtin `format(value[, spec])` was unmodelled, producing a spurious `Unsupported function`.**

`format(255, "x")` should give `"ff"` (CPython), but the builtin `format` had no handler and reported
`Unsupported function 'format'` → `FAILED`. (The `str.format()` *method* was already handled.)

**Fix** (`function_call/str_conv.cpp`): add `handle_format()`, dispatched from `get_dispatch_table()`
when the call is a bare `Name` call to `format` (`_type == "Name"`, so the `str.format()` method —
an `Attribute` call — is unaffected). It constant-folds a **literal** integer value with a bare
presentation-type spec — `'d'`/`'x'`/`'X'`/`'o'`/`'b'` or empty/default — to the base string (no
`0x`/`0o`/`0b` prefix, leading `-` for negatives, magnitude taken in the unsigned domain so
`LLONG_MIN` is safe), and a constant string with the default spec to itself. Width/alignment/precision
specs (e.g. `"08x"`), float values, and unsupported arity raise a clean error, never a wrong fold.

A code review surfaced a **soundness hazard** that was fixed before commit: the value is folded only
from a genuine literal node (`Constant`/`UnaryOp`), never a `Name` — `extract_constant_integer` would
otherwise resolve a *reassigned* variable (`x = 255; x = 10`) to its stale constant, a potential false
`SUCCESSFUL`. A variable argument is now left unsupported (a clean error, no worse than the prior
fully-unsupported state).

Like §39a this **restores a working feature**. New regression pair
`regression/python/builtin_format{,_fail}` (CORE) covering the five base specs, the default spec,
negatives, zero, a constant string, `len`/index composition, and the `str.format()` method coexistence;
the positive test is the liveness witness (`Unsupported`→`FAILED` pre-fix → `SUCCESSFUL` post-fix).
Verified bit-for-bit against CPython; CPython sanity passes
(`scripts/check_python_tests.sh builtin_format`); the focused
`regression/python/(builtin_format|str_format|format|hex|ascii|oct|bin)` ctest subset (77 tests) is
100% green — `str.format`/`hex`/`oct`/`bin`/`ascii` all unaffected. Code-reviewed (0 critical/major;
the one soundness hazard — `Name`-resolved stale value — fixed before commit). Solver-agnostic (a
frontend constant-fold, no SMT encoding).

### 46b. Next candidate & everything else: unchanged disposition
The builtin `format()` now folds the int base specs and the default. Remaining catalogued candidates:
the **list-literal-receiver** method gap (`[1,2].count(x)`/`.index(x)` — `handle_list_count`/`index`
need the literal's element metadata materialised for the `$literal_list$` temp, a list-machinery
change, newly characterised this sweep); `format()` width/alignment specs (the format mini-language,
larger); `str.maketrans`/`translate` dict-table; multi-byte (non-ASCII) UTF-8 encode/decode; and the
§36a bytes-literal-argument lowering (deferred). The separately-tracked inline-`len`/strlen concern
(§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()` (materialisation via
`list(zip(...))` — a larger feature), symbolic/user-function `max`/`min(key=)`,
## 45. 2026-06-28 re-validation (thirty-fifth sweep) & int.to_bytes() keyword/default arguments

Re-test against current `master` (tip `810d1bc2d5`). KNOWNBUG classification unchanged — §3 holds.
This sweep took the `int.to_bytes` keyword form — the `to_bytes` analogue of §40's `from_bytes`
`byteorder=` keyword fix.

### 45a. New isolated, soundly-fixable defect found & fixed
**`int.to_bytes(2, byteorder="big")` and the no-arg default form errored — only the all-positional
form was accepted.**

CPython's signature is `int.to_bytes(length=1, byteorder='big', *, signed=False)` — both arguments may
be passed by keyword and both default since 3.11. But `handle_int_to_bytes` (`function_call/str_conv.cpp`)
read `length`/`byteorder` purely positionally and required the exact positional count, so
`x.to_bytes(2, byteorder="big")`, `x.to_bytes(length=2, byteorder="big")`, and `x.to_bytes()` all hit
`ERROR: int.to_bytes() expects 2 or 3 positional arguments`.

**Fix** (`function_call/str_conv.cpp`): resolve `length` and `byteorder` each from its positional slot,
else a keyword of that name, else the CPython default (`length=1`, `byteorder='big'`), via two small
`positional()`/`keyword()` lookups. The `value_offset` shift keeps the unbound type-method form
`int.to_bytes(x, ...)` correct (the integer value is the leading positional, so the length/byteorder
slots start one later). The positional slot is consulted before the keyword, so a positional length is
never shadowed. A code review also surfaced a **pre-existing** latent soundness gap — a *non-constant*
byteorder silently defaulted to big-endian, mis-folding a little-endian intent into a wrong byte array
— now closed: an explicit non-constant byteorder raises a clean error (matching the constant-length
guard), never a wrong fold. `signed=` stays accepted-and-ignored, as the positional form already did.

Like §34a/§40a this **restores a working feature**. New regression pair
`regression/python/int_to_bytes_kwargs{,_fail}` (CORE) covering byteorder-keyword, both-keyword,
little-endian keyword, length-only (default byteorder), the no-arg default form, the type-method form
with keywords, and the unchanged positional form; the positive test is the liveness witness (errored
pre-fix). Verified bit-for-bit against CPython; CPython sanity passes
(`scripts/check_python_tests.sh int_to_bytes`); the focused `regression/python/int_(to|from)_bytes*`
ctest subset (10 tests) is 100% green — the existing `int_to_bytes` positional test still passes.
Code-reviewed (0 critical/major; the one medium pre-existing non-constant-byteorder gap fixed before
commit). Solver-agnostic (a frontend value-handler change, no SMT encoding).

### 45b. Next candidate & everything else: unchanged disposition
The `int.to_bytes`/`from_bytes` keyword/default surface is now complete. Remaining catalogued
candidates: `str.maketrans`/`translate` dict-table and non-constant forms (fall through cleanly today);
multi-byte (non-ASCII) UTF-8 encode/decode; `float.as_integer_ratio()` on a literal (returns a tuple —
larger); the list-literal-receiver method gap (`[1,2].count(x)` — the list analogue of §42, a
distinct list-dispatch path); and the §36a bytes-literal-argument lowering (deferred). The
separately-tracked inline-`len`/strlen concern (§14b/§32b/§33b) still stands. Other deferred candidates
stand: `zip()` (materialisation via `list(zip(...))` — a larger feature), symbolic/user-function
`max`/`min(key=)`, `list.index()`-in-`try/except`, `float.hex()` (infeasible), and `str.isascii()`
(string-soundness, §5-#2 — the constant-fold-vs-symbolic interaction makes it unsound, §19a). The §3
design-level blockers, §3c timeouts, §3d questionable expectation, and the infeasible `hashlib` case
all stand; the §5 priority order stands.

> **Note on numbering.** §39 (PR #5661), §40 (PR #5663), §41 (PR #5665), §42 (PR #5668), §43 (PR
> #5669), and §44 (PR #5670) are in flight and not yet on `master`; this sweep is appended as §45.
> When all land, the maintainer orders §39 → §40 → §41 → §42 → §43 → §44 → §45.
## 44. 2026-06-28 re-validation (thirty-fourth sweep) & str.rindex()

Re-test against current `master` (tip `810d1bc2d5`). KNOWNBUG classification unchanged — §3 holds.
An idiom battery over the str search methods surfaced `str.rindex` as the one unmodelled member — the
right-side analogue of `str.index`, missing while `find`/`index`/`rfind` were all handled.

### 44a. New isolated, soundly-fixable defect found & fixed
**`str.rindex()` was unmodelled, producing a spurious `Unsupported function`.**

`"abcabc".rindex("b")` should give `4` (CPython — the last occurrence; like `rfind` but raising
`ValueError` when absent), but `rindex` had no handler and reported `Unsupported function 'rindex'` →
`FAILED`.

**Fix** mirrors the existing `index`/`find` relationship onto `rindex`/`rfind`:
- `string/string_method_handler.cpp`: new `handle_string_rindex()` / `handle_string_rindex_range()`
  call the existing `handle_string_rfind()` / `handle_string_rfind_range()` then reuse
  `build_string_index_result()` — the same builder `index` uses, which raises `ValueError` on a `-1`
  (not-found) result. `rindex` is added to the search-method dispatcher, routed before the `rfind`
  fall-through so it cannot leak into the non-raising path.
- `python_consteval.cpp`: `rindex` is added to the constant-fold path — it searches like `rfind`
  (`window.rfind`) and raises like `index` (returns nullopt on not-found, leaving BMC to raise), so
  the constant and runtime paths agree.

The underlying `__python_str_rfind`/`__python_str_rfind_range` OMs (which return `-1` on not-found, the
same sentinel `find` uses) are reused unchanged — no new operational model. Like §24a (`rsplit`) this
**restores a working feature**. New regression pair `regression/python/str_rindex{,_fail}` (CORE)
covering last-occurrence, the start/end window, the catchable not-found `ValueError`, int-typed result
arithmetic, and a variable receiver; the positive test is the liveness witness (`Unsupported`→`FAILED`
pre-fix → `SUCCESSFUL` post-fix). Verified bit-for-bit against CPython; CPython sanity passes
(`scripts/check_python_tests.sh str_rindex`); the focused `regression/python/(str_rindex|string-rfind|
str_index)` ctest subset is green (the 12 failing are the pre-existing `--z3` environmental set on this
Bitwuzla-only build — confirmed by re-running one without `--z3`). Code-reviewed (0 critical/major;
soundness, const-eval/runtime consistency, dispatcher ordering, and the `build_string_index_result`
reuse all confirmed). Solver-agnostic (a frontend constant-fold + existing OM reuse, no SMT change).

### 44b. Next candidate & everything else: unchanged disposition
The str search family (`find`/`index`/`rfind`/`rindex`) is now complete. Remaining catalogued
candidates: `str.maketrans`/`translate` dict-table and non-constant forms (fall through cleanly today);
multi-byte (non-ASCII) UTF-8 encode/decode; `float.as_integer_ratio()` on a literal (returns a tuple —
larger); and the §36a bytes-literal-argument lowering (deferred). The separately-tracked
inline-`len`/strlen concern (§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`
(materialisation via `list(zip(...))` — a larger feature), symbolic/user-function `max`/`min(key=)`,
`list.index()`-in-`try/except`, `float.hex()` (infeasible), and `str.isascii()` (string-soundness,
§5-#2). The §3 design-level blockers, §3c timeouts, §3d questionable expectation, and the infeasible
`hashlib` case all stand; the §5 priority order stands.

> **Note on numbering.** §39 (PR #5661), §40 (PR #5663), §41 (PR #5665), §42 (PR #5668), §43 (PR
> #5669), §44 (PR #5670), and §45 (PR #5671) are in flight and not yet on `master`; this sweep is
> appended as §46. When all land, the maintainer orders §39 → … → §46.
> **Note on numbering.** §39 (PR #5661), §40 (PR #5663), §41 (PR #5665), §42 (PR #5668), and §43 (PR
> #5669) are in flight and not yet on `master`; this sweep is appended as §44. When all land, the
> maintainer orders §39 → §40 → §41 → §42 → §43 → §44.
## 39. 2026-06-28 re-validation (twenty-ninth sweep) & bytes.hex(sep, bytes_per_sep)

Re-test against current `master` (tip `f343663bde`). KNOWNBUG classification unchanged — §3 holds.
With PR #5585 (`bytes.hex()`) now on `master`, this sweep took its catalogued continuation — the
optional separator arguments named in §32b/§33b/§37b/§38b — a clean direct task on `master` (no PR
stacking).

### 39a. New isolated, soundly-fixable defect found & fixed
**`bytes.hex(sep[, bytes_per_sep])` was unmodelled, producing a spurious `VERIFICATION FAILED`.**

`bytes([1, 2, 3]).hex("-")` should give `"01-02-03"` and `bytes([0xb9, 0x01, 0x9e, 0xf3]).hex("_", 2)`
should give `"b901_9ef3"` (CPython), but the §32 handler only fired for the no-argument form
(`args.empty()`); any separator argument fell through to the unsupported-function `assert(false)` and
reported `FAILED` — the wrong-verdict class.

**Fix** (`string/string_handler.cpp`): widen the `bytes.hex` interception to `args.size() <= 2` and,
when a constant one-character separator (and optional constant `bytes_per_sep`) is supplied, insert the
separator between byte groups. CPython's grouping is reproduced exactly: a **positive** `bytes_per_sep`
groups from the **right** (`(n - i) % g == 0`), a **negative** one from the **left** (`i % g == 0`),
and `0` inserts none (default `1`). The separator string and grouping count are pulled with the
existing `extract_constant_string` / `json_utils::extract_constant_integer` helpers (the latter already
resolves `UnaryOp` negatives and variable receivers); a **non-constant** separator or count is not
folded — `foldable` stays false and control falls through to the regular (unsupported) dispatch, never
a wrong verdict. A `len(sep) != 1` constant is rejected with CPython's `ValueError` text; the separate
"sep must be ASCII" branch is omitted as unreachable — a length-1 ESBMC string is necessarily ASCII
because any single non-ASCII character is multi-byte in UTF-8, so the length check subsumes it. The
method name is unchanged, so §32's `bytes.hex → str` type mapping still applies — no type-inference
change. The grouping negate is done in the unsigned domain (`0ULL - (unsigned long long)group`) so
`LLONG_MIN` cannot overflow (a code-review hardening; CPython rejects that literal anyway).

Like §16a–§20a/§32a/§38a this **restores a working feature**. New regression pair
`regression/python/bytes_hex_sep{,_fail}` (CORE) covering every-byte separation, right-grouping
(`hex("_", 2)`), left-grouping (`hex(":", -2)`), zero-grouping, single-byte (no separator), the
unchanged no-arg form, and `len`/index on the result; the positive test is the liveness witness
(FAILED pre-fix → SUCCESSFUL post-fix, the **C-Live** witness for the added branch). Verified
bit-for-bit against CPython (the grouping was additionally cross-checked against CPython over 20,000
randomized trials in code review with zero mismatches); CPython sanity passes
(`scripts/check_python_tests.sh bytes_hex`); the `regression/python/bytes_hex_sep{,_fail}` ctest pair
is green and the focused `bytes_hex` set shows zero new failures (the pre-existing `--z3`/`--ir`/
`--boolector` environmental set excepted, on this Bitwuzla-only build). Code-reviewed (1 low-severity
signed-overflow finding fixed before commit; 0 remaining). Solver-agnostic (a frontend constant-fold,
no SMT encoding).

### 39b. Next candidate & everything else: unchanged disposition
The bytes/encoding family's `hex` arm is now complete (`hex` / `hex(sep)` / `hex(sep, bytes_per_sep)`).
Remaining catalogued candidates: the §36a bytes-**literal-argument** lowering (deferred — needs the
frontend `get_literal` context fix); the `int.from_bytes(byteorder=)` keyword form (model parameter
named `big_endian`); `str.maketrans`/`translate` dict-table and non-constant forms (fall through
cleanly today); and multi-byte (non-ASCII) UTF-8 encode/decode. The separately-tracked inline-`len`/
strlen concern (§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`, symbolic/
user-function `max`/`min(key=)`, `list.index()`-in-`try/except`, `float.hex()` (infeasible), and
`str.isascii()` (string-soundness, §5-#2). The §3 design-level blockers, §3c timeouts, §3d questionable
expectation, and the infeasible `hashlib` case all stand; the §5 priority order stands.
## 41. 2026-06-28 re-validation (thirty-first sweep) & bytes.fromhex()

Re-test against current `master` (tip `810d1bc2d5`). KNOWNBUG classification unchanged — §3 holds.
With the `bytes.hex` family complete (§32/§39), this sweep took its natural inverse, `bytes.fromhex`
— a constant-fold parsing a hex string into a bytes object, the same shape as the §32 fold run
backwards.

### 41a. New isolated, soundly-fixable defect found & fixed
**`bytes.fromhex("0102")` was unmodelled, producing a spurious `VERIFICATION FAILED`.**

`bytes.fromhex("0102")` should give `b"\x01\x02"` (CPython), but `fromhex` — a `bytes` classmethod —
had no handler, so the call lowered to the unsupported-function `assert(false)` and any program using
it reported `FAILED` — the wrong-verdict class.

**Fix** (`function_call/expr.cpp`): add a `handle_bytes_fromhex()` constant-fold, dispatched from
`get_dispatch_table()` when the function is `fromhex` and the receiver is the `bytes` builtin (the
predicate short-circuits on the method name, so `get_object_name()` runs only for `fromhex` calls).
The handler parses a constant hex string into a byte array via the existing `build_raw_byte_array`
(the same representation as a bytes literal / the `bytes([...])` constructor), reproducing CPython
exactly: pairs of hex digits, ASCII whitespace skipped **between** byte pairs but not **within** one
(a space inside a pair is a `ValueError`), uppercase accepted, the empty string yielding the size-0
`bytes` representation. Odd-length or non-hex input raises CPython's `ValueError` text (a clean
frontend error, never a wrong value); a non-constant string argument is rejected with a clean error
rather than mis-folded. Because `build_raw_byte_array` returns a genuinely `bytes`-typed array, the
assignment target is typed correctly with no type-inference change — `len`, indexing, and the
`fromhex(...).hex()` round-trip all work.

Like §16a–§20a/§30a/§32a this **restores a working feature**. New regression tests
`regression/python/bytes_fromhex{,_fail}` (CORE) plus `bytes_fromhex_invalid_fail` pinning the
`ValueError` boundary (odd length / non-hex → clean error, mirroring §30's
`list_from_bytes_unsupported`). The positive test is the liveness witness (FAILED pre-fix) and covers
literal/variable receivers, whitespace, uppercase, embedded NUL, `len`/index, the empty case, and the
`.hex()` round-trip. Verified bit-for-bit against CPython; CPython sanity passes
(`scripts/check_python_tests.sh bytes_fromhex`); the focused `regression/python/(bytes|hex)` ctest
subset (19 tests) shows zero new failures. Code-reviewed (0 critical/major; the parsing loop, dispatch
predicate, and non-constant fall-through all confirmed sound). Solver-agnostic (a frontend
constant-fold, no SMT encoding). Inline `len(bytes.fromhex(""))` mis-measures via the separately-
tracked §14b inline-`len` concern, so the empty-case test binds the result to a variable first.

### 41b. Next candidate & everything else: unchanged disposition
The `bytes.hex`/`fromhex` round-trip is now complete. Remaining catalogued candidates: the §36a
bytes-**literal-argument** lowering (deferred — needs the frontend `get_literal` context fix); the
`int.from_bytes(byteorder=)` keyword form (shipped in flight as §40/PR #5663); `bytes.hex(sep)`
(shipped in flight as §39/PR #5661); `str.maketrans`/`translate` dict-table and non-constant forms
(fall through cleanly today); and multi-byte (non-ASCII) UTF-8 encode/decode. The separately-tracked
inline-`len`/strlen concern (§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`
(materialisation via `list(zip(...))` still unsupported — a larger feature), symbolic/user-function
`max`/`min(key=)`, `list.index()`-in-`try/except`, `float.hex()` (infeasible), and `str.isascii()`
(string-soundness, §5-#2). The §3 design-level blockers, §3c timeouts, §3d questionable expectation,
and the infeasible `hashlib` case all stand; the §5 priority order stands.

> **Note on numbering.** §39–§47 (PRs #5661/#5663/#5665/#5668/#5669/#5670/#5671/#5672/#5673) are in
> flight and not yet on `master`; this sweep is appended as §48. When all land, the maintainer orders
> §39 → … → §48.
> **Note on numbering.** §39–§46 (PRs #5661/#5663/#5665/#5668/#5669/#5670/#5671/#5672) are in flight
> and not yet on `master`; this sweep is appended as §47. When all land, the maintainer orders
> §39 → … → §47.
> **Note on numbering.** §39 (PR #5661, `bytes.hex(sep)`) and §40 (PR #5663, `int.from_bytes`
> `byteorder=` keyword) are in flight and not yet on `master`; this sweep is appended as §41. When all
> land, the maintainer orders §39 → §40 → §41.
## 40. 2026-06-28 re-validation (thirtieth sweep) & int.from_bytes(byteorder=) keyword form

Re-test against current `master` (tip `810d1bc2d5`). KNOWNBUG classification unchanged — §3 holds.
This sweep took the `int.from_bytes(byteorder=)` keyword form catalogued in §37b/§38b/§39b — the last
small adjacent entry in the `from_bytes` family before the deferred §36a literal-argument lowering.

### 40a. New isolated, soundly-fixable defect found & fixed
**`int.from_bytes(b, byteorder="big")` (the keyword endianness form) raised a spurious `TypeError`.**

The positional form `int.from_bytes(b, "big")` verifies (§35/§37), but the keyword form reported
`FAILED` (uncaught `TypeError: from_bytes() missing 1 required positional argument: 'big_endian'`).
Two layers conspired: the AST normalizer `_normalize_int_from_bytes_endianness`
(`preprocessor/core_visitors_mixin.py`) folded the `"big"`/`"little"` string to a bool only for the
**positional** argument (`node.args[1]`, guarded on `len(node.args) > 1`); and the operational model
(`models/int.py`) names the endianness parameter `big_endian`, whereas CPython's keyword is
`byteorder` — so a `byteorder=` keyword matched no model parameter and was reported as a missing
positional argument by `_fill_missing_args_with_defaults`.

**Fix** (`preprocessor/core_visitors_mixin.py`): extend `_normalize_int_from_bytes_endianness` to also
walk `node.keywords` — when a `byteorder` keyword is present, **rename** it to the model's `big_endian`
parameter and **fold** its constant string value to the bool the model expects (`"big" → True`,
otherwise `False`, the same rule the positional path uses). The positional branch is refactored to the
same `is_big` form (behaviour-identical: `True` only for the constant `"big"`, else `False`). The
keyword-only `signed` argument already matches the model parameter name, so it is untouched and
continues to work (`signed=True` two's-complement verified). A non-constant `byteorder` keyword folds
to little-endian — the same pre-existing limitation as the positional path, not a new regression.

Like §35a/§37a this **restores a working feature**. New regression pair
`regression/python/int_from_bytes_byteorder_kw{,_fail}` (CORE) covering keyword big/little, the
`signed=True`/`signed=False` keyword composition, the unchanged positional form, and a single byte;
the positive test is the liveness witness (uncaught-`TypeError` `FAILED` pre-fix → `SUCCESSFUL`
post-fix). All receivers are bytes **variables** because a bytes literal passed directly as the
argument is the separate, deferred §36a lowering issue. Verified bit-for-bit against CPython; CPython
sanity passes (`scripts/check_python_tests.sh from_bytes`); pylint clean on the changed file (the two
pre-existing `listcomp_counter` E1101 false-positives are unrelated); the focused
`regression/python/int_from_bytes*` ctest subset (8 tests) is 100% green. The preprocessor is
FLAIL-mangled into the binary, so it was rebuilt before testing. Solver-agnostic (an AST-normalisation
change, no SMT encoding).

### 40b. Next candidate & everything else: unchanged disposition
The `from_bytes` family is now complete except the deferred §36a bytes-**literal-argument** lowering
(needs the frontend `get_literal` context fix). Remaining catalogued candidates: `str.maketrans`/
`translate` dict-table and non-constant forms (fall through cleanly today); multi-byte (non-ASCII)
UTF-8 encode/decode; and `bytes.hex(sep)` (shipped in flight as §39/PR #5661). The separately-tracked
inline-`len`/strlen concern (§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`,
symbolic/user-function `max`/`min(key=)`, `list.index()`-in-`try/except`, `float.hex()` (infeasible),
and `str.isascii()` (string-soundness, §5-#2). The §3 design-level blockers, §3c timeouts, §3d
questionable expectation, and the infeasible `hashlib` case all stand; the §5 priority order stands.

> **Note on numbering.** §39 (PR #5661, `bytes.hex(sep[, bytes_per_sep])`) is in flight and not yet on
> `master`; this sweep is appended as §40. When both land, the maintainer orders §39 → §40.
