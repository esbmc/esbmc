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

## 42. 2026-06-28 re-validation (thirty-second sweep) & int methods on a literal receiver

Re-test against current `master` (tip `810d1bc2d5`). KNOWNBUG classification unchanged — §3 holds.
An idiom battery over the int instance methods surfaced a literal-receiver dispatch gap — the same
shape §34 fixed for `int.to_bytes`, here affecting the zero-argument `bit_length`/`bit_count`/
`conjugate` methods.

### 42a. New isolated, soundly-fixable defect found & fixed
**`(255).bit_length()` (an int method on a *constant literal* receiver) reported a spurious
`Unsupported function`.**

`x.bit_length()` on a variable verifies (the int operational model handles it), but the bare-literal
forms `(255).bit_length()`, `(7).bit_count()`, `(5).conjugate()` reported `Unsupported function
'bit_length'` → `FAILED`. A literal receiver is not classified as an int instance (only a `Name`/
`BinOp` receiver is), so the call never reached the model — the same class-resolution gap §34 found
for `int.to_bytes`.

**Fix** (`function_call/expr.cpp`): add a `handle_int_literal_method()` constant-fold, dispatched from
`get_dispatch_table()` (next to the §34 `int.to_bytes` entry) when the method is `bit_length`/
`bit_count`/`conjugate` and the receiver is a constant int literal or a unary `+`/`-` over one. It
computes the result directly — `bit_length`/`bit_count` over the magnitude (CPython ignores the sign;
the negation is done in the unsigned domain so `LLONG_MIN` cannot overflow), `conjugate` is the
identity — and returns a `long_long_int_type()` constant (the type Python int literals already use).
The predicate matches only `Constant`/`UnaryOp` receivers, so the working `Name` (variable) and
`BinOp` paths are untouched; a magnitude exceeding signed-64-bit, a non-`USub`/`UAdd` unary operator
(`~`, `not`), or any argument is declined and falls through to the model — sound, never a wrong fold.

Like §31a/§34a this **restores a working feature**. New regression pair
`regression/python/int_literal_method{,_fail}` (CORE) covering positive/zero/negative/unary-plus
literals for all three methods, arithmetic composition, and the unchanged variable form; the positive
test is the liveness witness (`Unsupported`→`FAILED` pre-fix → `SUCCESSFUL` post-fix). Verified
bit-for-bit against CPython; CPython sanity passes
(`scripts/check_python_tests.sh int_literal_method`); the focused
`regression/python/(int_literal_method|int_bit_count|int_bit_length|int_conjugate)` ctest subset is
green (the pre-existing `--ir` Z3-only `github_1964_bit_length_bignum` excepted, on this Bitwuzla-only
build). Code-reviewed (1 high + 1 medium + 1 low finding, all fixed before commit: a `~`/`not`
predicate/handler mismatch, a `[2^63, 2^64)` literal truncation, and an unguarded keyword argument).
Solver-agnostic (a frontend constant-fold, no SMT encoding).

### 42b. Next candidate & everything else: unchanged disposition
The int instance-method family now folds on literal receivers. Remaining catalogued candidates:
`str.maketrans`/`translate` dict-table and non-constant forms (fall through cleanly today); multi-byte
(non-ASCII) UTF-8 encode/decode; the §36a bytes-literal-argument lowering (deferred); and
`float.is_integer()`/`as_integer_ratio()` on literal receivers (the float analogue of this gap). The
separately-tracked inline-`len`/strlen concern (§14b/§32b/§33b) still stands. Other deferred
candidates stand: `zip()` (materialisation via `list(zip(...))` — a larger feature), symbolic/
user-function `max`/`min(key=)`, `list.index()`-in-`try/except`, `float.hex()` (infeasible), and
`str.isascii()` (string-soundness, §5-#2). The §3 design-level blockers, §3c timeouts, §3d questionable
expectation, and the infeasible `hashlib` case all stand; the §5 priority order stands.

> **Note on numbering.** §39 (PR #5661, `bytes.hex(sep)`), §40 (PR #5663, `int.from_bytes`
> `byteorder=`), and §41 (PR #5665, `bytes.fromhex()`) are in flight and not yet on `master`; this
> sweep is appended as §42. When all land, the maintainer orders §39 → §40 → §41 → §42.

---

## 49. 2026-06-28 re-validation (thirty-ninth sweep) & bytes.index()/rindex()

Re-test against current `master` (tip `810d1bc2d5`). KNOWNBUG classification unchanged — §3 holds.
This sweep continued the bytes search family from §48 with the raising variants `bytes.index`/`rindex`.
(A `bytes.replace` fold was attempted first but declined: its result is a *bytes* object and the
return-type inference `get_string_method_return_type` is receiver-type-blind — it maps `replace → str`
for both str and bytes — so the folded bytes value would be mistyped as a char array; fixing that needs
receiver-aware return-type inference, recorded as §49b. The int-returning `index`/`rindex` avoid this.)

### 49a. New isolated, soundly-fixable defect found & fixed
**`bytes.index(sub)`/`bytes.rindex(sub)` returned the wrong index for a literal bytes object.**

`bytes([1,2,3]).index(bytes([2,3]))` should be `1` (CPython); like §48's `find`/`rfind` the bytes
search is routed through the str `strncmp`/`strlen` machinery, wrong for the int-array representation.

**Fix** (`string/string_handler.cpp`): fold `bytes.index`/`bytes.rindex` when the receiver is a literal
`bytes([…])` constructor and the argument is either a literal `bytes([…])` subsequence or a single
integer byte. On a match it returns the index (a `long_long_int_type()` constant, like `find`); when the
subsequence is **absent** it returns a catchable `ValueError` via `gen_exception_raise` — the
distinguishing behaviour of `index`/`rindex` over `find`/`rfind`. As in §47/§48 the match is **purely
syntactic** (AST only), so no symbolic, branch-merged, or partially-evaluated value can reach the fold;
a str receiver, a variable/expression receiver, a `b"…"` literal, and the position-argument (2/3-arg)
forms all fall through to the existing dispatch. (`bytes` receivers are int arrays — not tuples/lists —
so the pre-existing count/index defer block does not divert them.)

Like §44a (`str.rindex`) this **restores a working feature** for the literal case. New regression pair
`regression/python/bytes_index{,_fail}` (CORE) covering index/rindex, the single-int argument, repeated
occurrences, int-result composition, embedded NUL bytes, the **catchable** not-found `ValueError` (via
`try/except`, for both index and rindex), and `str.index` coexistence; the positive test is the liveness
witness (FAILED pre-fix). Verified bit-for-bit against CPython; CPython sanity passes
(`scripts/check_python_tests.sh bytes_index`); the focused
`regression/python/(bytes_index|str_index|bytes)` ctest subset (71 tests) is 100% green — `str.index`
unaffected (the not-found case still raises catchably). Code-reviewed: confirmed **sound** (no path
admits a non-constant value; the not-found-vs-found decision and the catchable-raise idiom verified;
search math + `size_t` bounds checked; 0 critical/major). Solver-agnostic (a frontend constant-fold +
an existing exception-raise, no SMT encoding).

### 49b. Next candidate & everything else: unchanged disposition
The literal bytes affix/search methods now fold (`startswith`/`endswith` §47, `find`/`rfind` §48,
`index`/`rindex` here). Remaining catalogued candidates: the *bytes-returning* methods
(`replace`/`split`/`join`/`upper`/…) — these need **receiver-aware return-type inference**
(`get_string_method_return_type` currently maps by method name only, so `bytes.replace` mistypes as
str; newly characterised this sweep); the **symbolic** bytes affix/search methods (the str
`strncmp`/`strlen` path is unsound for the int-array representation — §47b); the list-literal receiver
method gap (§46b); `format()` width specs; `str.maketrans`/`translate` dict-table; and multi-byte
(non-ASCII) UTF-8 encode/decode. The separately-tracked inline-`len`/strlen concern (§14b/§32b/§33b)
still stands. Other deferred candidates stand: `zip()`, symbolic/user-function `max`/`min(key=)`,
## 51. 2026-06-28 re-validation (forty-first sweep) & set ^ (symmetric difference) operator

Re-test against current `master` (tip `810d1bc2d5`). KNOWNBUG classification unchanged — §3 holds.
An idiom battery over the set operators surfaced `^` (symmetric difference) as the one unmapped member.

### 51a. New isolated, soundly-fixable defect found & fixed
**The `^` operator on sets (`a ^ b`, and `a ^= b`) returned the wrong result.**

`{1,2,3} ^ {2,3,4}` should be `{1,4}` (CPython symmetric difference), but ESBMC computed the wrong
value: the set-operation dispatch in `converter_binop.cpp` mapped only `-`/`&`/`|` (difference /
intersection / union) to the set handler; `^` (`BitXor`) fell through to the generic bitwise path,
which is meaningless over the list-pointer set representation. The `symmetric_difference()` *method*
was modelled, but the operator was not wired to it.

**Fix**:
- `converter/converter_binop.cpp`: add `BitXor` to the set-operation dispatch condition (beside
  `Sub`/`BitAnd`/`BitOr`), so a set `^` is routed to `python_set::handle_operations`.
- `python_set.cpp`: add a `BitXor` case in `handle_operations` and a new
  `build_set_symmetric_difference_call(lhs, rhs)` builder computing `(lhs - rhs) ∪ (rhs - lhs)` into a
  fresh set (the two filtered halves are disjoint by construction, so no duplicates). The existing
  `symmetric_difference()` method handler is refactored to reuse the builder — a net de-duplication, no
  behaviour change. Augmented `^=` lowers through the same operator path, so it is fixed too.

The gating stays sound: the set path fires only when an operand is the list type (sets are modelled as
lists), so a genuine integer `^` is untouched, and the adjacent dict-bitwise diagnostic is neither
bypassed nor newly triggered (dicts are structs, not lists).

Like §20a (set union/intersection/difference methods) this **restores a working feature**. New
regression pair `regression/python/set_symmetric_difference_op{,_fail}` (CORE) covering the `^`
operator, the `^=` augmented form, and the unchanged `symmetric_difference()` method (variable
receiver); the positive test is the liveness witness (wrong pre-fix). Verified bit-for-bit against
CPython; CPython sanity passes (`scripts/check_python_tests.sh set_symmetric`); the focused
`regression/python/set_*` ctest subset is green (the 2 failing — `set_difference`/`set_intersection` —
are the pre-existing `--ir` Z3-only environmental set on this Bitwuzla-only build, confirmed by
re-running under `--incremental-bmc`). Code-reviewed (0 critical/major/minor; the symmetric-difference
semantics, the behaviour-preserving method refactor, the operator gating, and `^=` all confirmed). The
positive test is the C-Live liveness witness for the added `BitXor` branch (wrong→correct). A
**pre-existing, out-of-scope** gap was noted: a *set-literal receiver* on the affix/value methods
(`{1,2}.union(...)` etc.) errors with `Object "" not found` — the set analogue of the §46b list-literal
gap, general across set methods.

### 51b. Next candidate & everything else: unchanged disposition
Remaining catalogued candidates: the set-literal-receiver method gap (general, §46b-style); variadic
`math.hypot` (the gcd/lcm pattern, but float — a nested fold risks float-equality divergence, so it
needs a sum-of-squares model, deferred); the aliased/from-import `gcd`/`lcm` spellings (§50b); the
*bytes-returning* methods (`replace`/`split`/`join` — receiver-aware return-type inference, §49b); the
symbolic bytes affix/search methods (§47b); `format()`/f-string width specs; `str.maketrans`/`translate`
dict-table; and multi-byte (non-ASCII) UTF-8 encode/decode. The separately-tracked inline-`len`/strlen
concern (§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`, symbolic/user-function
## 50. 2026-06-28 re-validation (fortieth sweep) & variadic math.gcd()/lcm()

Re-test against current `master` (tip `810d1bc2d5`). KNOWNBUG classification unchanged — §3 holds.
This sweep stepped out of the bytes family to the `math` module's variadic `gcd`/`lcm`.

### 50a. New isolated, soundly-fixable defect found & fixed
**`math.gcd`/`math.lcm` returned the wrong result for any argument count other than 2.**

`math.gcd(12, 8, 6)` should be `2` (CPython 3.9+ accepts any number of integer arguments), but ESBMC
computed the wrong value: the operational model (`models/math.py`) defines `gcd(a, b)`/`lcm(a, b)` as
**binary**, so the 0-, 1-, and 3-or-more-argument forms (`gcd()`, `gcd(x)`, `gcd(a, b, c, …)`) all
mis-evaluated. The 2-argument form was correct.

**Fix** (`preprocessor/core_visitors_mixin.py`): a new AST normalizer
`_normalize_math_gcd_lcm_variadic`, called from `visit_Call` beside `_normalize_int_from_bytes_endianness`,
rewrites a variadic `math.gcd`/`math.lcm` call into nested **binary** calls that reuse the existing
model — `f()` → `f(identity, identity)`, `f(x)` → `f(x, identity)`, `f(a, b, c, …)` →
`f(f(a, b), c, …)` (a left fold). gcd's identity is `0` (`gcd(x, 0) == abs(x)`), lcm's is `1`
(`lcm(x, 1) == abs(x)`); the model's existing 2-argument handling of `0`/`1` makes these exact. Because
the rewrite produces nested binary calls, it works on **symbolic** arguments too (the model runs on
each pair), not just constants. The 2-argument form is left untouched; `*args` splat, keyword
arguments, and the `from math import gcd` / `import math as m` spellings fall through unchanged (a
coverage limitation noted in the code — those forms were already unmodelled, so the canonical
`math.gcd(...)` path this sweep fixes is a strict improvement).

Like §40a (`int.from_bytes`) this **restores a working feature**. New regression pair
`regression/python/math_gcd_lcm_variadic{,_fail}` (CORE) covering 0/1/3/4-argument gcd and lcm, the
identities, negative arguments, the unchanged 2-argument form, a symbolic argument, and the wrong-value
case; the positive test is the liveness witness (FAILED pre-fix). Verified bit-for-bit against CPython;
CPython sanity passes (`scripts/check_python_tests.sh math_gcd`); the focused `regression/python/math*`
ctest subset (187 tests) is 100% green — no regression in the 2-argument or other math paths. pylint
clean on the changed file (the two pre-existing `listcomp_counter` E1101 false-positives are unrelated).
The preprocessor is FLAIL-mangled, so the binary was rebuilt before testing. Code-reviewed (0
critical/high; the fold arithmetic, AST synthesis, and signature-defaults interaction all confirmed; the
one medium finding — aliased/from-import coverage — documented rather than expanded, and a `*args` guard
added). Solver-agnostic (an AST-normalisation change, no SMT encoding).

### 50b. Next candidate & everything else: unchanged disposition
Remaining catalogued candidates: the aliased/from-import `gcd`/`lcm` spellings (need module-alias
tracking threaded into the normalizer); the *bytes-returning* methods (`replace`/`split`/`join` — need
receiver-aware return-type inference, §49b); the **symbolic** bytes affix/search methods (§47b); the
list-literal receiver method gap (§46b); `format()` width specs; `str.maketrans`/`translate` dict-table;
and multi-byte (non-ASCII) UTF-8 encode/decode. The separately-tracked inline-`len`/strlen concern
(§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`, symbolic/user-function
`max`/`min(key=)`, `list.index()`-in-`try/except`, `float.hex()` (infeasible), and `str.isascii()`
(string-soundness, §5-#2). The §3 design-level blockers, §3c timeouts, §3d questionable expectation, and
the infeasible `hashlib` case all stand; the §5 priority order stands.

> **Note on numbering.** §39–§50 (PRs #5661–#5676) are in flight and not yet on `master`; this sweep is
> appended as §51. When all land, the maintainer orders §39 → … → §51.

---

> **Note on numbering.** §39–§49 (PRs #5661–#5675) are in flight and not yet on `master`; this sweep is
> appended as §50. When all land, the maintainer orders §39 → … → §50.

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

> **Note on numbering.** §39–§48 (PRs #5661–#5674) are in flight and not yet on `master`; this sweep is
> appended as §49. When all land, the maintainer orders §39 → … → §49.
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

---

## 68. 2026-06-29 re-validation (fifty-eighth sweep) & str % flags/width/precision

Re-test against current `master` (tip `c11d07e970`). KNOWNBUG classification unchanged — §3 holds. After
probing several complex leads (char-`join` — multi-layer string-iteration typing; closures; the known
`sorted`-of-`dict.items` tuple drop), this sweep took the printf-`%` format mini-language, a common
real-world pattern.

### 68a. New isolated, soundly-fixable defect found & fixed
**`str % args` with any flag/width/precision (e.g. `"%.2f" % x`, `"%5d"`, `"%03d"`) was rejected; the
float conversions `%f/%e/%g` were unsupported entirely.**

`py_percent_format` (`converter/converter_binop.cpp`) constant-folds a printf-style `str % args` but read
only a bare conversion char after `%`, so `%.2f`, `%5d`, `%03d`, `%-5d`, `%+d`, `%8.2f`, `%.3s` all hit
the "unsupported conversion" reject, and `%f`/`%e`/`%g` had no case at all.

**Fix**: parse the full `%[flags][width][.precision]<conv>` spec and render it. Integers format manually
(sign split off so `+`/space/precision zero-fill land correctly), floats via a two-pass `std::snprintf`
with **literal** `"%.*f"`/`"%.*e"`/`"%.*g"` formats (`*` precision — keeps `-Wformat-nonliteral`/`-Werror`
clean), strings honour width and `.precision` truncation, and `pad_format_field` applies width with
sign-aware `0`/`-` padding. Added `%f/%F/%e/%E/%g/%G` and integer precision (`%.3d → 005`). True to the
folder's fold-or-reject contract, anything not faithfully renderable still throws a clean diagnostic
rather than miscompiling.

This is a **crash/unsupported→correct-result fix** (and removes a soundness hazard — the function exists
precisely so an unsupported `%` spec is rejected, never mis-lowered). New regression pair
`percent_format_spec{,_fail}` (CORE) covering float precision/width, sign-aware zero-pad of a negative
(`%07.2f % -3.1 → -003.10`), int width/zero-pad/left/sign/precision, `%e`, string width/truncation, and
the unchanged bare conversions; the positive is the liveness witness (most errored pre-fix). Verified
bit-for-bit against CPython; CPython sanity passes (`scripts/check_python_tests.sh percent_format`); the
`percent_format`/`str_mod`/`format` ctest subset (37 tests) is 100% green. Code-reviewed: the reviewer's
three findings — the `#` flag was parsed-then-dropped (silent miscompile), integer precision parsed-then-
ignored, and a signed-overflow on a pathological digit run — were **all fixed** (`#` now rejects cleanly,
integer precision implemented, width/precision clamped); memory-safety of the two-pass snprintf and the
parse-bound alignment were confirmed clean. Solver-agnostic (a compile-time string fold).

Remaining `%` limitations (rejected, not miscompiled): the `#` alternate form, `*` dynamic width, and the
`%(name)s` mapping form — recorded as follow-ups.

### 68b. Next candidates & everything else: unchanged disposition
Priority follow-ups: (i) char-`join` — `"".join(c for c in s)` (string-iteration elements typed as
chars, not single-char strings; §67b); (ii) `%(name)s` mapping `%`-format and `%#x`/`%*d` (above); (iii)
nested-function closures (free-variable capture — `Variable 'x' is not defined`); (iv)
`sorted(dict.items())` tuple-structure drop (known multi-layer); (v) strided list-slice delete
`del a[::k]` (§66b); (vi) pointer-form bytes (params/slices) content `len`/`==` (§58a/§59a); (vii)
`round(int, n)` returns float (§63b).

---
## 70. 2026-06-29 re-validation (sixtieth sweep) & str.format() format specs

Re-test against current `master` (tip `c11d07e970`). KNOWNBUG classification unchanged — §3 holds. Probing
the `{}`-format mini-language showed f-strings already honour specs (`f"{x:.2f}"` works) but
`str.format()` did not — this sweep closes that gap.

### 70a. New isolated, soundly-fixable defect found & fixed
**`str.format()` with any format spec (`"{:.2f}".format(x)`, `"{:>5}"`, `"{:05d}"`) errored
"format() missing keyword argument".**

The replacement-field parser in `handle_string_format` (`string/string_method_handler.cpp`) treated the
entire `{...}` body as a field name, so `{:.2f}` became a lookup for a keyword literally named `:.2f`,
which failed. The spec suffix after `:` was never split off, and the spec itself was never applied — even
though the original argument values were available on the call node.

**Fix**: split each field at the first `:` into `name` and `spec`; resolve `name` to both its
stringified value and its AST node (kept in parallel `arg_nodes`/`keyword_nodes`); when a spec is
present, format the original constant value through a new `apply_format_spec` helper implementing the
`{}` mini-language `[[fill]align][sign][0][width][.precision][type]` (int `d/x/X/o/b`, float
`f/F/e/E/g/G`, string with width/precision; default alignment numbers-right / strings-left; `0` forces
`0` fill; floats via two-pass `snprintf` with literal `%.*f` formats). Anything not faithfully modelled
(a non-constant value, `#`, grouping `,`/`_`, a typeless float spec, `!r`/`.attr`/`[idx]` access)
**throws and degrades to the existing sound nondet-string fallback** rather than mis-folding — so the
change never trades a hard error for a wrong answer, only for a folded value or a sound over-approximation.

This is a **crash/unsupported→correct-result fix** that also turns the prior hard error on a *variable*
argument into a sound nondet result. New regression pair `str_format_spec{,_fail}` (CORE) covering float
precision/width, string align (`<>^`), int zero-pad/sign/precision, explicit-align-plus-`0`-flag, and
indexed/keyword specs; the positive is the liveness witness (it errored pre-fix). Verified bit-for-bit
against CPython; CPython sanity passes (`scripts/check_python_tests.sh str_format`); the broad `str`/`format`
ctest subset shows no new failures (12 pre-existing `--z3`/`--ir` environmental, stable before/after).
Code-reviewed: the reviewer's two findings — explicit-alignment-plus-`0` using the wrong fill char, and a
typeless-float spec mis-folding as `%f` — were **both fixed** (`0` now forces `0` fill; typeless float
degrades to nondet); pointer-lifetime, fallback soundness, and spec-less-field regression were confirmed
clean. Solver-agnostic (a compile-time string fold). Sibling to §68/§69 (`%`-format spec/mapping).

### 70b. Next candidates & everything else: unchanged disposition
Priority follow-ups: (i) `format()` builtin width/precision (same mini-language, the single-value entry —
`apply_format_spec` is now reusable for it); (ii) char-`join` — `"".join(c for c in s)` (string-iteration
elements typed as chars, §67b); (iii) nested-function closures (free-variable capture); (iv)
`sorted(dict.items())` tuple drop (known multi-layer); (v) strided list-slice delete `del a[::k]` (§66b);
(vi) `round(int, n)` returns float (§63b); (vii) the `{}` grouping option `,`/`_` and typeless-float
general format.

## 65. 2026-06-29 re-validation (fifty-fifth sweep) & variadic set.union/intersection/difference

Re-test against current `master` (tip `c11d07e970`). KNOWNBUG classification unchanged — §3 holds. This
sweep took the §64b lead: variadic set methods.

### 65a. New isolated, soundly-fixable defect found & fixed
**`s.union(a, b)` / `s.intersection(...)` / `s.difference(...)` with any argument count ≠ 1 raised
"<method>() takes exactly one argument".**

CPython's `set.union`/`intersection`/`difference` are variadic — `s.union(*others)` — and the
zero-argument form returns a copy. ESBMC's `handle_set_method` (`function_call/expr.cpp`) hard-required
exactly one argument, so `a.union({2}, {3})`, `a.intersection({2,3}, {3,4})`, and `a.union()` all
errored (with a variable receiver; the set-*literal*-receiver `{1}.union(...)` is the separate §51a gap).

**Fix**: mark union/intersection/difference as variadic (`symmetric_difference` stays binary), and fold
the arguments left — `s.union(a, b) == (s ∪ a) ∪ b` — reusing the existing single-argument
`build_set_method_call`, whose result is a fresh set symbol that feeds the next step. The zero-argument
form routes to a new `python_set::build_set_copy` (`self ∪ ∅`, a fresh deduplicated copy). The retained
exactly-one-argument throw still guards `symmetric_difference`, the relation methods, `update`, and
`add`/`discard`.

This is a **crash/unsupported→correct-result fix**. New regression pair `set_variadic_methods{,_fail}`
(CORE) covering a two-argument `union` and the zero-argument copy on a variable receiver (element/`len`
checks rather than full set equality to keep the proof small); the positive is the liveness witness (it
errored pre-fix). Verified bit-for-bit against CPython; CPython sanity passes
(`scripts/check_python_tests.sh set_variadic`); `set_variadic_methods{,_fail}` pass, and the other set
failures in the subset are the pre-existing `--ir`/`--z3` environmental tests (`set_intersection`,
`set_difference`) on this Bitwuzla-only build. Code-reviewed (0 critical/major; the left-fold semantics
for all three methods, the fresh-set-symbol chaining, the zero-arg copy correctness for
union/intersection/difference alike, the unchanged single-arg / symmetric_difference / add-discard
paths, and the no-aliasing of `build_set_copy` all confirmed). Solver-agnostic. **Note**: chaining 3+
arguments is correct but slow under BMC (each step materialises a set + dedup loop), so deep variadic
calls may need a higher bound; the common 2-argument form is fast.

### 65b. Next candidates & everything else: unchanged disposition
Priority follow-ups: (i) pointer-form bytes (params/slices) content `len`/`==` — propagate a `bytes`
identity (§58a/§59a); (ii) the set/list-literal-receiver method gap (`{1}.union(...)` errors with
`Object "" not found`; §46b/§51a); (iii) `sorted`/`sort` with an arithmetic/function `key` silently sorts
by natural order (§56b/§57b); (iv) `int`-literal-receiver methods (`(255).bit_length()`) unsupported
though the variable form works (§57b); (v) `round(int, n)` returns float instead of int (§63b); (vi)
`str.format` field width/spec (the format mini-language, §46/§61b).

Remaining catalogued candidates: `frozenset` (unsupported AST type); `dict |` PEP 584 union (explicitly
unsupported); variadic `math.hypot` (float-precision, §51b); the bytes-returning methods (receiver-aware
return-type inference, §49b); the symbolic bytes affix/search methods (§47b); `str.maketrans`/`translate`
dict-table; and multi-byte UTF-8 encode/decode. The separately-tracked inline-`len`/strlen concern
(§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`,
`list.index()`-in-`try/except` (ValueError not catchable), `float.hex()` (infeasible), and
`str.isascii()` (string-soundness, §5-#2). The §3 design-level blockers, §3c timeouts, §3d questionable
expectation, and the infeasible `hashlib` case all stand; the §5 priority order stands.

> **Note on numbering.** §42/§43 and §49–§64 (PRs #5668, #5669, #5675–#5687, #5689–#5691) are in flight
> and not yet on `master`; this sweep is appended as §65.
## 54. 2026-06-28 re-validation (forty-fourth sweep) & list.insert() negative index

Re-test against current `master` (tip `d761b93cf2`, with §39/§40/§41 now merged). KNOWNBUG
classification unchanged — §3 holds. This sweep took a wrong-value defect in `list.insert` with a
negative index (deliberately in the list operational model, not the contended `string_handler.cpp`).

### 54a. New isolated, soundly-fixable defect found & fixed
**`list.insert(i, x)` with a negative `i` produced the wrong list.**

`[1, 2, 3].insert(-1, 9)` should give `[1, 2, 9, 3]` (CPython — a negative index counts from the end and
clamps to `[0, len]`; insert never raises), but ESBMC appended, yielding `[1, 2, 3, 9]`. The operational
model `__ESBMC_list_insert` declared its index parameter as `size_t`, so a negative Python index was
reinterpreted as a huge unsigned value and always took the append fast-path.

**Fix** (`c2goto/library/python/list.c`): change the index parameter to `int64_t` and normalize it —
`if (index < 0) { index += (int64_t)l->size; if (index < 0) index = 0; }`, then the existing
`index >= size → append`. This mirrors the established `__ESBMC_list_pop(PyListObject*, int64_t index)`
negative-index pattern in the same file. The C++ builder is unchanged: the GOTO call sign-extends the
(signed) Python int index into the `int64_t` parameter; the only other caller (`appendleft`, which
passes a signed-64 zero) matches the new parameter type exactly.

This is a **wrong-value fix** (a false verdict on any program using a negative insert index). New
regression pair `regression/python/list_insert_negative{,_fail}` (CORE) covering `insert(-1)`/`(-2)`/
`(-3)`/`(-5)` (before-last, mid, front, clamp-to-front) and the unchanged non-negative front / middle /
beyond-end-append forms; the positive test is the liveness witness (`[1,2,3,9]` pre-fix). Verified
bit-for-bit against CPython; CPython sanity passes (`scripts/check_python_tests.sh list_insert`); the
focused `regression/python/list_insert*` ctest subset (84 tests) is 100% green and the list-mutation
subset shows no new failures (the failing ones are the pre-existing `--z3`/`--ir` environmental set on
this Bitwuzla-only build, all confirmed solver-flagged). The OM is FLAIL-mangled, so the binary was
rebuilt before testing. Code-reviewed (0 critical/high; the normalization vs CPython across all six
index regimes, the post-normalization signed/unsigned comparison safety, sign preservation through the
argument cast, and single-caller compatibility all confirmed). Solver-agnostic (an OM arithmetic
change, no SMT-encoding change).

### 54b. Next candidate & everything else: unchanged disposition
## 58. 2026-06-28 re-validation (forty-eighth sweep) & len() of a bytes slice

Re-test against current `master` (tip `81cdad042c`). KNOWNBUG classification unchanged — §3 holds. This
sweep fixed a wrong-value bug in `len()` of a bytes slice.

### 58a. New isolated, soundly-fixable defect found & fixed
**`len(b[1:3])` returned 1 instead of 2 for a bytes value `b`.**

bytes are modelled as arrays of `long_long_int` (8-byte elements; `type_handler.cpp` builds `bytes` as
`build_array(long_long_int_type(), n)`). A bytes slice `b[a:b]` was produced by `handle_range_slice`
(`python-list/list_access.cpp`), which (1) sized the result array as `slice_len + 1` and wrote a
trailing null terminator — string behaviour — and (2) the resulting value lost its "bytes" frontend
type, so `len()` routed to `strlen`. `strlen` over a wide-int array stops at the first element's zero
high bytes, so `len(bytes([1,2,3,4])[1:3])` evaluated to **1**. The slice *content* was already correct
(`s[0] == 2`, `s[1] == 3`); only the length was wrong (a false verdict on any program branching on a
bytes-slice length).

**Fix** (two files, three changes):
- `list_access.cpp`: gate the null terminator on `elem_type == char_type()`. String (char-array)
  slices keep the `slice_len + 1` size and the null write; non-char (bytes / numpy) slices are sized
  exactly `slice_len` with no phantom null element.
- `builder.cpp` (`len` routing): a Name whose followed symbol type is a non-char array now routes to
  `__ESBMC_get_object_size` (element count) instead of `strlen` — covers `s = b[a:b]; len(s)`; and an
  inline `len(b[a:b])` whose base `Name` has frontend type "bytes" routes the same way. String bases
  (var_type "str") and single-index subscripts are excluded, so the string `strlen` path is untouched.

This is a **wrong-value/soundness fix**. New regression pair `bytes_slice_len{,_fail}` (CORE) covering
the inline and variable forms, open-ended slices, embedded zero bytes (which `strlen` would have stopped
at), and element access; the positive is the liveness witness (`len(b[1:3])` was 1 pre-fix). Verified
bit-for-bit against CPython; CPython sanity passes (`scripts/check_python_tests.sh bytes_slice`); the
focused `bytes`/`slice` ctest subset (83 tests) is 100% green, and string / list / **numpy** 1-D slice
tests are unaffected (a bonus: numpy slices flow through the same path, so their `len` — previously
`slice_len + 1` — is now also corrected to `slice_len`). Code-reviewed (0 critical/high; the char-array
gating, the non-char-array routing predicate non-regression, the narrow inline predicate's
misfire/JSON-safety, and the numpy net-improvement all confirmed by an independent rebuild). Solver-
agnostic.

A **separate, still-open** facet was found and left for a follow-up: bytes-slice **equality**
(`b[1:3] == bytes([2,3])`) is still wrong — fixing sizing/`len` did not resolve it, so the comparison
has a distinct root (the slice value is not recognised as a `bytes` object by the equality path). The
durable fix for both this and the inline/`call()`-base `len` corner is to **propagate the `bytes`
frontend type through a slice** so the slice value routes through the same paths as a real bytes value;
recorded as the top bytes-slice follow-up.

### 58b. Next candidates & everything else: unchanged disposition
Priority follow-ups: (i) **bytes-slice equality / bytes type-propagation through slicing** (above);
(ii) `sorted`/`sort` with an arithmetic/function `key` silently sorts by natural order (§56b/§57b);
(iii) `int`-literal-receiver methods (`(255).bit_length()`) unsupported though the variable form works
(§57b classification gap).
## 59. 2026-06-28 re-validation (forty-ninth sweep) & bytes content equality

Re-test against current `master` (tip `c11d07e970`). KNOWNBUG classification unchanged — §3 holds. This
sweep took the §58b top follow-up (bytes-slice equality) and found its **root**: bytes equality compares
by identity, not content — affecting *all* bytes, not just slices.

### 59a. New isolated, soundly-fixable defect found & fixed
**`a == b` for two equal-content bytes variables was wrongly False.**

`a = bytes([2,3]); b = bytes([2,3]); a == b` verified as False (and `a != b` as True) — a soundness
hole. bytes are modelled as arrays of `long_long_int`; for two array operands the generic binop path
lowered `a == b` to `&a[0] == &b[0]` (confirmed via `--show-vcc`), an **address/identity** compare. Two
distinct-but-equal byte sequences therefore compared unequal. Both-inline-literal comparisons already
worked (a constant fold elsewhere); only the case with a variable operand was broken. str (strcmp) and
list (list compare) already compared by content — bytes was the outlier.

**Fix** (`converter/converter_binop.cpp`): before the generic builder, fold `==`/`!=` over two
constant-length value arrays of `long_long_int` to an **element-wise** content comparison — unequal
lengths fold to the constant verdict, length 0 folds to equal, otherwise the `AND` of `a[i] == b[i]`
(negated for `!=`). Symbolic-length operands fall through unchanged, and the unroll is capped at 4096
elements so a pathological literal cannot build a huge equality chain.

This is a **wrong-value/soundness fix** and the root cause behind the §58 slice-equality follow-up (once
PR #5684's slice sizing also lands, `b[1:3] == bytes([2,3])` follows from this fold). New regression pair
`bytes_equality{,_fail}` (CORE) covering equal/unequal content, unequal length, empty bytes, `!=`
negation, and `b"…"` literals; the positive is the liveness witness (`a == b` was False pre-fix).
Verified bit-for-bit against CPython; CPython sanity passes
(`scripts/check_python_tests.sh bytes_equality`); the focused `bytes`/`numpy` ctest subset (55 tests) is
100% green. Code-reviewed (0 correctness defects; the element-wise logic, list/tuple/str exclusion, and
the `acc`/index/size handling all confirmed). Two review points were addressed: (i) numpy integer arrays
materialised as array *values* share the `array(long_long_int)` representation and are now also compared
by content — a **strict improvement** over the prior identity compare (numpy's full elementwise-bool-
array `==` remains a separate, unimplemented feature; numpy arrays held as pointers are untouched, hence
the honest "value int array" framing); (ii) the unroll bound above. Solver-agnostic.

A known residual: a bytes **parameter** (modelled as a pointer, not an array) keeps the identity compare
— the same pointer-vs-value-array split noted for `len()` in §58a; the durable fix remains propagating a
`bytes` identity through slices/params so pointer-form bytes route through the content path too.

### 59b. Next candidates & everything else: unchanged disposition
Priority follow-ups: (i) **pointer-form bytes (params/slices) content ops** — propagate a `bytes`
identity so pointer-typed bytes get content `len`/`==` (subsumes the §58 slice-equality remainder once
#5684 lands); (ii) `sorted`/`sort` with an arithmetic/function `key` silently sorts by natural order
(§56b/§57b); (iii) `int`-literal-receiver methods (`(255).bit_length()`) unsupported though the variable
form works (§57b).
## 71. 2026-06-29 re-validation (sixty-first sweep) & list[float] param fed an int list

Re-test against current `master` (tip `c11d07e970`). KNOWNBUG classification unchanged — §3 holds. A
math-module probe surfaced `math.dist` returning ~0 for integer coordinates; root-causing it led to a
general list-read bug.

### 71a. New isolated, soundly-fixable defect found & fixed
**A `list[float]`-annotated parameter fed a list with integer elements misread those elements as float
garbage (~0).**

`math.dist([0,0],[3,4])` returned ~0 (so even `dist > 0` failed), while float coordinates worked. The
root is general, not dist-specific: ESBMC stores list elements heterogeneously (each carries a runtime
`type_id`; floats live in a global `__ESBMC_float_buf`). The subscript-read path
(`python-list/list_access.cpp`) only dispatched on the stored `type_id` for a *non-constant* index into a
*statically heterogeneous* list. A `list[float]` parameter is statically "pure float", so a constant-index
read `p[0]` took the fast path and read `__ESBMC_float_buf[float_idx]` for an element that was actually an
int — never stored in float_buf — reinterpreting the int payload as a denormal ~0. Python does not enforce
annotations, so a `list[float]` can hold ints at runtime; the static type lied.

**Fix**: widen the dispatch — `dispatch_numeric = mixed_numeric || elem_type.is_floatbv()` — so *every*
float-typed element read dispatches on the runtime `type_id` (float elements read float_buf; int payloads
are promoted `(double)*(long long*)value`), and bind the element to a temp once to avoid re-evaluating the
access three times. The dispatch is the already-validated `#5160` mixed-numeric path, reused verbatim — no
new IR.

This is a **wrong-value→correct-result fix** (a soundness-relevant miscompute, not a crash). New
regression pair `list_float_param_int{,_fail}` (CORE) covering the `list[float]` param with int and float
lists and `math.dist` with integer/float coordinates; the positive is the liveness witness for the
widened dispatch (it returned the wrong value pre-fix). Verified bit-for-bit against CPython; CPython
sanity passes (`scripts/check_python_tests.sh list_float_param_int`); the focused math/float/list ctest
subset shows no new failures (the non-environmental two, `list_extend11_{fail,nondet}`, pass when run
without parallel load — `--unwind 17` timed out only under `-j4`). Code-reviewed: correct/minimal/well-
targeted, 0 critical/high; the dispatch hash-equality, the int-payload promotion, the temp-bind lock-step,
and the `--ir` back-migration were confirmed clean. **Known limitation** (reviewer Medium, acknowledged):
a `list[float]` element whose runtime type is actually a `str`/`None`/object (an outright wrong
annotation) now routes to the int-payload branch and may raise a *spurious* bounds violation rather than
silently returning garbage — a possible false positive on wrong-annotation input, never a missed bug. The
other ~15 `extract_pyobject_value` callers (dict/set/comprehension/pop/min-max) keep the annotation-
trusting fast path; that latent inconsistency is low-risk today (overloaded numeric builtins dispatch by
argument type) and is recorded as a follow-up. C-Live for the widened dispatch branch is discharged by the
positive regression (it exercises the new path; pre-fix it miscomputed). Solver-agnostic.

### 71b. Next candidates & everything else: unchanged disposition
Priority follow-ups: (i) extend the `type_id` dispatch to the other `extract_pyobject_value` read paths
(dict/set/comprehension) for fixed-`list[float]`-signature models that iterate instead of subscripting
(above); (ii) `format()` builtin width/precision (the §70 `apply_format_spec` is reusable); (iii)
char-`join` — `"".join(c for c in s)` (§67b); (iv) nested-function closures; (v) `sorted(dict.items())`
tuple drop (known multi-layer); (vi) strided list-slice delete `del a[::k]` (§66b); (vii) `round(int, n)`
returns float (§63b).
## 67. 2026-06-29 re-validation (fifty-seventh sweep) & __str__() on built-in scalars

Re-test against current `master` (tip `c11d07e970`). KNOWNBUG classification unchanged — §3 holds. This
sweep took the §66b top follow-up — `int.__str__()` — which also unblocks the §66 join-generator-`str`
lead set aside last sweep.

### 67a. New isolated, soundly-fixable defect found & fixed
**`x.__str__()` on a built-in scalar (int/float/bool/str) raised "Unsupported '__str__'".**

`(5).__str__()`, `x.__str__()` for an int/float/bool/str variable — all rejected, while a user object's
`__str__` worked. The dunder, called explicitly, was classified as a class method on the built-in type
and looked up as an operational-model function that does not exist. This also kept
`"".join(str(x) for x in xs)` broken for int `x`: the join-generator preprocessor lowering rewrites
`str(x)` to `x.__str__()`, which then failed for ints (the §66 lead).

**Fix** (`function_call/expr.cpp`): add a method-dispatch entry — `x.__str__()` with no arguments on a
built-in scalar receiver routes to `convert_to_string(x)`, the exact lowering the `str()` builtin uses
(constant-folds, else dispatches to the `__python_*_to_str` model). The predicate matches a `Constant`
number/bool/string receiver or a `Name` whose frontend type is `int`/`float`/`bool`/`str`; a user-class
instance has its class name as the type, so it does not match and falls through to the instance-method
dispatch, preserving the user's `__str__`. Placed after the analogous `__iter__`-on-builtins entry; the
predicate is side-effect-free (IR is only emitted in the handler, after a match).

This is a **crash/unsupported→correct-result fix** that also closes the §66 join-`str(int)` follow-up.
New regression pair `str_dunder_builtin{,_fail}` (CORE) covering `__str__()` on int/float/bool/str
variables, an int literal, and `"".join(str(n) for n in [1,2,3])`; the positive is the liveness witness
(it errored pre-fix). Verified bit-for-bit against CPython; CPython sanity passes
(`scripts/check_python_tests.sh str_dunder`); the focused `str`/`join`/`genexp` ctest subset (37 tests)
is 100% green, and the user-object `recursive_join_genexp` test still passes (no regression). Code-reviewed
(0 critical/high; the `x.__str__() == str(x)` equivalence for scalars, the non-match of user objects
through `get_var_type`, the first-match-wins dispatch ordering, and the predicate null-safety/purity all
confirmed). Solver-agnostic.

### 67b. Next candidates & everything else: unchanged disposition
Priority follow-ups: (i) strided list-slice delete `del a[::k]` (§66b); (ii) pointer-form bytes
(params/slices) content `len`/`==` (§58a/§59a); (iii) `sorted`/`sort` with an arithmetic/function `key`
(§56b/§57b); (iv) `int`-literal-receiver methods (`(255).bit_length()`) (§57b); (v) `round(int, n)`
returns float instead of int (§63b); (vi) `str.format` field width/spec (§46/§61b); (vii) `join` of a
non-`str()` generator over chars (`"".join(c for c in s)` — observed FAILED in §66's probe).
## 60. 2026-06-28 re-validation (fiftieth sweep) & `in` on a dict comprehension

Re-test against current `master` (tip `c11d07e970`). KNOWNBUG classification unchanged — §3 holds. This
sweep stepped away from the bytes family to a membership bug on dict comprehensions.

### 60a. New isolated, soundly-fixable defect found & fixed
**`x in d` / `x not in d` on a dict comprehension result raised "Unsupported expression for 'in'".**

`d = {x: x*x for x in range(3)}; 2 in d` aborted with
`ERROR: Unsupported expression for 'in' operation`, even though `d[2]`, `d.get(2)`, `len(d)`, and
`2 in d.keys()` all worked on the same value. Root cause: `handle_membership_operator`
(`converter/converter_binop.cpp`) dispatched dict membership only when
`python_aggregate_kind(struct_type) == "dict"`. The aggregate-kind irep is dropped as a dict
comprehension result flows through type inference, while the `__python_dict__` struct **tag** survives —
and subscript / `len` already key off the tag via `dict_handler_->is_dict_type`. So membership was the
lone dict consumer that rejected a tag-only dict.

**Fix**: add `|| dict_handler_->is_dict_type(rhs_resolved_type)` to the dict dispatch condition, so
membership recognises a dict by its tag as well as its aggregate-kind marker — the same trust subscript
and `len` already rely on. One line; the kind check stays first (cheap), both arms route to the same
`handle_dict_membership`.

This is a **crash/unsupported→correct-result fix**. New regression pair `dictcomp_membership{,_fail}`
(CORE) covering `in`/`not in` on a dict comprehension and unchanged dict-literal membership; the positive
is the liveness witness (it errored pre-fix). Verified bit-for-bit against CPython; CPython sanity passes
(`scripts/check_python_tests.sh dictcomp_membership`); the focused dict ctest subset passes (the failures
are the pre-existing `--z3`/`--ir` environmental set on this Bitwuzla-only build). Code-reviewed (0
critical/high; the tag equivalence — `__python_dict__` is produced only by the canonical
`get_dict_struct_type`, always with `keys`/`values` components — and the non-diversion of tuple/class
structs confirmed). Solver-agnostic.

### 60b. Next candidates & everything else: unchanged disposition
Priority follow-ups: (i) pointer-form bytes (params/slices) content `len`/`==` — propagate a `bytes`
identity (§58a/§59a); (ii) `sorted`/`sort` with an arithmetic/function `key` silently sorts by natural
order (§56b/§57b); (iii) `int`-literal-receiver methods (`(255).bit_length()`) unsupported though the
variable form works (§57b). A new adjacent lead from this sweep: `x in dict(a=1)` errors with
"Projecting from non-tuple based AST" — a separate dict()-keyword-construction path.

## 69. 2026-06-29 re-validation (fifty-ninth sweep) & str % (name) mapping form

Re-test against current `master` (tip `c11d07e970`). KNOWNBUG classification unchanged — §3 holds. This
sweep took §68b's natural follow-up — the `%(name)s` mapping form of printf-`%` formatting (a sibling of
the §68 flags/width/precision work, on a separate branch).

### 69a. New isolated, soundly-fixable defect found & fixed
**`str % {dict}` mapping conversions (`"%(x)s" % {"x": "hi"}`) were rejected with
"unsupported conversion '%('".**

`py_percent_format` (`converter/converter_binop.cpp`) only handled positional conversions; a right-hand
dict was pushed as a single positional argument, so any `%(name)conv` hit the unsupported-conversion
reject.

**Fix**: `py_percent_format` gains a constant `key → value-node` mapping parameter. On encountering
`%(`, it parses the key to the matching `)`, looks it up, and formats that value (without consuming a
positional argument; a key may be referenced repeatedly); a missing key raises a clean KeyError-style
diagnostic, matching CPython. The caller builds the mapping from a `Dict` right-hand side over its
constant-string keys (last write wins, matching Python dict-literal semantics); Tuple/scalar right-hand
sides keep the unchanged positional path.

This is a **crash/unsupported→correct-result fix**. New regression pair
`percent_format_mapping{,_fail}` (CORE) covering single/multi-key mapping, repeated keys, `%%` in a
mapping string, and the unchanged positional form; the positive is the liveness witness (it errored
pre-fix). Verified bit-for-bit against CPython; CPython sanity passes
(`scripts/check_python_tests.sh percent_format_mapping`); the `percent_format`/`str_mod` ctest subset
is green and the broad `str` subset shows no new failures (the 12 are the pre-existing `--z3`/`--ir`
environmental set on this Bitwuzla-only build). Code-reviewed: the reviewer's one finding — `emplace`
keeping the *first* duplicate-key value where CPython keeps the *last* — was **fixed** (last-write-wins);
bounds/`forced`-reset/positional-non-regression were confirmed clean. Solver-agnostic (a compile-time
string fold). Composes with §68: once both land, `%(name).2f` (mapping plus a spec) folds through the
shared path.

### 69b. Next candidates & everything else: unchanged disposition
Priority follow-ups: (i) char-`join` — `"".join(c for c in s)` (string-iteration elements typed as chars,
§67b); (ii) nested-function closures (free-variable capture); (iii) `sorted(dict.items())` tuple drop
(known multi-layer); (iv) strided list-slice delete `del a[::k]` (§66b); (v) pointer-form bytes
(params/slices) content `len`/`==` (§58a/§59a); (vi) `round(int, n)` returns float (§63b); (vii) the `%`
`#` alternate form and `*` dynamic width (§68b).

## 66. 2026-06-29 re-validation (fifty-sixth sweep) & del a[lower:upper] slice deletion

Re-test against current `master` (tip `c11d07e970`). KNOWNBUG classification unchanged — §3 holds. A
probe round (a join-generator `str()` lead was investigated and **set aside** — removing the
`str(x) → x.__str__()` lowering fixes `"".join(str(int) for ...)` but regresses a user-object
`__str__` test by routing it through the slower runtime join; the real fix is `int.__str__()` support in
the method dispatch, a larger change, recorded as a follow-up). This sweep took a clean, isolated del
bug instead.

### 66a. New isolated, soundly-fixable defect found & fixed
**`del a[lower:upper]` on a list errored with "function call: argument".**

`del a[1:3]`, `del a[:2]`, `del a[2:]` all aborted, while single-index `del a[i]` worked. Root cause: the
Delete-statement handler (`converter/converter_stmt.cpp`) desugared `del a[subscript]` on a list to
`a.pop(subscript)`. For a single index that is correct, but for a `Slice` subscript it passed the Slice
AST node as `pop`'s index argument — invalid (pop wants an int) — producing the internal error.

**Fix**: in the list branch, detect a `Slice` subscript and route it to the existing slice-assignment
lowering with an empty replacement — `del a[1:3]` becomes `a[1:3] = []`, the CPython-equivalent removal,
reusing the proven `handle_slice_assignment` path. A single-index subscript still desugars to
`a.pop(i)`. A strided slice (`del a[::k]`, step ≠ 1) is rejected with a clean diagnostic rather than the
misleading assignment-flavoured ValueError the empty-slice-assign model would otherwise raise (an
extended-step delete is legal in CPython but `a[::k] = []` is not — handling strided deletion needs a
dedicated model; recorded as a follow-up).

This is a **crash/error→correct-result fix**. New regression pair `del_list_slice{,_fail}` (CORE)
covering `del a[i:j]`, open-ended `del a[:j]`/`del a[i:]`, the post-delete length/contents, and the
unchanged single-index delete; the positive is the liveness witness (it errored pre-fix). Verified
bit-for-bit against CPython; CPython sanity passes (`scripts/check_python_tests.sh del_list`); the
`regression/python/del`/`slice` ctest subset (34 tests) is 100% green. Code-reviewed (the contiguous
desugaring correctness, the empty-list value-node validity, the non-impact on dict/single-index/attribute
deletes, and the constructor-json inertness all confirmed; the reviewer's extended-step divergence was
addressed by the strided-reject guard). Solver-agnostic.

### 66b. Next candidates & everything else: unchanged disposition
Priority follow-ups: (i) `int.__str__()` / `"".join(str(int) for ...)` — support `__str__` as a method
on built-in numerics so the join-generator lowering works for ints as well as user objects (above); (ii)
strided list-slice delete `del a[::k]` (above); (iii) pointer-form bytes (params/slices) content
`len`/`==` (§58a/§59a); (iv) `sorted`/`sort` with an arithmetic/function `key` (§56b/§57b); (v)
`int`-literal-receiver methods (`(255).bit_length()`) (§57b); (vi) `round(int, n)` returns float instead
of int (§63b); (vii) `str.format` field width/spec (§46/§61b).

Remaining catalogued candidates: the set/list-literal-receiver method gap (§46b/§51a); `frozenset`
(unsupported AST type); `dict |` PEP 584 union (explicitly unsupported); variadic `math.hypot`
(float-precision, §51b); the bytes-returning methods (§49b); the symbolic bytes affix/search methods
(§47b); `str.maketrans`/`translate` dict-table; and multi-byte UTF-8 encode/decode. The separately-tracked
inline-`len`/strlen concern (§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`,
`list.index()`-in-`try/except` (ValueError not catchable), `float.hex()` (infeasible), and `str.isascii()`
(string-soundness, §5-#2). The §3 design-level blockers, §3c timeouts, §3d questionable expectation, and
the infeasible `hashlib` case all stand; the §5 priority order stands.

> **Note on numbering.** §42/§43 and §49–§67 (PRs #5668, #5669, #5675–#5687, #5689–#5694) are in flight
> and not yet on `master`; this sweep is appended as §68.
(§47b); `str.maketrans`/`translate` dict-table; `%(name)s` mapping `%`-format; and multi-byte UTF-8
(§47b); `str.maketrans`/`translate` dict-table; and multi-byte UTF-8 encode/decode. The separately-tracked
inline-`len`/strlen concern (§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`,
`list.index()`-in-`try/except` (ValueError not catchable), `float.hex()` (infeasible), and `str.isascii()`
(string-soundness, §5-#2). The §3 design-level blockers, §3c timeouts, §3d questionable expectation, and
the infeasible `hashlib` case all stand; the §5 priority order stands.

> **Note on numbering.** §42/§43 and §49–§69 (PRs #5668, #5669, #5675–#5687, #5689–#5696) are in flight
> and not yet on `master`; this sweep is appended as §70.

## 64. 2026-06-29 re-validation (fifty-fourth sweep) & max/min/sum over a dict

Re-test against current `master` (tip `c11d07e970`). KNOWNBUG classification unchanged — §3 holds. This
sweep took a wrong-value bug in the reducers over a dict.

### 64a. New isolated, soundly-fixable defect found & fixed
**`max(d)` / `min(d)` / `sum(d)` over a dict gave the wrong answer.**

In CPython, iterating a dict yields its keys, so `max({3:1, 1:2})` is the max **key** (3). ESBMC's bare
`max`/`min`/`sum` reinterpreted the dict struct as a list and produced wrong results (false verdicts),
even though `list(d)`, `sorted(d)`, and `for k in d` already iterated the keys correctly. Root cause: the
preprocessor `_maybe_rewrite_dict_to_list_call` (`preprocessor/core_visitors_mixin.py`) already rewrote
`list(d) → d.keys()` and `sorted(d) → sorted(d.keys())` for names known to be dicts, but it covered only
`list`/`sorted`; the reducers were never routed through the dict-keys list path.

**Fix**: extend that rewrite's covered builtins to `list`/`sorted`/`max`/`min`/`sum`. The existing tail
replaces only the first positional with `d.keys()`, so `sum(d, start)` and `max(d, default=…)` preserve
their extra positional/keyword arguments. A guard restricts the `max`/`min` rewrite to a **single**
positional argument, so the variadic form `max(a, b)` (max *of* the arguments) is left untouched; the
rewrite still fires only when the argument is an `ast.Name` reliably known to be a dict, so non-dict
arguments and non-`Name` expressions (`max([1,2])`, `max(f())`) are unaffected.

This is a **wrong-value/soundness fix**. New regression pair `reduce_over_dict_keys{,_fail}` (CORE)
covering `max`/`min`/`sum` over a dict and the unchanged variadic `max(3, 7)`; the positive is the
liveness witness (it was wrong pre-fix). Verified bit-for-bit against CPython; CPython sanity passes
(`scripts/check_python_tests.sh reduce_over_dict`), pylint clean (9.87/10, no new errors); the focused
`reduce_over_dict`/`min_max`/`sorted`/`dict` ctest subset (12 tests) is 100% green and a broader
list/dict/reducer sweep showed no new failures. The preprocessor is FLAIL-mangled, so the binary was
rebuilt before testing. Code-reviewed (0 critical/high; the rewrite-tail preservation of extra args, the
single-arg `max`/`min` guard excluding the variadic form, the `sum(iterable, start)` non-gating, and the
absence of any new misfire surface beyond the existing `list`/`sorted` gate all confirmed). Solver-
agnostic.

### 64b. Next candidates & everything else: unchanged disposition
Priority follow-ups: (i) pointer-form bytes (params/slices) content `len`/`==` — propagate a `bytes`
identity (§58a/§59a); (ii) `set.union`/`set.intersection` with multiple arguments (`ERROR: Object …` —
variadic set methods); (iii) `sorted`/`sort` with an arithmetic/function `key` silently sorts by natural
order (§56b/§57b); (iv) `int`-literal-receiver methods (`(255).bit_length()`) unsupported though the
variable form works (§57b); (v) `round(int, n)` returns float instead of int (§63b); (vi) `str.format`
field width/spec (the format mini-language, §46/§61b).
## 61. 2026-06-28 re-validation (fifty-first sweep) & dict() keyword/empty constructor

Re-test against current `master` (tip `c11d07e970`). KNOWNBUG classification unchanged — §3 holds. This
sweep took the §60b adjacent lead: `dict()`/`dict(a=1)` construction.

### 61a. New isolated, soundly-fixable defect found & fixed
**`dict(a=1, b=2)` and `dict()` aborted with "Projecting from non-tuple based AST".**

The keyword form `dict(a=1, b=2)` and the empty form `dict()` failed at SMT-solve time
(`smt_solver.cpp` `Projecting from non-tuple based AST`) — every downstream use (`d["a"]`, `in`, `len`)
inherited the malformed result. Root cause: the `dict()`-constructor dispatch
(`converter/converter_funcall.cpp`) only fired for `args.size() == 1` (a single positional iterable),
and `handle_dict_constructor` (`python-dict/dict_methods.cpp`) returned nil for anything else. Keyword
arguments are not positional, so `dict(a=1)` and `dict()` both have `args.size() == 0`, fell through the
guard, and were lowered by a generic path into a non-tuple struct. The positional form `dict([pairs])`
already worked.

**Fix** (two changes): widen the dispatch guard to `args.size() <= 1`, and add an `args.empty()` branch
to `handle_dict_constructor` that synthesises a `Dict` AST from the call's `keywords` — each keyword
name becomes a string-`Constant` key (carrying the call's location), the value its value node — then
hands off to `get_dict_literal`, the same lowering `{k: v}` and `dict([pairs])` already use. An empty
`dict()` yields `keys=[]/values=[]`; `dict(**other)` (a keyword with a null `arg`) returns nil so it is
declined rather than silently mis-lowered. (As with the pre-existing single-arg form, the dispatch keys
on `func.id == "dict"`, so a user-defined `dict` would be shadowed — unchanged behaviour, noted in the
commit.)

This is a **crash/unsupported→correct-result fix**. New regression pair `dict_kwargs_constructor{,_fail}`
(CORE) covering the keyword form's key access / membership / len, empty `dict()` + mutation, and the
unchanged positional form; the positive is the liveness witness (it errored pre-fix). Verified
bit-for-bit against CPython; CPython sanity passes (`scripts/check_python_tests.sh dict_kwargs`); the
focused dict-constructor ctest subset (32 tests) is 100% green and the broad dict subset shows no new
failures (the 9 are the pre-existing `--z3`/`--ir` environmental set on this Bitwuzla-only build).
Code-reviewed (0 critical/high; the synthesized string-key compatibility with `get_dict_literal`, the
`dict(**other)` decline, the empty-dict shape, and the non-regression of the positional form and type
inference all confirmed). Solver-agnostic.

### 61b. Next candidates & everything else: unchanged disposition
Priority follow-ups: (i) pointer-form bytes (params/slices) content `len`/`==` — propagate a `bytes`
identity (§58a/§59a); (ii) `dict.update(**kwargs)` (`update() takes exactly one argument` — the keyword
form is unmodelled, the constructor analogue of this sweep); (iii) `sorted`/`sort` with an
arithmetic/function `key` silently sorts by natural order (§56b/§57b); (iv) `int`-literal-receiver
methods (`(255).bit_length()`) unsupported though the variable form works (§57b).

## 63. 2026-06-29 re-validation (fifty-third sweep) & round(x, negative ndigits)

Re-test against current `master` (tip `c11d07e970`). KNOWNBUG classification unchanged — §3 holds. This
sweep stepped to a builtin: `round()` with a negative `ndigits`.

### 63a. New isolated, soundly-fixable defect found & fixed
**`round(1234, -2)` timed out (no verdict within the cap).**

`round(x, n)` with a **negative** `ndigits` literal — `round(1234, -2)`, `round(1234, -1)`,
`round(1234.5, -2)` — hung, while positive `ndigits` returned instantly. Root cause: `handle_round`
(`function_call/builtins.cpp`) constant-folds `round(<numeric literal>, <int literal>)` by reading
`ndigits_arg["value"]`, but a negative literal `-2` parses as `UnaryOp(USub, Constant(2))` — it has no
`value` field — so the fold was skipped and the call fell through to the far more expensive symbolic
round path, which times out for integer input.

**Fix**: before the `is_number_integer()` check, detect `ndigits_arg` being a `UnaryOp`/`USub` and unwrap
it to its operand, negating the extracted `n`. The rest of the fold (`round_to_ndigits_ties_even(val,
n)`) already handles negative `n` (round to the nearest power of ten, ties to even). Guarded JSON access;
a non-literal `round(x, -y)` still declines the fold and falls through.

This is a **timeout→fast-correct-result fix**. New regression pair `round_negative_ndigits{,_fail}`
(CORE) covering round-down/up, banker's rounding at the power-of-ten boundary (`1250 → 1200`,
`1350 → 1400`), negative input, float input, and the unchanged positive/zero-`ndigits` forms; the
positive is the liveness witness (it timed out pre-fix). Verified bit-for-bit against CPython; CPython
sanity passes (`scripts/check_python_tests.sh round_negative`); the `regression/python/round` ctest
subset is 100% green. Code-reviewed (0 critical/high; the `USub` unwrap producing `n = -2` not `2`, the
guarded JSON access, the variable-`ndigits` fall-through, and the unchanged positive path all confirmed).
A pre-existing note (unchanged here): the fold returns a float even for integer input — `round(int, n)`
is int in CPython — but every comparison passes via `1200.0 == 1200`; left as-is. Solver-agnostic.

### 63b. Next candidates & everything else: unchanged disposition
Priority follow-ups: (i) pointer-form bytes (params/slices) content `len`/`==` — propagate a `bytes`
identity (§58a/§59a); (ii) `round(int, n)` returns float instead of int (above, low priority); (iii)
`sorted`/`sort` with an arithmetic/function `key` silently sorts by natural order (§56b/§57b); (iv)
`int`-literal-receiver methods (`(255).bit_length()`) unsupported though the variable form works (§57b);
(v) `str.format` field width/spec (`"{:5}".format(...)` — the format mini-language, §46/§61b).

Remaining catalogued candidates: the set/list-literal-receiver method gap (§46b/§51a); `frozenset`
(unsupported AST type); `dict |` PEP 584 union (explicitly unsupported); variadic `math.hypot`
(float-precision, §51b); the bytes-returning methods (receiver-aware return-type inference, §49b); the
symbolic bytes affix/search methods (§47b); `format()`/f-string width specs; `str.maketrans`/`translate`
dict-table; and multi-byte UTF-8 encode/decode. The separately-tracked inline-`len`/strlen concern
(§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`, `list.index()`-in-`try/except`
(ValueError not catchable), `float.hex()` (infeasible), and `str.isascii()` (string-soundness, §5-#2).
The §3 design-level blockers, §3c timeouts, §3d questionable expectation, and the infeasible `hashlib`
case all stand; the §5 priority order stands.

> **Note on numbering.** §42/§43 and §49–§58 (PRs #5668, #5669, #5675–#5684) are in flight and not yet
> on `master` (§39/§40/§41/§44–§48 merged); this sweep is appended as §59.
symbolic bytes affix/search methods (§47b); `str.maketrans`/`translate` dict-table; and multi-byte UTF-8
encode/decode. The separately-tracked inline-`len`/strlen concern (§14b/§32b/§33b) still stands. Other
deferred candidates stand: `zip()`, `list.index()`-in-`try/except` (ValueError not catchable),
`float.hex()` (infeasible), and `str.isascii()` (string-soundness, §5-#2). The §3 design-level blockers,
§3c timeouts, §3d questionable expectation, and the infeasible `hashlib` case all stand; the §5 priority
order stands.

> **Note on numbering.** §42/§43 and §49–§66 (PRs #5668, #5669, #5675–#5687, #5689–#5693) are in flight
> and not yet on `master`; this sweep is appended as §67.
> **Note on numbering.** §42/§43 and §49–§65 (PRs #5668, #5669, #5675–#5687, #5689–#5692) are in flight
> and not yet on `master`; this sweep is appended as §66.
> **Note on numbering.** §42/§43 and §49–§63 (PRs #5668, #5669, #5675–#5687, #5689, #5690) are in flight
> and not yet on `master`; this sweep is appended as §64.
> **Note on numbering.** §42/§43 and §49–§62 (PRs #5668, #5669, #5675–#5687, #5689) are in flight and not
> yet on `master`; this sweep is appended as §63.
## 62. 2026-06-28 re-validation (fifty-second sweep) & dict.update() keyword form

Re-test against current `master` (tip `c11d07e970`). KNOWNBUG classification unchanged — §3 holds. This
sweep took the §61b sibling lead: `dict.update(**kwargs)` — the method analogue of §61's constructor
fix.

### 62a. New isolated, soundly-fixable defect found & fixed
**`d.update(b=2)` and `d.update()` raised "update() takes exactly one argument".**

CPython's signature is `dict.update([E], **F)` — an optional positional iterable plus any number of
keyword pairs — but `handle_dict_update` (`python-dict/dict_methods.cpp`) required `args.size() == 1`, so
the keyword form `d.update(b=2)` and the empty form `d.update()` (both zero positional args) were
rejected outright.

**Fix**: relax the guard to `args.size() > 1` (at most one positional source), and add an
`apply_keyword_args` step that assigns each keyword pair as `dict[name] = value` (the name lowered to a
string-`Constant` key, reusing the existing `handle_dict_subscript_assign` primitive — the same
synthesised-key pattern §61 and `fromkeys` use). It runs **after** the optional positional source so
`d.update(other, k=v)` matches CPython's order (keyword wins on a clash), is a no-op when there are no
keywords (so `update({…})` / `update(other_dict)` are byte-for-byte unchanged), and throws for
`d.update(**other)` (a null keyword `arg`) rather than silently dropping the updates.

This is a **crash/unsupported→correct-result fix**. New regression pair `dict_update_kwargs{,_fail}`
(CORE) covering the keyword form, keyword overwrite, empty `update()`, and the combined
positional-plus-keyword form; the positive is the liveness witness (it errored pre-fix). Verified
bit-for-bit against CPython; CPython sanity passes (`scripts/check_python_tests.sh dict_update`); the
focused `dict_update`/`dict_method` ctest subset (8 tests) is 100% green. Code-reviewed (0 critical/high;
the keyword key/value lowering equivalence to `d["b"]=2`, the positional-then-keyword order and
exactly-once application, the inert-without-keywords non-regression of both existing paths, the
synchronous `[&]` lambda capture, and the `**other` throw-vs-drop soundness all confirmed). Solver-
agnostic.

### 62b. Next candidates & everything else: unchanged disposition
Priority follow-ups: (i) pointer-form bytes (params/slices) content `len`/`==` — propagate a `bytes`
identity (§58a/§59a); (ii) `sorted`/`sort` with an arithmetic/function `key` silently sorts by natural
order (§56b/§57b); (iii) `int`-literal-receiver methods (`(255).bit_length()`) unsupported though the
variable form works (§57b).

## 57. 2026-06-28 re-validation (forty-seventh sweep) & constant string ordered comparison

Re-test against current `master` (tip `39fceef565`). KNOWNBUG classification unchanged — §3 holds. This
sweep took the §56b string-comparison-via-variable lead and narrowed it to a clean, isolated
**constant-fold soundness bug** in ordered string comparison.

### 57a. New isolated, soundly-fixable defect found & fixed
**`<=` / `>=` (and `<` / `>`) on constant strings were wrong whenever the answer differed from
inequality.**

`"abc" <= "abc"` verified as **False** (should be True), `"abd" <= "abc"` as **True** (should be
False), `"abc" >= "abc"` as False, and so on — a soundness hole (false SUCCESSFUL/FAILED on any program
branching on a constant string order). Root cause: `compare_constants_internal`
(`converter/converter_compare.cpp`) computed `bool equal` and returned
`gen_bool((op == "Eq") ? equal : !equal)` for every branch — so all four ordered operators collapsed to
`!equal`. `==`/`!=` were correct, and `<`/`>` were correct only by coincidence (strict order of unequal
operands equals `!equal`); `<=`/`>=` and the equal/greater cases were wrong.

**Fix**: replace the `(op == "Eq") ? equal : !equal` folds with a real three-way comparison. A `resolve`
helper maps an operator plus a comparison sign (`cmp < 0`/`== 0`/`> 0`) to the correct bool (and nil for
non-comparison ops); a `char_value` helper reads a constant char's **unsigned** byte
(`binary2integer(..., false)`) so high-bit characters (≥ 128) order by code point, matching Python and
the lexicographic char-array/`strcmp` paths. Char-array strings now fold via `lhs_str.compare(rhs_str)`
(lexicographic, prefix-correct: `"ab" < "abc"`); single-char and mixed char/array branches fold via the
integer char values. The type-identifier-mismatch and multi-char-vs-char branches now fold only
`Eq`/`NotEq` and return nil for ordered ops (falling through to the runtime `strcmp` path) instead of
the old arbitrary `op == "NotEq"`.

This is a **wrong-value/soundness fix**. New regression pair `str_ordered_compare{,_fail}` (CORE)
covering all six operators across equal / less / greater / prefix / ASCII-case cases; the positive is
the liveness witness (`"abc" <= "abc"` etc. were False pre-fix). The tests use constant operands only —
variable-string ordering goes through runtime `strcmp` (orthogonal, unwinding-dependent) and is not part
of this fold. Verified bit-for-bit against CPython; CPython sanity passes
(`scripts/check_python_tests.sh str_ordered`); a focused 117-test string-comparison/sort/min/max ctest
subset shows no non-environmental regression (the only failures are pre-existing `--z3`/`--ir`
solver-flagged tests on this Bitwuzla-only build). Code-reviewed (0 critical/high; the operator mapping,
lexicographic/unsigned-byte correctness, Eq/NotEq preservation, and the safe terminating fall-through of
the new nil paths all confirmed — the reviewer's unsigned-char-consistency caveat was applied as the
`char_value` unsigned read). Solver-agnostic (a frontend constant fold).

### 57b. Next candidates & everything else: unchanged disposition
Priority follow-ups still standing from §56b: (i) `sorted`/`sort` with an arithmetic/function `key`
(`key=lambda x: -x`, `key=len`) silently sorts by natural order (field-extraction keys work); (ii)
`int`-literal-receiver methods (`(255).bit_length()`) unsupported though the variable form works
(classification gap). The string-comparison-via-variable symptom from §56b is the *runtime-strcmp
unwinding* concern, distinct from the constant fold fixed here.

## 55. 2026-06-28 re-validation (forty-fifth sweep) & list.index(x, start[, end])

Re-test against current `master` (tip `39fceef565`, with §39/§40/§41/§44/§45/§46 now merged). KNOWNBUG
classification unchanged — §3 holds. This sweep took the catalogued `list.index` start/end gap (§52b),
in the list operational model (non-contended).

### 55a. New isolated, soundly-fixable defect found & fixed
**`list.index(x, start[, end])` was rejected outright.**

CPython `list.index(x, start[, end])` searches the slice `l[start:end]` and returns the absolute index
(ValueError if absent); ESBMC threw `list.index() takes exactly one argument`. The handler only ever
built the one-argument model call.

**Fix** (three layers, all list-specific):
- `c2goto/library/python/list.c`: new model `__ESBMC_list_index_range(l, item, type_id, size, int64_t
  start, int64_t end)` — normalizes `start`/`end` as CPython slice bounds (negative `+= size`, then each
  clamps to `[0, size]`), scans `[start, end)`, returns the first match, else the same
  `__ESBMC_assert(0, "ValueError…")` the existing `__ESBMC_list_index` uses. Mirrors that function.
- `python-list/list_query.cpp`: new builder `build_index_range_list_call` appending the `start`/`end`
  args (the GOTO call casts them to the `int64_t` parameters, sign-preserving — the same mechanism the
  §54 `list.insert` fix relied on).
- `function_call/expr.cpp` (`handle_list_index`): accept 1–3 args; one arg keeps the existing path; a
  missing `end` passes an `INT32_MAX` sentinel the model clamps to `size`.

This **adds a missing feature**. New regression pair `regression/python/list_index_start_end{,_fail}`
(CORE): the positive covers `index(x, start)`, `index(x, start, end)`, `start == 0`, negative start and
negative end (slice-bound normalization), and the unchanged one-argument form; the `_fail` pins the
not-found ValueError branch of the new model (`index(1, 1)` on `[1, 2, 3]`). Verified bit-for-bit
against CPython; CPython sanity passes (`scripts/check_python_tests.sh list_index`); the focused
`list_index`/`list_count` ctest subset is 100% green. The OM is FLAIL-mangled, so the binary was rebuilt
before testing. Code-reviewed (0 critical/high; the normalize-then-scan vs CPython across every bound
regime, the `INT32_MAX` sentinel soundness, sign preservation through the argument cast, OOB-safety of
the scan loop, and the catchable-exception consistency all confirmed). Solver-agnostic (an OM/frontend
change, no SMT-encoding change).

### 55b. Next candidate & everything else: unchanged disposition
Remaining catalogued candidates: the set/list-literal-receiver method gap (§46b/§51a); `frozenset`
(unsupported AST type); `dict |` PEP 584 union (explicitly unsupported); variadic `math.hypot`
(float-precision, §51b); the bytes-returning methods (receiver-aware return-type inference, §49b); the
symbolic bytes affix/search methods (§47b); `format()`/f-string width specs; `str.maketrans`/`translate`
dict-table; and multi-byte UTF-8 encode/decode. The separately-tracked inline-`len`/strlen concern
(§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`, `list.index()`-in-`try/except`
(ValueError not catchable), `float.hex()` (infeasible), and `str.isascii()` (string-soundness, §5-#2).
The §3 design-level blockers, §3c timeouts, §3d questionable expectation, and the infeasible `hashlib`
case all stand; the §5 priority order stands.

> **Note on numbering.** §42/§43 and §48–§57 (PRs #5668, #5669, #5674–#5683) are in flight and not yet
> on `master` (§39/§40/§41/§44/§45/§46/§47 merged); this sweep is appended as §58.
> **Note on numbering.** §42/§43 and §49–§59 (PRs #5668, #5669, #5675–#5685) are in flight and not yet
> on `master`; this sweep is appended as §60.
> **Note on numbering.** §42/§43 and §49–§60 (PRs #5668, #5669, #5675–#5686) are in flight and not yet
> on `master`; this sweep is appended as §61.
> **Note on numbering.** §42/§43 and §49–§61 (PRs #5668, #5669, #5675–#5687) are in flight and not yet
> on `master`; this sweep is appended as §62.
> **Note on numbering.** §42/§43 and §47–§56 (PRs #5668, #5669, #5673–#5682) are in flight and not yet
> on `master` (§39/§40/§41/§44/§45/§46 merged); this sweep is appended as §57.
(§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`, symbolic/user-function
`max`/`min(key=)`, `list.index()`-in-`try/except` (the ValueError is not catchable — frontend-wide
limitation), `float.hex()` (infeasible), and `str.isascii()` (string-soundness, §5-#2). The §3
design-level blockers, §3c timeouts, §3d questionable expectation, and the infeasible `hashlib` case all
stand; the §5 priority order stands.

> **Note on numbering.** §42/§43 and §47–§54 (PRs #5668, #5669, #5673–#5680) are in flight and not yet
> on `master` (§39/§40/§41/§44/§45/§46 merged); this sweep is appended as §55.
## 43. 2026-06-28 re-validation (thirty-third sweep) & float.is_integer() on a literal receiver

Re-test against current `master` (tip `810d1bc2d5`). KNOWNBUG classification unchanged — §3 holds.
This sweep took the float analogue named in §42b — the literal-receiver dispatch gap for
`float.is_integer()`, the float counterpart of §42's int methods.

### 43a. New isolated, soundly-fixable defect found & fixed
**`(2.0).is_integer()` (a float method on a *constant literal* receiver) reported a spurious
`Unsupported function`.**

`x.is_integer()` on a variable verifies (the float operational model `models/float.py` returns
`x == int(x)`), but the bare-literal forms `(2.0).is_integer()`, `(2.5).is_integer()` reported
`Unsupported function 'is_integer'` → `FAILED`. A literal receiver is not classified as a float
instance (only a `Name`/`BinOp` receiver is), so the call never reached the model — the same
class-resolution gap §34/§42 found for int methods.

**Fix** (`function_call/expr.cpp`): add a `handle_float_is_integer_literal()` constant-fold,
dispatched from `get_dispatch_table()` (next to the §34 `int.to_bytes` entry) when the method is
`is_integer` and the receiver is a constant float literal or a unary `+`/`-` over one. It folds to a
Python bool — `std::isfinite(d) && d == std::trunc(d)`, mirroring the float OM's `x == int(x)` and
matching CPython for integral / fractional / zero / negative / inf / nan — built via
`migrate_expr_back(gen_true_expr()/gen_false_expr())`, the same construction Python `True`/`False`
literals use (so `is True`/`is False` identity comparisons work). The predicate matches only
`Constant`/`UnaryOp(USub|UAdd)` float receivers, so the working `Name` (variable) path is untouched,
int literals (`is_number_float()` is false for ints) and `~`/`not` operators are declined, and any
argument is rejected by the handler's guard — no predicate-admits/handler-throws mismatch (the HIGH
finding the sibling §42 PR fixed).

Like §31a/§42a this **restores a working feature**. New regression pair
`regression/python/float_is_integer_literal{,_fail}` (CORE) covering integral/fractional/zero/
unary-signed literals, boolean context, and the unchanged variable form; the positive test is the
liveness witness (`Unsupported`→`FAILED` pre-fix → `SUCCESSFUL` post-fix). Verified bit-for-bit
against CPython; CPython sanity passes (`scripts/check_python_tests.sh float_is_integer_literal`); the
focused `regression/python/(float_is_integer|is_integer|numeric_tower)` ctest subset (8 tests) is
100% green — the existing `float_is_integer` variable-form tests still pass. Code-reviewed (0
critical/major; predicate/handler consistency, bool typing, and the inf/nan guard all confirmed
sound). Solver-agnostic (a frontend constant-fold, no SMT encoding).

### 43b. Next candidate & everything else: unchanged disposition
The int/float instance-method families now fold on literal receivers. Remaining catalogued
candidates: `float.as_integer_ratio()` on a literal (returns a tuple — needs the tuple-construction
path, larger); `str.maketrans`/`translate` dict-table and non-constant forms (fall through cleanly
today); multi-byte (non-ASCII) UTF-8 encode/decode; and the §36a bytes-literal-argument lowering
(deferred). The separately-tracked inline-`len`/strlen concern (§14b/§32b/§33b) still stands. Other
deferred candidates stand: `zip()` (materialisation via `list(zip(...))` — a larger feature),
symbolic/user-function `max`/`min(key=)`, `list.index()`-in-`try/except`, `float.hex()` (infeasible),
and `str.isascii()` (string-soundness, §5-#2). The §3 design-level blockers, §3c timeouts, §3d
questionable expectation, and the infeasible `hashlib` case all stand; the §5 priority order stands.

> **Note on numbering.** §39 (PR #5661, `bytes.hex(sep)`), §40 (PR #5663, `int.from_bytes`
> `byteorder=`), §41 (PR #5665, `bytes.fromhex()`), and §42 (PR #5668, int methods on a literal
> receiver) are in flight and not yet on `master`; this sweep is appended as §43. When all land, the
> maintainer orders §39 → §40 → §41 → §42 → §43.
---

## 52. 2026-06-28 re-validation (forty-second sweep) & abs() of a bool (crash fix)

Re-test against current `master` (tip `ff63b29e57`, with §40/§41 now merged). KNOWNBUG classification
unchanged — §3 holds. An idiom battery surfaced a **crash** (not a wrong verdict) on `abs()` of a
boolean.

### 52a. New isolated, soundly-fixable defect found & fixed
**`abs(True)` crashed ESBMC with a solver sort assertion.**

`bool` is an int subclass in Python (`abs(True) == 1`, `abs(False) == 0`), but `abs()` over a bool
operand built an `abs` node of **bool** type whose lowered `>= 0` comparison mixed a bool sort and a
signed-bitvector sort, tripping `Assertion failed: is_signedbv_type(...)` in `smt_solver.cpp` — a hard
abort under asserts-on builds (CI DebugOpt, and the local RelWithDebInfo here), not a recoverable
verdict.

**Fix** (`function_call/builtins.cpp`): in `handle_abs`'s `build_abs` lambda, cast a bool operand to
the Python int type (`build_typecast(operand, get_typet("int", 0))`) before building the abs node, so
the `>= 0` comparison is over two signed-bitvector sorts. bools are non-negative, so `abs` is identity
on the cast value (0/1), matching CPython. The cast sits at the top of `build_abs`, before the
`__abs__` dunder dispatch and the complex check — both safe, since `bool` is final (no custom
`__abs__`) and never complex. Every bool-producing operand — literal, variable, and boolean expression
(`abs(2 > 1)`) — routes through `build_abs`, so the one cast covers them all; the int/float/complex and
`__abs__` paths are untouched (`is_bool()` matches only a Python bool).

This is a **crash→correct-result** fix (§5-item-5 class, but resolved rather than merely diagnosed).
New regression pair `regression/python/abs_bool{,_fail}` (CORE) covering `abs(True)`/`abs(False)`, a
bool variable, boolean expressions, int composition, and the unchanged int/float cases; the positive
test is the liveness witness — confirmed to **abort with the solver assertion pre-fix** and verify
SUCCESSFUL post-fix (the durable crash regression). Verified bit-for-bit against CPython; CPython
sanity passes (`scripts/check_python_tests.sh abs_bool`); the `regression/python/abs*` ctest subset
(4 tests) is 100% green — the existing `abs`/`abs-fail` int/float tests still pass. Code-reviewed (0
critical/high; correctness, the `is_bool()` non-regression, the pre-dunder placement, and full operand
coverage all confirmed; the added branch's C-Live witness is the new positive test). Solver-agnostic
(a frontend type cast, no SMT-encoding change beyond removing the malformed sort).

A sibling crash was characterised and left for a follow-up: `x = divmod(a, b); x == (q, r)` aborts with
a **tuple-sort** assertion (`smt_tuple_node_ast.cpp`) — the divmod result stored in a single variable
mis-types against a tuple literal; distinct from this abs fix.

### 52b. Next candidate & everything else: unchanged disposition
Remaining catalogued candidates: the `divmod`-to-variable tuple-sort crash (above); `list.index(x,
start[, end])` position arguments (unsupported — `a.index(2, 2)` errors; reuses slice+index or a range
OM); the set/list-literal-receiver method gap (§46b/§51a); `frozenset` (unsupported AST type); `dict |`
PEP 584 union (explicitly unsupported); variadic `math.hypot` (float-precision, §51b); the
bytes-returning methods (receiver-aware return-type inference, §49b); the symbolic bytes affix/search
methods (§47b); `format()`/f-string width specs; `str.maketrans`/`translate` dict-table; and multi-byte
UTF-8 encode/decode. The separately-tracked inline-`len`/strlen concern (§14b/§32b/§33b) still stands.
Other deferred candidates stand: `zip()`, symbolic/user-function `max`/`min(key=)`,
## 53. 2026-06-28 re-validation (forty-third sweep) & divmod() tuple-sort crash

Re-test against current `master` (tip `ff63b29e57`). KNOWNBUG classification unchanged — §3 holds. This
sweep took the `divmod`-to-variable tuple-sort crash characterised in §52b.

### 53a. New isolated, soundly-fixable defect found & fixed
**`x = divmod(a, b); x == (q, r)` crashed ESBMC with a tuple-sort assertion.**

`divmod` returns a 2-tuple; storing it in a variable and comparing to a tuple literal aborted with
`Assertion failed: sort->get_tuple_type() == other->sort->get_tuple_type()`
(`smt_tuple_node_ast.cpp`). Root cause: `handle_divmod` built its result tuple struct with a
**hard-coded** tag `"tag-tuple_divmod"`, whereas a tuple literal `(q, r)` gets a **content-based** tag
from `tuple_handler::build_tuple_tag` (`"tag-tuple_<elemtype>_<elemtype>"`). The two tuples therefore
had different SMT sorts, and the equality comparison tripped the sort assertion. (The inline form
`divmod(a, b) == (q, r)` and the unpack form `q, r = divmod(...)` did not crash — only the
store-then-compare path reached the cross-sort tuple EQ.)

**Fix** (`python_math.cpp`): replace all three manual tuple-struct constructions in `handle_divmod`
(the float-constant, int-constant, and symbolic paths) with
`converter.get_tuple_handler().create_tuple_struct_type({result_type, result_type})` — the same shared
builder a tuple literal uses, which sets the content-based tag, the `element_0`/`element_1` components,
and the python-aggregate-kind. The divmod result is now sort-compatible with a literal of the same
element types (the elements already used `result_type`, which for the int path is the dividend's int
type — the same type an int literal gets), so the comparison no longer mismatches sorts. A net
de-duplication (−25 lines).

This is a **crash→correct-result** fix (§5-item-5 class, resolved). New regression pair
`regression/python/divmod_tuple_eq{,_fail}` (CORE) covering store-then-compare, negative and float
divmod, and the unchanged index / unpack / inline forms; the positive test is the liveness witness —
confirmed to **abort with the tuple-sort assertion pre-fix** and verify SUCCESSFUL post-fix (the durable
crash regression). Verified bit-for-bit against CPython; CPython sanity passes
(`scripts/check_python_tests.sh divmod`); the `regression/python/divmod` (17) and
`regression/python/tuple` (38) ctest subsets are 100% green — `divmod` indexing/unpacking and all tuple
behaviour unchanged. Code-review was started; the change is a minimal drop-in (only the struct tag
changes, to the content-based one the literal uses; `create_tuple_struct_type` sets the identical
components/kind), empirically confirmed across the float/int/symbolic paths. Solver-agnostic (a frontend
struct-type change, no SMT-encoding change beyond removing the malformed sort).

### 53b. Next candidate & everything else: unchanged disposition
Remaining catalogued candidates: `list.index(x, start[, end])` position arguments (unsupported, §52b);
the set/list-literal-receiver method gap (§46b/§51a); `frozenset` (unsupported AST type); `dict |` PEP
584 union (explicitly unsupported); variadic `math.hypot` (float-precision, §51b); the bytes-returning
methods (receiver-aware return-type inference, §49b); the symbolic bytes affix/search methods (§47b);
`format()`/f-string width specs; `str.maketrans`/`translate` dict-table; and multi-byte UTF-8
encode/decode. The separately-tracked inline-`len`/strlen concern (§14b/§32b/§33b) still stands. Other
deferred candidates stand: `zip()`, symbolic/user-function `max`/`min(key=)`,
`list.index()`-in-`try/except`, `float.hex()` (infeasible), and `str.isascii()` (string-soundness,
§5-#2). The §3 design-level blockers, §3c timeouts, §3d questionable expectation, and the infeasible
`hashlib` case all stand; the §5 priority order stands.

> **Note on numbering.** §42–§53 (PRs #5668–#5679) are in flight and not yet on `master` (§39/§40/§41
> merged); this sweep is appended as §54.
> **Note on numbering.** §39 and §42–§51 (PRs #5661, #5668–#5677) are in flight and not yet on
> `master` (§40/§41 merged this sweep); this sweep is appended as §52.
> **Note on numbering.** §39 and §42–§52 (PRs #5661, #5668–#5678) are in flight and not yet on `master`
> (§40/§41 merged); this sweep is appended as §53.
---

## 56. 2026-06-28 re-validation (forty-sixth sweep) & min()/max() non-numeric multi-arg crash

Re-test against current `master` (tip `39fceef565`). KNOWNBUG classification unchanged — §3 holds. This
sweep took a **crash** in `min()`/`max()` over multiple non-numeric direct arguments. The probing this
round surfaced two larger, complex defects first (recorded in §56b as priority follow-ups): `sorted`/
`sort` with an *arithmetic* `key` (e.g. `key=lambda x: -x`) silently sorts by natural order — field-
extraction keys like `key=lambda p: p[1]` work (github_4364/lambda19), arithmetic keys do not — and a
string-comparison-via-variable gap (`m = a if a > b else b` over strings is wrong while `"a" < "b"` is
right). Both need real feature work; this sweep fixed the clean, isolated crash instead.

### 56a. New isolated, soundly-fixable defect found & fixed
**`max(a, b)` / `min(a, b)` over multiple non-numeric arguments crashed the SMT backend.**

`max("apple", "banana")`, `max((1, 2), (1, 3))`, and `max([1, 2], [3])` all aborted with
`ERROR: compute_pointer_offset, unexpected irep`. `handle_min_max`'s N≥2-argument path builds a
comparison chain that lowers to arithmetic `</>` (`greaterthan2tc`/`lessthan2tc`); over string
(char-array/pointer), tuple (struct), or list operands that becomes a pointer/struct comparison the
backend rejects in `compute_pointer_offset` — or, for some inputs, silently yields a wrong result (a
soundness hole, since a false SUCCESSFUL is possible).

**Fix** (`function_call/builtins.cpp`): after materialising the argument exprs and before the
type-promotion loop, reject any operand whose followed type is not numeric/bool
(`signedbv`/`unsignedbv`/`floatbv`/`fixedbv`/`bool`) with a clean
`std::runtime_error` — "min()/max() with multiple non-numeric arguments is not supported; pass a single
iterable instead, e.g. max([...])". This converts the crash (and the silent wrong results) into an
honest diagnostic and points at the **single-iterable** form (`max([...])`), which the model already
handles lexicographically and is unaffected. Numeric multi-arg `max`/`min` and the single-iterable/tuple
paths are untouched (the guard sits on the N≥2 direct-args path only; single-iterable forms never reach
it).

This is a **crash→clean-diagnostic / soundness** fix (§5-item-5 class). New regression pair: the
positive `min_max_multi_args` (CORE, SUCCESSFUL) pins numeric multi-arg `max`/`min` *and* the supported
single-iterable string `max([...])`/`min([...])`; `min_max_nonnumeric_args` (CORE) pins the new
`^ERROR: max() with multiple non-numeric arguments…` diagnostic (the `bytearray_unsupported` convention
— valid CPython, ESBMC-rejected). Verified against CPython; CPython sanity passes
(`scripts/check_python_tests.sh min_max`); the focused `min_max`/sort-key ctest subset (12 tests incl.
`min_max_key`, `github_4364`, `github_3765`) is 100% green — the existing `key=`-sort tests are
unaffected. Code-reviewed (0 critical/high; the predicate matches exactly what the downstream promotion
loop can compare, the N≥2-only scoping with single-list/tuple bypass, the error mechanism, and the
absence of false rejections for numeric variables/calls all confirmed; one pre-existing non-blocking
`max(True, 5)` bool×int limitation noted, unchanged by this patch). Solver-agnostic.

### 56b. Next candidates & everything else: unchanged disposition
**New priority follow-ups from this sweep's probing:** (i) `sorted`/`sort` with an arithmetic/function
`key` (`key=lambda x: -x`, `key=len`) silently sorts by natural order — a soundness hole; field-
extraction keys work, so the fix is to extend key application (or reject the unsupported key shapes).
(ii) string `<`/`>` comparison over *variables* in a conditional (`a if a > b else b`) is wrong while
literal `"a" < "b"` is right. (iii) `int`-literal-receiver methods (`(255).bit_length()`) are
unsupported though the variable form works (classification gap).

Remaining catalogued candidates: the set/list-literal-receiver method gap (§46b/§51a); `frozenset`
(unsupported AST type); `dict |` PEP 584 union (explicitly unsupported); variadic `math.hypot`
(float-precision, §51b); the bytes-returning methods (receiver-aware return-type inference, §49b); the
symbolic bytes affix/search methods (§47b); `format()`/f-string width specs; `str.maketrans`/`translate`
dict-table; and multi-byte UTF-8 encode/decode. The separately-tracked inline-`len`/strlen concern
(§14b/§32b/§33b) still stands. Other deferred candidates stand: `zip()`, `list.index()`-in-`try/except`
(ValueError not catchable), `float.hex()` (infeasible), and `str.isascii()` (string-soundness, §5-#2).
The §3 design-level blockers, §3c timeouts, §3d questionable expectation, and the infeasible `hashlib`
case all stand; the §5 priority order stands.

> **Note on numbering.** §42/§43 and §47–§55 (PRs #5668, #5669, #5673–#5681) are in flight and not yet
> on `master` (§39/§40/§41/§44/§45/§46 merged); this sweep is appended as §56.

---

## 73. 2026-06-30 re-validation (sixty-third sweep) & dict `**` unpacking crash→diagnostic

Re-test against current `master` (tip `efc3a92b5f`). KNOWNBUG classification unchanged — §3 holds.
An idiom battery surfaced an isolated **SIGABRT crash** (uncaught C++ exception) on dict-literal `**`
unpacking.

### 73a. New isolated, soundly-fixable defect found & fixed
**A `**` unpack inside a dict literal crashed ESBMC with an uncaught `nlohmann::json type_error`
(PR #5718).**

`{**m}`, `{**m, "y": 2}`, and `{**a, **b}` aborted with `terminating due to uncaught exception of
type nlohmann::json::detail::type_error` (SIGABRT / core dump) during the annotation pass — as a bare
statement, a function argument, a return value, or a nested value. Python's AST serialises `**m` as a
`Dict` element with a **null key** (no AST node). Two sites dereferenced that null: the type-inference
helper `get_argument_type` (`python_annotation/annotation_conversion.inl`) read `keys[0]["_type"]`
when the first element was an unpack, and the converter `create_dict_from_literal`
(`python-dict/dict_construction.cpp`) would later call `get_expr(null)`.

Dict `**` unpacking is not modelled (the same status as dict union `|`, §15). **Fix** — a
crash→clean-diagnostic robustness change in two layers: (1) `get_argument_type` returns the safe
default `"Any"` for a non-object arg (`!arg.is_object()`), so the annotation pass no longer
dereferences the null key; (2) `create_dict_from_literal` scans the key array for a null entry and
throws a clean `std::runtime_error` reported as
`ERROR: dict unpacking ({**d}) is not supported` before any null deref. A literal `None` key
(`{None: 1}`) is unaffected — it serialises as a `Constant(None)` node (an object), not a null, so
`is_null()` distinguishes the two exactly (verified: `{None: 1, "b": 2}` still verifies SUCCESSFUL).

This is the §5-item-5 robustness category (crash → clean diagnostic) and, like #5042/§15, **does not
flip a KNOWNBUG** — it removes a crash on an unmodelled feature. New regression test (CORE)
`regression/python/dict_unpack_unsupported` pins the diagnostic (valid CPython, ESBMC-rejected — the
`bytearray_unsupported`/`dict_union_unsupported` convention) and is the **C-Live** liveness witness
for the added guard (it SIGABRT'd pre-fix). Verified across all `**` positions and contexts
(bare/arg/return/nested); CPython sanity passes (`scripts/check_python_tests.sh dict_unpack`); the
`regression/python/dict` ctest subset (206 tests) shows zero non-environmental failures (only the
pre-existing `--z3`/`--ir` set on this Bitwuzla-only build). Normal dict literals and comprehensions
are untouched. Solver-agnostic (a frontend guard, no SMT encoding).

Modelling dict merge proper (`{**a, **b}` = copy-then-update with right-wins precedence) is a feature
for a dedicated change, the same disposition dict union `|` carries.

### 73b. Everything else: unchanged disposition
## 72. 2026-06-30 re-validation (sixty-second sweep) & arithmetic key in sorted/min/max

Re-test against current `master` (tip `efc3a92b5f`). KNOWNBUG classification unchanged — §3 holds.
This sweep took the §56b/§57b priority follow-up: `sorted`/`sort`/`min`/`max` with an *arithmetic*
or function `key` silently sorted by natural order — a **soundness** hole (false verdicts), the
highest-value class.

### 72a. New isolated, soundly-fixable defect found & fixed
**An arithmetic/identity `key=` was silently dropped, so `sorted`/`list.sort`/`min`/`max` returned
the wrong order and could verify *false* assertions as `SUCCESSFUL` (PR #5715).**

`sorted([1, 2, 3], key=lambda v: -v)` returned `[1, 2, 3]` instead of `[3, 2, 1]`, so
`assert sorted([1, 2, 3], key=lambda v: -v) == [1, 2, 3]` verified **SUCCESSFUL** (CPython:
`[3, 2, 1]`). The same drop affected `a.sort(key=lambda v: -v)` (which the preprocessor rewrites to
`a = sorted(a, key=...)`), `max([1, -5, 3], key=lambda v: -v)`, `min(..., key=lambda v: v * v)`, and
`sorted([...], key=len)` over a list of *strings*. Only `lambda x: x[K]` (subscript over tuples) and
the bare `abs`/`len` builtins were constant-folded; every other key form fell through to a dispatch
that drops `key=`.

**Root cause** (`src/python-frontend/preprocessor/generator_mixin.py`). `_lower_sorted_with_key_call`
only matched a single subscript-over-tuple key shape, and `_eval_min_max_key_values` only matched
that shape plus the `abs`/`len` builtin names. Any other key returned None → the lowerer returned
None → the regular dispatch dropped the key (the in-code note "which today drops the key= keyword").
The dropped result is in natural order, which can satisfy an incorrect assertion.

**Fix**: replace the two narrow matchers with one shared recursive evaluator, `_eval_const_key_value`
/ `_eval_key_body` (and rename the per-element helper to `_eval_const_key_values`), that constant-
folds any key built from the parameter, numeric constants, unary `+`/`-`, binary arithmetic
(`+ - * % // / **`), a constant subscript `p[K]`, and the `abs`/`len` builtins, over a constant list
literal. The key is evaluated at preprocess time using CPython semantics and the **original element
nodes are reordered** (a stable sort, first-occurrence tie-break) — the elements themselves are
untouched, so their runtime values are unchanged and the computed ordering matches CPython
bit-for-bit, independent of ESBMC's integer width (the key is never encoded as SMT). Both `sorted`
and `min`/`max` now share this evaluator; the now-redundant `_extract_min_max_key_index` helper was
removed (a net de-duplication). Non-constant/symbolic iterables keep the pre-existing behaviour (key
still dropped — out of scope for this constant-fold sweep), so no new over-abort is introduced and
the maintainers' "don't reject unreachable branches" decision in `function_call/builtins.cpp` is
respected.

**Crash guard (code-review catch).** The generalization made identity/arithmetic keys foldable over
*any* constant list, so a heterogeneous list (`sorted([1, "a"], key=lambda v: v)`) would compare
mixed `str`/`int` keys and raise an uncaught `TypeError` in the preprocessor — a crash the old narrow
matcher never reached (it folded only subscript/abs/len, which can't mix types). The comparison sites
(`sorted(...)` and `_select_min_max_index`) now bail on `TypeError` to the existing fallback, which
emits the same clean `ERROR: Type mismatch in sorted()/min() call` a keyless `sorted([1, "a"])`
already produces. No preprocessor crash.

This is a **wrong-value/soundness fix** (and a feature extension: `sorted(key=abs)` now folds too).
New regression tests (all CORE): `sorted_min_max_arith_key` is the liveness witness, covering
arithmetic/unary/identity/`%`/`len`/subscript keys across `sorted`, `list.sort`, `min`, and `max`,
plus the previously-supported `abs`/`len`/subscript forms; `sorted_min_max_arith_key_fail` pins the
previously-false-SUCCESSFUL assertion (`sorted([1,2,3], key=lambda v:-v) == [1,2,3]`), now correctly
FAILED; `sorted_mixed_key_fail` pins the graceful mixed-type diagnostic (no crash). Verified
bit-for-bit against CPython; CPython sanity passes (`scripts/check_python_tests.sh sorted_m`); a
focused `sort`/`lambda`/`min_max`/`list`/`tuple` ctest subset shows zero non-environmental regression
(the only failures are the pre-existing `--z3`/`--ir`/`--boolector` tests on this Bitwuzla-only
build). Solver-agnostic (a frontend constant fold, no SMT encoding).

Residuals left out of scope (pre-existing, narrowed but not closed by this fold): a `key=` over a
*symbolic* (non-literal) iterable, or a non-foldable key shape (a `Compare`, a user function, an
unsupported operator like `~`/`&`, or a key that legitimately evaluates to `None`), is still dropped
to a natural-order sort — applying a key to a symbolic sequence, or rejecting an unfoldable key over
a known-constant list, are separate changes on the §5 roadmap.

### 72b. Everything else: unchanged disposition
## 74. 2026-06-30 re-validation (sixty-fourth sweep) & str.isascii/isdecimal/isprintable

Re-test against current `master` (tip `efc3a92b5f`). KNOWNBUG classification unchanged — §3 holds.
A string-predicate battery found three `is*` methods that reported a spurious verdict for want of a
model.

### 74a. New isolated, soundly-fixable defect found & fixed
**`str.isascii()`, `str.isdecimal()`, and `str.isprintable()` on a constant string were unmodelled,
reporting a spurious `VERIFICATION FAILED` (PR #5720).**

`"abc".isascii()`, `"123".isdecimal()`, and `"abc".isprintable()` each hit the "Unsupported function"
→ `assert(false)` fallback, so any program calling them reported `FAILED` (a false alarm). The sibling
predicates `isalpha`/`isdigit`/`isalnum`/`isspace`/`isupper`/`islower` were already constant-folded;
these three had simply never been added.

**Fix** (`src/python-frontend/python_consteval.cpp`): add three entries to the existing `pred_all`
constant-fold helper — the same per-character mechanism the sibling predicates use — with the correct
predicate and empty-string result: `isascii` (char < 128, True on empty), `isdecimal` (`'0'..'9'`,
False on empty), `isprintable` (`std::isprint`, True on empty). Like the existing predicates this is a
byte-level model: it matches CPython exactly for ASCII (the constant-fold domain) and shares the
sibling predicates' known non-ASCII/Unicode limitation. Operands are cast to `unsigned char`, so the
ctype calls are well-defined.

This **restores working behaviour** (the three predicates now verify with exact values). New
regression pair `regression/python/str_is_predicates{,_fail}` (CORE): the positive is the liveness
witness, covering all three predicates across ASCII text, the empty string, tab, DEL (`0x7f`), and a
high byte (`0x80`); the `_fail` pins `"a\tb".isprintable()` as a real `FAILED`. Verified bit-for-bit
against CPython; CPython sanity passes (`scripts/check_python_tests.sh str_is_predicates`); the
focused `str`/`string`/`consteval` ctest subset (457 tests) shows zero non-environmental regression
(only the pre-existing `--z3`/`--ir`/`--boolector` set on this Bitwuzla-only build). Solver-agnostic
(a frontend constant fold, no SMT encoding).

`str.istitle()` remains unmodelled — it needs word-boundary (cased-run) tracking rather than a flat
per-character predicate, so it is a separate change. Variable-string receivers (`s.isascii()` for a
non-literal `s`) still route to the runtime handler, which models a different subset; extending that
is orthogonal.

### 74b. Everything else: unchanged disposition
The §3 design-level blockers, §3c policy-banned timeouts, §3d questionable expectation, and the
infeasible `hashlib` case all stand. The §5 priority order stands.

## 78. 2026-07-02 re-validation (sixty-eighth sweep) & f-string format specs

Re-test against current `master` (tip `af8495a058`). KNOWNBUG classification unchanged — §3 holds.
A 15-idiom battery found f-string format specs on variables **silently dropped** — a
**wrong-value/soundness** defect: `assert f"{x:03d}" == "7"` verified `SUCCESSFUL` (CPython renders
`"007"`), and the valid `== "007"` claim reported a spurious `FAILED`.

### 78a. New isolated, soundly-fixable defect found & fixed
**f-string format specs were dropped in two sites; unmodelled specs folded to an exact-but-wrong
unformatted value.**

Sites: (1) the consteval JoinedStr fold ignored `format_spec` (and `!r`/`!a` conversions) entirely;
(2) `apply_format_specification` (`string_handler.cpp`) only understood `[[fill]align][width]` and
whole-format `d`/`i`/`.Nf` — everything else (zero-pad `03d`, width+type `5d`, width+precision
`07.2f`, `x`, …) fell through to the unformatted `str()` value.

**Fix (both sites, review-hardened):**
- consteval declines to fold any `FormattedValue` carrying a `format_spec` or a `!r`/`!a`
  conversion (`!s` renders identically), routing to the string handler.
- `parse_format_padding` learns the `0`-before-width shorthand as a `zero_pad` flag whose implied
  alignment is resolved **by type** at the call site — CPython zero-fills numbers sign-aware (`=`)
  but strings left-aligned (`format('ab','05') == 'ab000'`; the initial single-flag version wrongly
  implied `=` for strings — caught by code review, F1).
- `'='` alignment is sign-aware (`f"{-42:05d}"` → `"-0042"`, not `"00-42"`).
- The padding branch accepts trailing `d`/`i` on integers and `s` on any value; resolves symbols to
  compile-time constants (`unary-` over a constant for negative int literals; string symbols to
  their constant arrays), so `f"{s:>5}"` and `f"{n:05d}"` fold instead of aborting.
- The float `.Nf` branch composes the parsed fill/align/width with the precision-formatted text
  (`f"{1.5:07.2f}"` → `"0001.50"`; review F2 — previously the width was silently dropped) and sees
  through `unary-` for negative float literals (IEEE sign-bit flip).
- `joinedstr_len` declines when a part carries a spec/conversion (review F4 — a `len()` fold
  bypassed the decline and returned the unpadded length).
- **Soundness keystone:** any spec/conversion still unmodelled (e.g. `x`, `,`, `!r`) now yields a
  **nondet string** (with a warning) instead of the exact-but-wrong unformatted value — a false
  claim can no longer verify `SUCCESSFUL`; comparisons against the nondet either FAIL or abort with
  a clean diagnostic (verified in both function and module contexts).

This is a **wrong-value/soundness fix** plus a feature extension. New regression pair
`regression/python/fstring_zero_pad{,_fail}` (CORE): the positive covers zero-pad/width/type-char
combos on positive/negative ints, string alignment incl. `{s:05}` → `"ab000"`, and float
width+precision incl. `-1.5` → `"-001.50"`; the `_fail` pins the previously-false-SUCCESSFUL
`f"{x:03d}" == "7"`. Code review differential-tested the fold against CPython `format()` over 5 int
values × 16 specs + strings × 8 specs (each as `==` and `!=`) — all correct. CPython sanity passes;
the `fstring`/`format`/`joined` subset (46) and `str`/`string`/`consteval` subset (1192 on-branch)
pass 100%. Solver-agnostic (frontend constant folds).

Residuals (documented, sound): specs/conversions outside the modelled set (`x`, `b`, `,`, `e`, `%`,
`!r`) render a nondet string — imprecise, never wrong; module-level comparisons against that nondet
abort with `Cannot compare non-function side effects` (a pre-existing comparison limitation,
separate work).

### 78b. Everything else: unchanged disposition

## 79. 2026-07-02 re-validation (sixty-ninth sweep) & as_integer_ratio() literal fold

Re-test against current `master` (tip `e92296d1fe`). KNOWNBUG classification unchanged — §3 holds.
This sweep took a §78-battery residual: `float.as_integer_ratio()` was undefined, and unpacking its
result **crashed conversion**. (The battery's other hits resolved themselves: `str.encode()` now
passes on current master; set-method timeouts are the §3c perf-banned class;
`str.maketrans`/`translate` needs a dict intermediate — catalogued, not taken.)

### 79a. New isolated, soundly-fixable defect found & fixed
**`as_integer_ratio()` was undefined — `assert(false)` stub on call, and a conversion crash
(`Cannot unpack empty`) when the result was tuple-unpacked.**

`(2.5).as_integer_ratio()` hit "Undefined function … replacing with assert(false)" (spurious
`FAILED` for any caller), and `n, d = v.as_integer_ratio()` aborted the whole conversion with
`ERROR: Cannot unpack empty - only tuples and arrays can be unpacked`.

**Fix** (`src/python-frontend/preprocessor/core_visitors_mixin.py`): a preprocess-time fold
`_maybe_fold_as_integer_ratio`, following the Decimal-constructor precedent — the ratio is computed
by **CPython itself** during preprocessing, so the folded `(numerator, denominator)` tuple matches
the interpreter bit-for-bit (e.g. `0.1` → `(3602879701896397, 36028797018963968)`). Scope: int and
float literals, and unary `+`/`-` over one. Declines (unchanged behaviour) on bool receivers,
non-literals, `inf`/`nan`, and ratios exceeding 2⁶³−1. Tuple contexts (compare, unpack) work
because the result is an ordinary tuple literal.

This **restores working behaviour** for literal receivers and removes the unpack crash on that
path. New regression pair `regression/python/as_integer_ratio{,_fail}` (CORE): the positive covers
dyadic/non-dyadic floats, negatives, ints, zero, and tuple unpacking (the pre-fix crash shape); the
`_fail` pins a wrong ratio as a real `FAILED`. Code review differentially tested the imported fold
against CPython over ~9,000 generated cases (uniform floats, 62-bit ints, wide-exponent floats,
negative zero, boundary 2⁶³±1, must-decline set incl. `True`/`1e300`/`1e400`/`~5`) — bit-for-bit
exact, decline predicate exact. CPython sanity passes; the
decimal/from_bytes/gcd/lcm/preprocessor-adjacent subset (36) passes 100%; the `tuple|float` subset
shows only the two pre-existing `--ir`/`--z3` environmental failures (verified identical on vanilla
master). Solver-agnostic (a preprocess-time fold).

Variable receivers (`v.as_integer_ratio()`) keep the prior unsupported behaviour (no wrong fold) —
a runtime model needs frexp-style bit manipulation in the OM, a separate change.

### 79b. Everything else: unchanged disposition
## 75. 2026-07-02 re-validation (sixty-fifth sweep) — graph-quixbug class-instance signatures shifted post-heap-lifetime (characterized, deferred)

Re-test against current `master` (tip `c177d64d51`), focused on the class-graph cluster that the
object-model design note (`docs/python-object-model-design-2026-06-12.md`) tracks as its Stage 2.
The design note's Stage 1 (heap-allocated instances, PR #5339) and the Stage 2 escape/lifetime work
(GC lifetime for escaping instances #5424; `reverse_linked_list` #5438) have all **landed**, and
`reverse_linked_list` has flipped KNOWNBUG→CORE. The three remaining graph quixbugs the note names
(`detect_cycle`, `depth_first_search`, `topological_ordering`) stay KNOWNBUG — but their failure
signatures have **shifted off the crashes recorded in §3b/§6a**, which narrows the residual and is
worth pinning before the next fix cycle.

### 75a. Signature shift (characterized, no point fix)
- **`detect_cycle`** (§3b: `ERROR: Cannot resolve nested attribute: successor`; §6a: the
  short-circuit `is None` SMT abort) → now a **clean but spurious** `dereference failure: NULL
  pointer` (CWE-476) at `main.py:11` (`hare = hare.successor.successor`) — a false alarm, not a
  crash. GOTO evidence: the guard `if hare is None or hare.successor is None: return False` lowers
  to `IF !(ISNONE(hare, 0) || ISNONE(hare->successor, 0)) THEN GOTO 2`, which is a structurally
  correct short-circuit, yet symex still reaches line 11 with `hare->successor == NULL`. So the
  residual is precisely that **the null branch past an `is None` guard is not pruned for a
  heap-allocated `Class*` field whose value is `None` (a null `Class*`)** — i.e. `ISNONE` on a
  null-valued instance attribute is not forcing the guarded early return. This is a strictly
  smaller, better-localised problem than the pre-heap-lifetime "cannot resolve attribute" abort.
- **`depth_first_search`** (§3b: internal `ABORT: is_array_type(internal_deref_items...)`) → **no
  longer aborts**; it now exercises the recursion (`search_from` self-call + `node in nodesvisited`
  set membership + the `any(...)` generator) and does not converge under `--incremental-bmc` within
  a 120 s budget. Crash removed; residual is now traversal/unwinding, not a deref-model abort.
- **`topological_ordering`** (§3b/§3d: `SEGFAULT`) → **no longer segfaults**; it now drives the
  graph-traversal unwinder (`main.py:4` loop) without converging in-budget. Crash removed.

**Disposition: characterized, deferred** (like §36) — no isolated point fix is shipped this sweep.
The `detect_cycle` residual is the concrete next task: root-cause why `ISNONE(<null Class*>, 0)`
does not prune the guarded branch (candidates: the `is None`/`ISNONE` lowering vs the null-`Class*`
representation in `python_converter`'s `BoolOp`/`Compare` handling, or the entry harness feeding a
nondet receiver into `detect_cycle`). That is a focused frontend investigation + a `Class*`
null-guard fix with a pass/fail regression pair, dual-solver (Bitwuzla+Z3) and Mode C on the changed
branch — not a same-sweep edit.

**Branch hygiene note.** The long-lived local `feat/python-object-heap-lifetime` branch is now
**badly stale** — ~78 k lines diverged from current `master`, and it *deletes* unrelated files that
`master` has since added. Its useful Stage-1/Stage-2 content already reached `master` through the
focused PRs above (#5339/#5424/#5438, plus #5651 return-by-reference for #3067); the branch must
**not** be merged. New work on this cluster should branch fresh from `master`.

### 75b. Everything else: unchanged disposition
The §3 design-level blockers, §3c policy-banned timeouts, §3d questionable expectation, and the
infeasible `hashlib` case all stand. The §5 priority order stands; the object-model design note's
Stage 2 remains the home for the `detect_cycle` null-guard follow-up, with Stage 3 (#3067
`ClassHandler`/MRO refactor) still separable and unstarted.
## 76. 2026-07-02 re-validation (sixty-sixth sweep) & str.title() digit word-boundary

Re-test against current `master` (tip `af8495a058`). KNOWNBUG classification unchanged — §3 holds.
The §75 code review surfaced a **wrong-value/soundness** defect in the `title()` constant fold — the
highest-value class — and fixing it exposed a second copy of the same bug in a different fold site.

### 76a. New isolated, soundly-fixable defect found & fixed
**`str.title()` on a constant string used alnum-based word boundaries, so digit-adjacent letters
folded wrong and a false assertion could verify `SUCCESSFUL`.**

`"3d movie".title()` folded to `"3d Movie"` instead of CPython's `"3D Movie"`, and `"a1a".title()`
to `"A1a"` instead of `"A1A"` — so `assert "3d movie".title() == "3d Movie"` verified **SUCCESSFUL**
(a false claim) and the valid `assert "3d movie".title() == "3D Movie"` reported a spurious
`FAILED`. CPython (`Objects/unicodeobject.c` `do_title`) titlecases a letter iff the *previous*
character is **uncased**; digits are uncased, so they *end* a word. The old fold used
`new_word = !isalnum(c)`, treating digits as word-internal.

**Two fold sites had the same bug** — `python_consteval.cpp` (the `title` fold) and
`string_method_handler.cpp` `handle_string_title` (the constant-receiver path). They cover
*different* routing contexts (the initial single-site fix flipped one battery while the identical
top-level claim still folded wrong through the other handler — caught in this sweep's validation),
so both were switched to the same `prev_cased` state machine. Notably the **runtime OM**
`__python_str_title` (`src/c2goto/library/python/string.c`) already implemented the correct
cased-boundary rule, and the `python_str_title_runtime_model` test explicitly documented the
consteval divergence as a known pre-existing gap ("digits don't break words; that pre-existing
divergence is out of scope here") — this sweep closes exactly that documented divergence, and the
stale comment is updated. All three implementations now agree.

This is a **wrong-value/soundness fix**. New regression pair
`regression/python/str_title_digit_boundary{,_fail}` (CORE): the positive is the liveness witness
(digit-boundary cases `"3d movie"`/`"a1a"`/`"x2y3z"`/`"123abc"` plus unchanged non-digit boundary
behaviour, apostrophes, hyphens, all-caps, empty); the `_fail` pins the previously-false-SUCCESSFUL
`assert "3d movie".title() == "3d Movie"`, now correctly `FAILED`. Verified bit-for-bit against
CPython; CPython sanity passes (`scripts/check_python_tests.sh str_title_digit`); the `title` ctest
subset (12 tests) and the focused `str`/`string`/`consteval` subset (1194 tests) pass 100%.
Solver-agnostic (frontend constant folds, no SMT encoding).

### 76b. Everything else: unchanged disposition
## 75. 2026-07-02 re-validation (sixty-fifth sweep) & str.istitle()

Re-test against current `master` (tip `af8495a058`). KNOWNBUG classification unchanged — §3 holds.
This sweep took the §74a named follow-up: `str.istitle()` was the one remaining unmodelled `is*`
string predicate.

### 75a. New isolated, soundly-fixable defect found & fixed
**`str.istitle()` on a constant string was unmodelled, reporting a spurious `VERIFICATION FAILED`.**

`"Hello World".istitle()` hit the "Unsupported function" → `assert(false)` fallback, so any program
calling it reported `FAILED` (a false alarm). §74 deferred it because it needs word-boundary (cased-
run) tracking rather than the flat per-character `pred_all` mechanism.

**Fix** (`src/python-frontend/python_consteval.cpp`): add an `istitle` entry beside the existing
`isupper`/`islower` stateful block, implementing CPython's exact state machine
(`Objects/unicodeobject.c`): an uppercase letter may only follow an *uncased* character, a lowercase
letter may only follow a *cased* one, and at least one cased character must be present (empty string
→ `False`). Like the sibling predicates this is a byte-level ASCII model (operands cast to
`unsigned char` before the ctype calls, so they are well-defined) and shares their known
non-ASCII/Unicode limitation. The `title()` *transformer* was already modelled; this closes the
predicate side.

This **restores working behaviour** (the predicate now verifies with exact values). New regression
pair `regression/python/str_istitle{,_fail}` (CORE): the positive is the liveness witness for every
added branch, covering titlecased text, single letters, apostrophe boundaries (`"It'S A Dog"` True vs
`"It's A Dog"` False), double spaces, hyphens, digit-led words (`"3D Movie"` True vs `"3d Movie"`
False), the empty string, all-caps, and all-lower; the `_fail` pins `"Hello world".istitle()` as a
real `FAILED`. Verified bit-for-bit against CPython on all 13 cases; CPython sanity passes
(`scripts/check_python_tests.sh str_istitle`); the focused `str`/`string`/`consteval` ctest subset
(1194 tests) passes 100% with zero failures. Solver-agnostic (a frontend constant fold, no SMT
encoding).

Variable-string receivers (`s.istitle()` for a non-literal `s`) still route to the runtime handler,
the same disposition the §74 predicates carry — extending the runtime string model is orthogonal.

### 75b. Everything else: unchanged disposition
## 77. 2026-07-02 re-validation (sixty-seventh sweep) & str.center() odd-padding split

Re-test against current `master` (tip `af8495a058`). KNOWNBUG classification unchanged — §3 holds.
A 15-idiom battery (padding/formatting/numeric-base/dict-mutation idioms) found one CPython
divergence: `str.center()` split odd padding on the wrong side — a **wrong-value/soundness** defect
(the battery's deliberately-wrong oracle verified `SUCCESSFUL`).

### 77a. New isolated, soundly-fixable defect found & fixed
**`str.center()` put the extra fill char on the right for odd margins, so
`assert "ab".center(7, "-") == "--ab---"` verified `SUCCESSFUL` (CPython: `"---ab--"`).**

CPython (`Objects/unicodeobject.c` `unicode_center_impl`) computes
`left = marg/2 + (marg & width & 1)` — the extra fill char goes on the **left** exactly when both
the margin and the width are odd (`"ab".center(7)` → left 3/right 2; but `"a".center(4)` → left
1/right 2, since the width is even). `handle_string_center`
(`string_method_handler.cpp`) used the naive `left = pad / 2`, so every odd-margin/odd-width call
folded to the mirror-image string: the valid assertion reported a spurious `FAILED` and the false
one verified `SUCCESSFUL`. Single site — neither `python_consteval.cpp` nor the runtime OM models
`center` (non-constant receivers/widths take the nondet fallback), and `ljust`/`rjust`/`zfill`
(one-sided pads) are unaffected (battery-confirmed).

This is a **wrong-value/soundness fix**. New regression pair
`regression/python/str_center_odd_padding{,_fail}` (CORE): the positive is the liveness witness
(odd margin+width with custom and default fill, odd margin+even width, even splits, degenerate
`width <= len`, empty receiver); the `_fail` pins the previously-false-SUCCESSFUL mirror-image
claim, now correctly `FAILED`. Code review differential-tested the patched fold against real
CPython `str.center` over 448 (length, width, fill) combinations — zero mismatches. CPython sanity
passes (`scripts/check_python_tests.sh str_center`); the `center`/`width` ctest subset (11 tests)
and the focused `str`/`string`/`consteval` subset (1194 tests) pass 100% (existing center tests use
even margins and are unaffected). Solver-agnostic (a frontend constant fold, no SMT encoding).

The rest of the battery (`ljust`/`rjust`/`zfill`/`expandtabs`/`bit_length`/3-arg `pow`/
`setdefault`/`popitem`/`enumerate(start=)`/str-mult/slice-step/`bin`/`oct`/`hex`/format-specs/
`removeprefix`/`int(s, base)`) matches CPython — no further divergence found this sweep.

### 77b. Everything else: unchanged disposition
The §3 design-level blockers, §3c policy-banned timeouts, §3d questionable expectation, and the
infeasible `hashlib` case all stand. The §5 priority order stands.
## 81. 2026-07-04 re-validation (seventy-first sweep) & format() numeric mini-spec

Re-test against current `master`. KNOWNBUG classification unchanged — §3 holds. The §80 sweep's
battery flagged the `format()` builtin as a feature gap (it rejected every spec beyond a bare
presentation type); this sweep closes it. Unlike the wrong-value sweeps this is a
restore-working-behaviour feature, implemented **fail-closed** so it cannot introduce a false
`SUCCESSFUL`.

### 81a. Feature restored: the format-spec mini-language for numeric values
**`format(value, spec)` folded only a bare base/type char; any width/alignment/precision/grouping
spec (`format(5, "03d")`, `format(3.14, ".2f")`, `format(1000000, ",")`) errored "not supported"**,
even though the `%`-operator already renders the equivalent forms.

`handle_format` (`src/python-frontend/function_call/str_conv.cpp`) now parses the CPython
format-spec grammar `[[fill]align][sign][#][0][width][grouping][.precision][type]` for a constant
numeric value and renders the modelled subset: integer types `d`/`x`/`X`/`o`/`b` with
width/zero-pad/alignment/fill/sign, `#` alternate-form prefixes, and `,`/`_` grouping; float types
`f`/`F`/`e`/`E`/`g`/`G`/`%` with precision, sign, width, zero-pad and alignment (float rendering
reuses the same `round_to_nearest_guard` + two-pass `snprintf` approach as the `%`-operator fold, so
it matches CPython's round-half-to-even).

**Fail-closed discipline.** The renderer returns `std::nullopt` for any feature or combination it
does not fully model — the repr-like default float format, `c`/`n` types, `#`/grouping on floats,
and grouping combined with zero/`=` padding (which needs the fill to participate in the grouping,
`format(1234, "08,") == "0,001,234"`). The caller then rejects the spec with a clean error, exactly
as before, so an unmodelled spec can never mis-fold into a false `SUCCESSFUL`. This preserves the
report's "unsound fix is worse than an honest error" policy.

**Validation.** A CPython differential harness enumerated **660 (value, spec) combinations** (12
ints × 37 int specs + 10 floats × 24 float specs) and confirmed every folded result is **bit-for-bit
identical to CPython** — zero mismatches. The deliberately-excluded specs were separately confirmed
to reject with a clean error (never a false `SUCCESSFUL`, including against the correct CPython
value). New regression pair `regression/python/builtin_format_numeric_spec{,_fail}` (CORE): the
positive covers width/zero-pad/sign/base/alternate-form/grouping/alignment/fill for ints and
precision/sign/width/zero-pad/`%`/exponent types for floats; the `_fail` pins the now-active
`format(5, "03d") == "5"` as a real `FAILED`. All 14 existing `format`/`fstring`/`percent`
regression tests pass unchanged (incl. the `github_4642` bignum trap and the `github_4807` nondet
fallbacks); dual-solver Bitwuzla + Z3 agree; CPython sanity passes.

### 81b. Everything else: unchanged disposition
The §3 design-level blockers, §3c policy-banned timeouts, §3d questionable expectation, and the
infeasible `hashlib` case all stand. The §5 priority order stands.
