# Plan: extend verification-harness coverage of the Python operational models

**Status:** EXECUTED — Tiers 1–3 complete (see §9); all surfaced model gaps
resolved (see §10). `torch` was subsequently harnessed (PR #6017); no model
remains uncovered.
**Date:** 2026-07-10 (proposal); 2026-07-11 (execution)
**Related:** PR #5958 (math ints / int / builtins / nondet), PR #5961 (random /
collections / gamma-remainder), PR #5963 (gamma/lgamma wrong-π model fix). This
plan continues the same effort; the per-model harness PRs are listed in §9.
**Scope:** the operational models under `src/python-frontend/models/**`. The
Python *preprocessor* (`src/python-frontend/preprocessor/**`, `parser/**`) runs
on CPython and is not ESBMC-verifiable; it is exercised indirectly by the whole
`regression/python/` corpus and is out of scope here.

---

## 1. Goal

Give every operational model an explicit **verification harness**: a small
`regression/python/` test that drives the model with non-deterministic inputs,
states its precondition (`__ESBMC_assume`, *requires*) and postcondition
(`assert`, *ensures*), and pins each asserted property to a documented clause.
Each model gets a positive harness (`^VERIFICATION SUCCESSFUL$`) and, where a
plausible-but-wrong strengthening exists, a negative `_fail` harness
(`^VERIFICATION FAILED$`) proving the check has teeth.

The models are the only ESBMC-verifiable Python in the tree: they are
FLAIL-mangled into the `esbmc` binary and executed symbolically whenever a
Python program imports the corresponding module. A harness therefore both
documents the intended contract and guards it against regression.

## 2. Established methodology (reuse verbatim)

The three merged/open PRs above fixed the pattern; new harnesses must follow it.

- **Location & naming.** `regression/python/harness_<module>_<fn>/` with
  `main.py` + `test.desc`. The `harness_` prefix groups the suite
  (`ctest -R harness_`). A negative variant is `harness_<module>_<fn>_fail/`.
- **Shape.** nondet inputs → `__ESBMC_assume` preconditions → call the model →
  `assert` postconditions, each traced to a clause (E1, E2, …) in a header
  comment. Prefer `--incremental-bmc --unwind N`: k-induction turns bounded
  symbolic inputs into a genuine proof (forward-condition convergence).
- **CPython sanity.** `scripts/check_python_tests.sh` auto-skips any file
  containing `__ESBMC`, `__VERIFIER_`, or `nondet_` (intrinsic-using tests are
  not CPython-runnable). Intrinsic-free harnesses (concrete anchors) *do* run
  under CPython and must therefore also be valid Python.
- **Build.** Editing a model requires rebuilding `esbmc` (FLAIL-mangled). Adding
  a `regression/python/` dir requires a CMake reconfigure before `ctest`
  discovers it (the suite is globbed at configure time).
- **Acceptance per harness.** (a) positive harness proves under both the
  default solver and a rerun; (b) `_fail` harness falsifies on its *intended*
  assertion (confirm via the violated-property line, not just the verdict);
  (c) the CPython sanity sweep is green; (d) runtime is well under the 120 s CI
  cap.

### Known gotchas (carried from the prior PRs)

- Chained model-method calls mis-lower (`n.conjugate().conjugate()` becomes a
  pointer compare). Use an intermediate variable.
- Two symbolic floats through a transcendental/tolerance product blow up
  (`math.isclose(x, y)` > 2 min). Keep one nondet float; use concrete anchors
  for the second operand.
- Transcendental float paths (`exp`/`pow`/`log` via libm) are heavy: prefer
  concrete anchors over nondet for `gamma`-like code.
- `nondet_list(n, nondet_bool())` elements are **not** `{0,1}`-constrained; only
  the size bound `[0, n]` is contractual.
- `bytes([...])` needs constant integers — no nondet bytes.
- Repeated model calls multiply symex loops and can blow the CI budget — cache
  results in locals.
- Fixed-length lists of nondet *elements* explore every ordering while keeping
  reduce/sort/heap loops fully unwound — the default tactic for list models.

## 3. Coverage snapshot

Covered by the harness suite so far: `math` (integer + float rounding +
gamma/remainder), `int`, `builtins`, `nondet`, `random`, `collections`.

Remaining models, with the count of *existing* example-style tests that touch
each (functional tests, not contract harnesses — so a non-zero count is not
coverage of the contract):

| Model | Existing tests | Tier | Verifiability |
|---|---|---|---|
| `heapq` | 8 | 1 | Strong invariant, int-only sound |
| `queue` | 3 | 1 | Clean FIFO/LIFO list contract |
| `float` | 2 | 1 | `is_integer` exact |
| `string` | 6 | 1 | Constant-string invariants |
| `enum` | 14 | 1 | Value equality / hash |
| `decimal` | 16 | 2 | Fixed-point arithmetic identities |
| `exceptions` | 0 | 2 | Raise/catch value semantics |
| `datetime` | 17 | 2 | Small stub |
| `os` / `time` | 5 / 3 | 2 | Small stubs |
| `dataclasses` | 44 | 2 | Mostly preprocessor-driven |
| `typing` | 169 | 2 | Type aliases, little runtime |
| `consensus` | 0 | 2 | Domain stub |
| `cmath` | 29 | 3 | Complex transcendentals (float-heavy) |
| `re` | 27 | 3 | Regex engine (large) |
| `numpy` / `torch` | 0 / 1 | 3 | Array/tensor (heavy) |
| `threading` / `threading_deadlock` | 48 | 3 | Needs a concurrency-harness style |

## 4. Tier 1 — do first (clean contracts, low effort)

Each is a self-contained PR of a handful of harnesses. Concrete functions and
candidate properties below.

### 4.1 `heapq` — the highest-value target

The model reproduces CPython's sift routines element-for-element, so it admits a
real structural invariant. **Use `int` elements only** — the model header
records `float`/`str` as "sound but loud" and `tuple` as *unsound* (frontend
types tuple elements as int; pinned by `github_5931_tuple_knownbug_fail`).

- `harness_heapq_invariant` — build a fixed-length list of nondet ints, call
  `heapq.heapify(h)`, assert the **min-heap property** for every parent/child:
  `h[k] <= h[2k+1]` and `h[k] <= h[2k+2]` where the child index is in range.
- `harness_heapq_pop_min` — after `heapify`, assert `heappop(h)` equals the
  minimum of the original elements (compare against `min([...])`), and that
  size decreases by one.
- `harness_heapq_push_keeps_invariant` — `heappush` a nondet item onto a heap
  and re-assert the invariant; size increases by one.
- `harness_heapq_pushpop_size` / `heapreplace` — assert the size-preservation
  contract (`heappushpop`/`heapreplace` leave `len(h)` unchanged) and that the
  returned item is `<=` every element left in the heap.
- Negative `harness_heapq_pop_min_fail` — assert `heappop` returns something
  strictly less than the true minimum, which is impossible → FAILED.

Cost: `heapify`/`_siftup` loop over the list; a length-4–7 list unwinds cheaply.

### 4.2 `queue`

`Queue` is a list-backed FIFO, `LifoQueue` a LIFO. `get()` on an empty queue
pops an empty list (an IndexError violation), so harnesses must guard with
`empty()`/`qsize()`.

- `harness_queue_fifo_order` — `put(a); put(b); put(c)`; assert `get()` returns
  `a, b, c` in that order and `qsize()` tracks 3→0.
- `harness_queue_lifo_order` — same puts on `LifoQueue`; assert `get()` returns
  `c, b, a`.
- `harness_queue_empty_full` — assert `empty()` iff `qsize()==0`, and `full()`
  iff `maxsize>0 and qsize()>=maxsize`.
- Negative `harness_queue_fifo_order_fail` — assert FIFO returns LIFO order →
  FAILED (order confusion).

### 4.3 `float`

- `harness_float_is_integer` — for nondet int `n`, `float(n).is_integer()` is
  True; for a nondet float `x`, `x.is_integer()` implies `x == int(x)`.
- Negative `harness_float_is_integer_fail` — assert every nondet float is
  integral → FAILED.

### 4.4 `string`

Constants only — concrete, intrinsic-free (also runs under CPython).

- `harness_string_constants` — `len(digits)==10`, `len(ascii_lowercase)==26`,
  `ascii_letters == ascii_lowercase + ascii_uppercase`, `'a' in ascii_lowercase`,
  `'0' in digits`, `hexdigits` starts with `digits`.

### 4.5 `enum`

Subclass `Enum`, construct members, exercise value semantics with nondet values.

- `harness_enum_eq` — two members with equal nondet `value` compare equal and
  hash equal; unequal values compare `!=`; `__ne__` is the negation of `__eq__`.
- Negative `harness_enum_eq_fail` — assert distinct-value members are equal →
  FAILED.

## 5. Tier 2 — medium effort

- **`decimal`** — the largest arithmetic model. Target algebraic identities on
  bounded values: `Decimal(a) + Decimal(b) == Decimal(b) + Decimal(a)`,
  `x - x == 0`, `x * 1 == x`, comparison trichotomy. Watch for solver cost on
  the fixed-point representation; bound magnitudes tightly.
- **`exceptions`** — value semantics of raise/catch: a raised `ValueError`
  carrying a nondet payload is caught by the matching handler and the payload
  survives; a non-matching handler does not catch. Pairs with the symbolic
  exception lowering (`docs/design-symbolic-exceptions.md`).
- **`datetime` / `os` / `time` / `consensus`** — small stubs; one harness each
  pinning the modelled surface (e.g. stub return-value contracts). Low value but
  cheap; do opportunistically.
- **`dataclasses` / `typing`** — mostly preprocessor/type-system driven with
  little runtime model surface. Prefer to strengthen the existing example tests
  into contract harnesses rather than add net-new; audit before investing.

## 6. Tier 3 — hard / different harness style

- **`cmath`** — complex transcendentals; same float-blowup risk as `gamma`.
  Restrict to concrete anchors and algebraic identities that avoid deep libm
  paths (e.g. `abs(complex)` Pythagorean checks on small integer components).
- **`re`** — a regex engine; symbolic inputs explode. Harnesses should pin
  concrete match/no-match contracts on fixed patterns, not nondet strings.
- **`numpy` / `torch`** — array/tensor models; heavy. Defer until a specific
  contract (shape, elementwise identity) is worth the encoding cost. See
  `docs/numpy-support-assessment.md`.
- **`threading` / `threading_deadlock`** — need a *concurrency* harness style
  (race / deadlock reachability), not the sequential requires/ensures template.
  Track separately; the existing 48 tests already cover much of the surface.

## 7. Sequencing

One tier-1 model per PR, in this order (value × tractability): **heapq → queue →
float → string → enum**. Each PR is a handful of harnesses plus at least one
`_fail`, following §2. Land tier-1 before opening tier-2, and treat tier-3 as
research spikes rather than routine harness work.

## 8. Risks and limitations

- **Frontend limits** (chained calls, tuple-typed heap elements, nondet bytes)
  cap what can be asserted; §2 lists the known ones. When a limitation blocks a
  harness, pin it as a `KNOWNBUG` referencing the tracking issue rather than
  forcing an unsound assertion.
- **Float cost.** Transcendental models must lean on concrete anchors; nondet
  floats through libm are the main source of CI-timeout risk.
- **Postcondition mirroring.** For pure-stub models whose only contract *is* an
  `__ESBMC_assume` bound (e.g. `random.randint`), the positive harness mirrors
  the assume and cannot detect a wrong bound in the model itself; the `_fail`
  companion pins the edge case instead. Accept this explicitly.
- **Soundness of the harness, not just the model.** A harness that passes
  vacuously (precondition too tight, or asserting a tautology) adds no coverage.
  Every positive harness must have a `_fail` sibling or a concrete anchor that
  demonstrably exercises the model.

## 9. Execution log (2026-07-11)

All three tiers were executed as a sequence of small, per-model PRs. Each PR
carries a positive harness plus at least one `_fail`, follows §2, and was
verified individually (the shared dev box is contended, so parallel local ctest
runs can spuriously time out — see §8).

| Tier | Model | PR | Notes |
|---|---|---|---|
| 1 | heapq | #5965 | min-heap invariant; int-only; symbolic heappush + post-append index hit a list-model IndexError (avoided, worth a separate fix) |
| 1 | queue | #5966 | FIFO/LIFO order, empty/full — nondet payloads |
| 1 | float | #5968 | `is_integer`; chained `float(n).is_integer()` mis-lowers, split via a local |
| 1 | string | #5969 | constants via indexing; high-unwind `len`/`in` hangs the runner |
| 1 | enum | #5970 | value equality; concrete-only (symbolic member select unsupported) |
| 2 | exceptions | #5971 | nondet dispatch, hierarchy, message survival |
| 2 | decimal | #5972 | algebraic identities; concrete-only (`Decimal()` rejects nondet) |
| 2 | datetime/time/os | #5973 | field defaults, `sleep` guard, `listdir` |
| 2 | dataclasses | #5976 | preprocessor `@dataclass` synthesis (init/eq/default/replace) |
| 3 | cmath | #5977 | modulus non-negativity + perfect-square anchors; `abs(z)**2==re**2+im**2` is NOT sound in IEEE-754 (falsified by ESBMC) |
| 3 | re | #5981 | literal match/search/fullmatch; char-class patterns return an unconstrained bool (model gap) |
| 3 | threading | #5983 | Lock mutual exclusion; requires `--data-races-check` or the proof is vacuous |
| 3 | numpy | #5984 | concrete array/reduce/scalar-math; `np.add` unsupported, `np.sum` constants-only |
| 3 | torch | #6017 | mm right-identity, allclose reflexivity/discrimination, cat width+value; concrete anchors, minimal tensors to fit the CI budget |

**Audited and skipped.** `typing` — pure type-alias stubs, no runtime surface.
`consensus` — trivial `hash()==42`, negligible value.

**Torch, revisited.** `torch` was initially deferred as "no nondet surface, so
it adds nothing beyond numpy." On closer inspection the torch model carries real
algorithmic contracts numpy's constant stubs lack — a triple-loop matrix
multiply (`mm`/`matmul`), column concatenation (`cat`), and an element-wise
tolerance compare (`allclose`) — so it was harnessed with concrete anchors
(PR #6017). Like the other float-heavy models, `cat`+`allclose` stacked in one
harness blows the CI budget, so tensor sizes and unwind bounds are kept minimal
and value checks use direct indexing rather than a second model call.

## 10. Model gaps surfaced — all resolved

The harness effort surfaced three suspected model gaps; each was root-caused and
closed:

- **`re` character-class patterns returning an unconstrained bool** (#5981) —
  **fixed in PR #5986.** A bare `[x-y]` class (no quantifier) fell through to the
  matcher's non-deterministic fallback; `try_match_char_class_range` and
  `search()` now recognise it.
- **`time.time()` `global` counter stalls the converter** (#5973) — **was a
  misdiagnosis, not a bug** (PR #5987). `time.time()` verifies correctly; the
  apparent stall was shared-machine CPU contention. The monotonicity coverage
  the misdiagnosis had blocked was added instead.
- **`heapq`/list-model IndexError on `heappush`/post-append indexing** (#5965) —
  **fixed in PR #5997.** Root cause was general (not heapq-specific): the
  convert-time constant-index bounds check used the caller's static list length,
  blind to a mutation performed through a function argument. Lists that escape
  into a call now fall back to the runtime bounds check. Follow-up PR #6000
  makes non-negative out-of-bounds reads raise a catchable `IndexError`,
  matching CPython.
