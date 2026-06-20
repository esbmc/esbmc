# ESBMC NumPy Support — Assessment and Implementation Roadmap

**Status:** Assessment as of master commit `a5383a3a12` (2026-06-08).
**Scope:** ESBMC's Python frontend NumPy support — frontend lowering
(`src/python-frontend/numpy_call_expr.cpp`), operational models
(`src/c2goto/library/python/{linalg,umath,list,math}.c`,
`src/python-frontend/models/numpy*.py`), and the 329-test
`regression/numpy/` suite.

> **Evidence convention.** Each finding is tagged:
> **[V]** verified by direct code inspection and/or an existing regression test;
> **[T]** confirmed by a named regression test;
> **[I]** inferred from code but **not** runtime-confirmed (flagged for follow-up);
> **[?]** open question / requires further investigation.

---

## 0. Executive Summary

ESBMC has a **breadth-first, demonstration-grade** NumPy implementation. It
covers a wide surface of the API (array creation, 1-D/2-D indexing, broadcasting
checks, element-wise ufuncs, a dozen math functions, `dot`/`matmul`/`det`,
complex numbers) but the implementation strategy is dominated by **compile-time
constant folding in the C++ frontend**, not symbolic encoding. The practical
consequences:

1. **Programs with literal arrays verify well** — the frontend computes results
   in host C++ and emits constants. **[V]**
2. **Programs with symbolic array *elements*** are supported only on a few
   narrow paths (`dot`/`matmul` via `linalg.c`, unary math ufuncs, scalar
   `fmod`). Most element-wise array arithmetic on symbolic data either falls
   back to an `int64`-only OM or is rejected. **[V]**
3. **Soundness hazards exist** in the float element-wise path (`umath.c` is
   `int64`-only) and in host-side constant folding (host `double`/`int`
   arithmetic substituted for target semantics). **[V]/[I]**
4. **Hard structural ceiling at 2 dimensions** — 3-D+ arrays are rejected by
   design. **[T]**
5. **Nested-list / tensor performance** is the dominant scalability blocker
   (issue #5121). **[V]**

The roadmap below prioritises (Phase 1) closing the **soundness** gaps before
breadth, then (Phases 2–4) widening symbolic coverage, lifting the 2-D ceiling,
and adding the large missing API surface (reductions, reshape, slicing,
`linalg`, statistics, random).

---

## 1. Architecture: how NumPy is actually executed

Understanding the soundness story requires understanding the two-layer design.

### 1.1 Layer A — frontend lowering (`numpy_call_expr.cpp`, 2455 lines) **[V]**

`numpy_call_expr::get()` is the single dispatch entry. For each `np.<fn>(...)`
call it does one of:

- **Array constructors** (`array`, `zeros`, `ones`): build a literal nested
  `List` JSON node and hand it to `converter_.get_static_array` →
  a statically-shaped GOTO array. `dtype=` is parsed and mapped to a `typet`
  (`get_typet_from_dtype`). **[V]**
- **Math ufuncs** (`sin`, `cos`, `exp`, `sqrt`, `arctan`, `fabs`, `ceil`,
  `floor`, `round`, `trunc`, `fmod`, `arccos`, `copysign`, …): if operands are
  concrete, **constant-fold in host C++** (`apply_numpy_unary_math` calls
  `std::sin` etc.); list-backed literal `floor`/`fabs`/`trunc` now fold
  through the same 1-D/2-D path, list-backed literal `arccos` folds through
  the same 1-D/2-D path, and runtime 1-D `arccos` lists now lower through the
  libm array OM. Scalar `fmod` still delegates to the shared math handler for
  libm lowering. **[V]**
- **Element-wise arithmetic** (`add`, `subtract`, `multiply`, `divide`,
  `power`): a cascade of `try_extract_scalar_*` attempts to fold the whole
  operation to literals (scalar, 1-D, 2-D, complex). If folding succeeds, the
  result is emitted as a literal `List`. If shapes are literal but elements are
  symbolic, lower to the `umath.c` OM. Otherwise reject. **[V]**
- **Linear algebra** (`dot`, `matmul`): if 2-D×2-D / 1-D etc., emit a call to
  `linalg.c::dot`/`dot_double`; `transpose` now lowers runtime 2-D array
  variables through `linalg.c::transpose`/`transpose_double`; `det` is
  **host-folded** for 2×2/3×3 constant matrices only. **[V]**
- **Complex ufuncs** (`real`, `imag`, `conj`, `conjugate`, `angle`, `abs`):
  host-fold for constants; symbolic fallback builds a complex struct and uses
  the complex/math handlers. **[V]**
- **Broadcasting**: `broadcast_check` validates shape compatibility up front and
  throws a NumPy-style "operands could not be broadcast together" error. **[T]**

### 1.2 Layer B — C operational models (FLAIL-mangled into the binary) **[V]**

- `linalg.c`: `dot`/`dot_double` (triple-loop matmul), `matmul`(=`dot`),
  `transpose`, `det` (2×2 only in the live code path; an `#if 0` Gaussian-
  elimination `det` is disabled). **[V]**
- `umath.c`: `add`/`subtract`/`multiply`/`divide` with broadcasting via
  `get_value`. **All `int64_t`** — no float variant. **[V]**
- `math.c`, `list.c`, `slice.c`, `string.c`: shared Python OMs reused by NumPy.

### 1.3 Layer C — Python type-inference stubs (`models/numpy.py`) **[V]**

`models/numpy.py` and `models/numpy/linalg.py` are **type-inference stubs only**
— e.g. `def power(...) -> float: return 42.0`, `def dot(...) -> float:
return 0.0`. They exist to give the converter a return type; **they do not
define the runtime semantics** (Layer A/B do). This is a frequent source of
confusion: the `42.0`/`0.0` bodies are never the verified values.

> **Soundness consequence of the design:** because most concrete operations are
> folded in host C++, ESBMC is **not encoding NumPy arithmetic into SMT** for
> the common (literal) case. It is trusting host `double`/`int64` arithmetic to
> match the target/NumPy semantics. This is fast and precise for small literal
> programs but (a) bypasses ESBMC's own overflow/rounding checks and (b) does
> not generalise to symbolic inputs. See §4.

---

## 2. Feature-coverage matrix

Legend: **Full** = symbolic + concrete, semantics match NumPy; **Partial** =
works under stated restriction; **Fold-only** = concrete literal operands only,
host-folded, no symbolic support; **Unsupported** = explicit error;
**Unsound** = produces results but semantics provably wrong on some inputs;
**Missing** = no handling.

| Functional area | Status | Restriction / evidence |
|---|---|---|
| **Array creation** `array` | Partial | 1-D/2-D literal only; 3-D+ rejected **[T]** `unsupported_3d_array_fail` |
| `zeros`,`ones` | Partial | int / 1-elt-tuple / 2-tuple shape; 3-D rejected; empty tuple rejected **[T]** |
| `empty`,`full`,`arange`,`eye`,`identity`,`linspace` | Missing | no handler **[V]** |
| **Indexing** `a[i]`, `a[i,j]`, `a[i][j]` | Partial | 1-D/2-D; tuple/negative/boolean scalar index **[T]** `github_4666_*` |
| Multi-dim `a[i,j,k]` (3-D) | Unsupported | explicit `TypeError` **[T]** |
| **Slicing** `a[i:j]`, `a[:,k]`, strided | Missing/Partial | `zero_step_slice_fail`, `tuple_slice_index_fail` reject; general slices not modelled **[T]/[I]** |
| Boolean *mask* indexing `a[a>0]` | Missing | only scalar `a[True]` works **[T]/[I]** |
| Fancy/integer-array indexing `a[[0,2]]` | Missing | no handler **[I]** |
| **Broadcasting** compatibility check | Full | rightmost-aligned rule, clear errors **[V]** `broadcast_check` |
| Broadcasting *evaluation* (concrete) | Partial | literal fold up to 2-D **[T]** `broadcast_add_2d_1d_success` |
| Broadcasting *evaluation* (symbolic, int) | Partial | `umath.c` int64 path **[I]** |
| Broadcasting *evaluation* (symbolic, **float**) | **Unsound** | `umath.c` is int64-only **[V]** — see §4.1 |
| **Shape** `.shape` | Partial | 1-D/2-D, literal **[T]** `shape_*` |
| `.ndim`, `.size`, `.dtype`, `.T` attr | Missing/Partial | `.T` via `transpose()` only **[I]** |
| `reshape`, `ravel`, `flatten`, `squeeze`, `expand_dims` | Missing | no handler **[V]** |
| `transpose` (1-D/2-D) | Partial | 1-D literal identity + 2-D literal/runtime array lowering; `transpose2`,`transpose7` KNOWNBUG **[T]** |
| **Arithmetic** `+ - * /` scalar/array | Partial/Fold-only | literal fold; symbolic int via umath **[V]** |
| `power` | Partial | host `std::pow` fold; int dtype overflow modelled **[T]** `power`, `power-overflow-fail` |
| `//` floor-div, `%` mod (array) | Missing/Partial | `np.fmod` supports literal list-backed 1D/2D broadcasting; runtime array operands remain rejected **[T]** `fmod_array_unsupported` |
| **Comparison/logical** `> < == & |` | Missing | no element-wise comparison ufunc **[I]** `mixed-types-comp` is scalar |
| **Reductions** `sum`,`prod`,`min`,`max`,`mean`,`argmin`,`argmax`,`cumsum` | Missing | no handler **[V]** |
| `np.fmax`,`np.fmin` (binary, element-wise) | Unsound/KNOWNBUG | KNOWNBUG; stub returns 0.2 **[T]** |
| **Linear algebra** `dot` | Partial | 1-D/2-D, int + float backends **[T]** `dot*`, `github_5115_*`; `dot6`(bool),`dot7` KNOWNBUG |
| `matmul` | Partial | 1-D/2-D, symbolic elements OK **[T]** `matmul_*` |
| `linalg.det` | Partial | **constant** 2×2/3×3 only **[T]**; 1×1,4×4 rejected |
| `linalg.inv`,`solve`,`eig`,`svd`,`norm`,`qr`,`cholesky` | Missing | no handler **[V]** |
| **Math fns** `sin cos exp sqrt arctan fabs ceil floor round trunc arccos copysign` | Partial | scalar fold + list-backed literal `floor`/`fabs`/`trunc`; runtime 1-D `arccos` array lowering **[T]** |
| `tan arcsin tan2 log log2 log10 sinh cosh tanh` | Missing | not in `is_math_function` **[V]** |
| `remainder`,`rint`,`nextafter`,`modf`,`frexp`,`isclose` | KNOWNBUG | declared, not correctly modelled **[T]** |
| Constants `np.pi`, `np.e` | Missing/KNOWNBUG | `e` KNOWNBUG **[T]** |
| **Complex** `1+2j`, abs/conj/real/imag/angle | Partial | scalar + 1-D/2-D literal; symbolic fallback **[T]** `complex_*` |
| complex **dtype constructor** `np.zeros(..,dtype=complex)` | Unsupported | explicit `TypeError` **[T]** |
| **dtypes/casting** int/uint/float/bool widths | Partial | parsed, mapped to bitvector widths; overflow modelled for scalar fold **[T]** |
| `object` dtype, `astype`, structured dtype | Unsupported/Missing | `object` rejected **[T]** |
| **Statistics** `std`,`var`,`median`,`percentile`,`histogram` | Missing | no handler **[V]** |
| **Sorting/search** `sort`,`argsort`,`searchsorted`,`unique`,`where`,`nonzero` | Missing | no handler **[V]** |
| **Random** `rand`,`randn`,`randint`,`seed`,`choice` | Missing | no handler **[V]** |
| **Structured/record arrays** | Missing | no handler **[V]** |
| **Views vs copies / memory layout / strides** | Missing | not modelled; arrays are value lists **[V]** |
| **Iteration** `np.nditer`, `flat`, `ndenumerate` | Missing | no handler **[V]** |

---

## 3. Unsupported / partially supported API — consolidated list

**Explicitly rejected with a diagnostic (good — fails loudly): [T]**
- 3-D+ arrays in `array`/`zeros`/`ones` and `a[i,j,k]` indexing.
- `np.linalg.det` outside constant 2×2/3×3 square matrices.
- complex / `object` dtype in constructors; non-literal `dtype=` elements.
- `np.fmod` on array operands; operation on two empty arrays.
- broadcasting shape mismatches; zero-step slices.

**Declared but KNOWNBUG (silently wrong / not modelled): [T]**
`copysign`, `fmax`, `fmin`, `rint`, `round2`, `remainder`, `nextafter`,
`modf`, `frexp`, `isclose`, `dot6` (bool), `dot7`, `transpose2`, `transpose7`,
`e` constant. (15 KNOWNBUG tests total.)

**Entirely missing (no handler): [V]**
`empty`, `full`, `arange`, `eye`, `identity`, `linspace`, `reshape`, `ravel`,
`flatten`, general slicing, boolean-mask & fancy indexing, all reductions
(`sum`/`mean`/`min`/`max`/`argmax`/…), comparison ufuncs, `where`, `sort`,
`unique`, statistics, random, `linalg.{inv,solve,eig,svd,norm}`, structured
arrays, views/strides, `tan`/`log`/hyperbolic math fns, `astype`, `np.pi`.

---

## 4. Verification-risk assessment (the soundness core)

ESBMC is a formal tool; these are the findings that matter most.

### 4.1 **[V] HIGH — `umath.c` element-wise ops are `int64`-only (unsound for float arrays)**

`src/c2goto/library/python/umath.c` defines `add/subtract/multiply/divide`
exclusively over `int64_t*`. The frontend (`numpy_call_expr.cpp:2072–2144`)
routes **symbolic-element** array arithmetic to these models with a
`flat_ptr_type` cast. There is **no `*_double` variant**, unlike `linalg.c`
which gained `dot_double` precisely to fix this class of bug (#5115, PR #5205).

- **Impact:** `np.add(a, b)` where `a`,`b` are float arrays with non-constant
  elements reinterprets the IEEE-754 bit pattern as `int64`, producing
  garbage — a **false verdict** either way. **[V by inspection]**
- **Why tests don't catch it:** every passing element-wise test uses *literal*
  operands, which are host-folded **before** reaching `umath.c`; the surviving
  symbolic tests (`elementwise_name_*`) exercise the **unary math** path, not
  `umath` binary ops. **[V]** → genuine coverage hole.
- **Follow-up [I]:** construct `np.add` of two symbolic float arrays and confirm
  the wrong verdict, then add `add_double`/`subtract_double`/… mirroring
  `dot_double`, selected by `base_type.is_floatbv()` (same predicate already
  used at `numpy_call_expr.cpp:2020`).

### 4.2 **[V] MEDIUM — host-side constant folding substitutes host arithmetic for target/NumPy semantics**

Concrete ufuncs fold via `std::sin`, `std::pow`, host `double`/`int`
(`apply_numpy_unary_math`, `create_binary_op`, `compute_scalar_result`).
Risks:
- **FP rounding:** host `double` rounding may differ from the SMT FP encoding
  ESBMC would otherwise use; an assertion that is *just* satisfiable could flip.
- **Integer `power`:** `std::pow(left,right)` is computed in `double`; large
  integer powers lose precision before the int dtype wrap is applied
  (`numpy_call_expr.cpp:2323`). **[I]** — e.g. `np.power(3, 40)` exceeds 2^53.
- **Folding bypasses ESBMC's own overflow/div-by-zero checks** for the folded
  expression; correctness then depends on the frontend's ad-hoc
  `emit_numpy_overflow_assertion` and explicit zero checks rather than the
  symex checkers. **[V]**
- **Justification gate:** folding is acceptable *only* where the operands are
  provably concrete and the host op is proven equivalent to the target op.
  Today that equivalence is assumed, not asserted. Recommend (Phase 1) an
  audit + a `--no-numpy-fold` escape hatch that forces SMT encoding for
  differential testing.

### 4.3 **[V] MEDIUM — `linalg.c::dot` accumulator width / overflow**

Integer `dot` accumulates into `int64_t` with no overflow assertion; NumPy
would wrap at the array dtype width (e.g. `int32`). For symbolic int matrices
this can **miss real overflow bugs** or report results NumPy would not produce.
The float path uses `double` accumulation (matches NumPy reasonably). **[I]**

### 4.4 **[V] MEDIUM — 2×2 `det` host-folds; Gaussian `det` is `#if 0`-disabled**

`det` only supports constant 2×2/3×3 and computes in host `double`
(`determinant_2x2/3x3`). No symbolic determinant; the general
LU/Gaussian `det` in `linalg.c` is commented out. Safe (rejects the rest) but
narrow. **[V]**

### 4.5 **[V] MEDIUM — unbounded native recursion on deep NumPy expressions (issue #5048)**

Deep subscript/attribute chains hit unguarded recursion in
`converter_expr.cpp::get_expr`; a crafted nested NumPy access can SIGSEGV
rather than return a verdict — a robustness/DoS hazard, not a wrong answer.
**[V — tracked]**

### 4.6 **[V] LOW/SCALE — nested-list (tensor) state explosion (issue #5121)**

2-D/3-D nested-`PyListObject` element access encodes expensively; `cat`/`split`
take 17–35 s or never converge vs ~2.5 s scalar baseline. Caps practical
array sizes well below real workloads. Suggested fix: flat-array backing for
rectangular nested lists. **[V — tracked]**

### 4.7 **[I] LOW — broadcasting check vs broadcasting evaluation divergence**

`broadcast_check` validates compatibility, but the *evaluation* paths
(`build_broadcast_literal_result`, `umath.c`) re-derive shapes independently.
A mismatch between checker and evaluator (e.g. a shape the checker accepts but
the evaluator mis-tiles) would be silent. Recommend a single shared shape
oracle. **[I]**

### 4.8 State-space / scalability summary

- Every array is a **statically-shaped, fully-unrolled** value list → cost grows
  with element count; large arrays explode (4.6).
- Constant folding *reduces* state for literal programs but does nothing for
  symbolic ones.
- No symbolic *shape* support: shapes must be literal at conversion time (the
  element-wise path throws "require literal arrays" otherwise). **[V]**

---

## 5. Gap analysis: ESBMC vs NumPy by functional area

| Category | NumPy expectation | ESBMC today | Gap severity |
|---|---|---|---|
| Array creation | n-D, many constructors | 1-D/2-D `array`/`zeros`/`ones` | High (no `arange`/`full`/`eye`/`linspace`; 2-D ceiling) |
| Indexing/slicing | full slicing, fancy, boolean mask | scalar/tuple/negative index, 1-D/2-D | High (no real slicing, no masks, no fancy) |
| Broadcasting | n-D evaluate | check ✓, evaluate ≤2-D concrete; **float symbolic unsound** | Med-High (soundness 4.1) |
| Shape manip | reshape/ravel/transpose/stack | `.shape`, 2-D `transpose` | High |
| Arithmetic | n-D ufuncs symbolic | literal fold + int-only symbolic | High (4.1, 4.2) |
| Comparison/logical | element-wise `> == & |` | scalar only | High |
| Reductions | `sum/mean/min/max/argmax/...` | none | High |
| Linear algebra | `dot/matmul/inv/solve/eig/svd/det/norm` | `dot`/`matmul`/(const)`det` | High |
| Random | full RNG | none | Medium (verification of RNG is itself nondet) |
| Math fns | ~40 ufuncs symbolic | ~12, many fold-only; KNOWNBUGs | Medium |
| Statistics | `std/var/median/...` | none | Medium |
| Sorting/search | `sort/argsort/where/unique` | none | High (`where` is common) |
| Structured arrays | record dtypes | none | Low |
| dtype/casting | full promotion, `astype` | width map + scalar overflow; no `astype` | Medium |
| Views vs copies | aliasing semantics | value-list, no aliasing | Med (correctness if user relies on views) |
| Memory layout | strides/order | not modelled | Low (unless exposed) |
| Iteration | `nditer/flat` | none | Low |

---

## 6. Prioritised implementation roadmap

Guiding principle: **soundness before breadth.** A wrong "VERIFICATION
SUCCESSFUL" is worse than an honest "unsupported". Each item lists
*Rationale · Dependencies · Complexity · Tests · Verification risk · Priority*.

### Phase 1 — Soundness & core correctness (must-do)

**P1.1 — Fix `umath.c` float element-wise ops (4.1).** **[P0]**
- *Description:* add `add_double/subtract_double/multiply_double/divide_double`
  to `umath.c`; select by `base_type.is_floatbv()` in `numpy_call_expr.cpp`
  (mirror the `dot_double` pattern at `:2020`).
- *Rationale:* removes an active unsoundness on symbolic float arrays.
- *Dependencies:* none. *Complexity:* Low (≈ existing dot_double change).
- *Tests:* `np.add`/`subtract`/`multiply`/`divide` of two symbolic-float arrays
  (`nondet_float()` elements) — one SUCCESS, one FAIL; plus a regression that
  *fails on current `master`* (Mode C / sanitizer-style witness).
- *Risk:* low; *Priority:* **highest** (soundness).

**P1.2 — Differential-test the constant-folding paths (4.2).** **[P0]**
- *Description:* add `--no-numpy-fold` to force SMT encoding; run the numpy
  suite both ways; any verdict divergence is a folding bug. Fix integer
  `power` to fold in exact integer arithmetic (BigInt) not `double`.
- *Rationale:* validates the host-arithmetic assumption that underpins the
  whole frontend.
- *Dependencies:* small frontend flag plumbing. *Complexity:* Medium.
- *Tests:* `np.power(3,40)` int exactness; FP boundary assertions.
- *Risk:* may surface latent bugs (desired); *Priority:* **high**.

**P1.3 — Bound the recursion (4.5 / #5048).** **[P1]**
- *Description:* depth-guard `converter_expr.cpp::get_expr` (reuse the
  `#5048` get_type guard pattern from memory `esbmc-5048-deep-type-recursion`).
- *Complexity:* Low. *Tests:* deeply nested subscript → clean error, no crash.
- *Risk:* low; *Priority:* high (robustness).

**P1.4 — Integer `dot` overflow semantics (4.3).** **[P1]**
- *Description:* accumulate/wrap at the operand dtype width, or emit an overflow
  assertion. *Complexity:* Medium. *Tests:* int32 matmul that overflows.
- *Risk:* med (could change existing verdicts — gate behind dtype). *Priority:* med.

**P1.5 — Clear all 15 KNOWNBUG math fns or downgrade to explicit-unsupported.**
**[P1]** A KNOWNBUG that "passes" today (`copysign`, `fmax`, `fmin`, `rint`,
`remainder`, `nextafter`, `modf`, `frexp`, `isclose`, `e`) is a silent-wrong
risk. Either model correctly (reuse libm OMs) or make them throw "unsupported".
*Complexity:* Low–Med. *Priority:* high (each is a latent false verdict).

### Phase 2 — Broad practical coverage

**P2.1 — Reductions:** `sum`, `prod`, `min`, `max`, `mean`, `argmin`, `argmax`,
with `axis=`. *Rationale:* ubiquitous. *Dependencies:* shared loop-OM over the
list backing; symbolic-safe. *Complexity:* Medium. *Tests:* axis=None/0/1,
empty-array semantics, overflow on int `sum`. *Risk:* med (axis shape logic).
*Priority:* high.

**P2.2 — Comparison & logical ufuncs:** `>,<,>=,==,!=`, `logical_and/or/not`,
returning bool arrays. *Dependency:* element-wise framework from P1.1.
*Complexity:* Medium. *Tests:* element-wise compare + broadcast. *Priority:* high.

**P2.3 — `where`, boolean-mask indexing `a[mask]`.** *Rationale:* the most
common "missing" idiom; needed for realistic code. *Complexity:* High
(variable-length result → either bound it or keep shape via masked-select).
*Verification risk:* High — masked indexing yields data-dependent shape; prefer
a sound *bounded* model (fixed-capacity result + validity flags) over an
unsound dynamic one. *Priority:* high but careful.

**P2.4 — General slicing** `a[i:j]`, `a[:,k]`, `a[i:j:s]` (1-D/2-D).
*Dependency:* `slice.c` exists for Python lists — reuse. *Complexity:* Medium.
*Tests:* strided, negative, empty slice; view-vs-copy note (P3.5). *Priority:* high.

**P2.5 — `arange`, `full`, `eye`, `identity`, `linspace`.** *Complexity:* Low–Med
(literal-shape constructors). *Tests:* each + dtype. *Priority:* med.

**P2.6 — Missing scalar math fns** `tan`, `arcsin`, `log`, `log2`, `log10`,
`sinh/cosh/tanh`, constants `np.pi/np.e`. Reuse libm OMs. *Complexity:* Low.
*Priority:* med.

### Phase 3 — Advanced support

**P3.1 — `reshape`/`ravel`/`flatten`/`squeeze`** on a flat-array backing
(co-design with #5121 fix). *Complexity:* High. *Verification risk:* must
preserve element identity; *Priority:* med-high (unlocks tensor code).

**P3.2 — Symbolic shapes / `dtype` promotion table.** Allow shape values that
are symbolic within a static bound; implement NumPy's type-promotion lattice
faithfully. *Complexity:* High. *Risk:* High (promotion is subtle — verify
against NumPy's `result_type`). *Priority:* med.

**P3.3 — `linalg.{inv,solve,norm}`** (symbolic, small fixed N via OM);
generalise `det` to N×N via the disabled Gaussian routine **with** pivoting
overflow/zero checks. *Complexity:* High. *Risk:* High (numerical). *Priority:* med.

**P3.4 — Broadcasting evaluation to true n-D (within the 2-D→k-D lift).**
Unify checker + evaluator on one shape oracle (4.7). *Complexity:* High.
*Priority:* med.

**P3.5 — Views vs copies / aliasing.** Decide and document a sound model: either
"all NumPy ops copy" (sound, may miss aliasing bugs — document it) or model
views explicitly (high cost). *Priority:* low-med; **must be explicit**.

### Phase 4 — Long-term / research

- **P4.1** Lift the hard 2-D ceiling to general n-D (depends on flat backing +
  reshape + n-D indexing). *Complexity:* Very High.
- **P4.2** `np.random` with a sound nondet model (treat draws as nondet within
  documented distribution support; `seed` ignored or modelled as nondet).
  *Risk:* verifying RNG-dependent code is inherently about all draws.
- **P4.3** Structured/record arrays; `astype` with full casting rules.
- **P4.4** Sorting/searching (`sort`, `argsort`, `searchsorted`, `unique`) as
  bounded symbolic OMs (sorting networks for small N).
- **P4.5** Statistics (`std`, `var`, `median`, `percentile`).
- **P4.6** Scalability: SMT-array (theory-of-arrays) backing instead of unrolled
  value lists, to break the #5121 wall for large arrays.

---

## 7. Recommended pull-request sequence

Each PR is small, independently testable, and ships ≥1 passing + ≥1 failing
regression test (repo convention). Soundness PRs first.

1. **PR-1** `umath.c` float models + selection (P1.1) — *soundness*; include the
   `master`-failing witness. Label `python`.
2. **PR-2** recursion depth-guard for `get_expr` (P1.3). Label `python`.
3. **PR-3** `--no-numpy-fold` differential flag + integer-exact `power` (P1.2).
4. **PR-4** KNOWNBUG math fns: model or hard-unsupport, per fn (P1.5) — can be a
   short series (one fn or small group per PR).
5. **PR-5** int `dot` overflow semantics (P1.4).
6. **PR-6** reductions `sum/min/max/mean` + `axis` (P2.1).
7. **PR-7** comparison/logical ufuncs returning bool arrays (P2.2).
8. **PR-8** general 1-D/2-D slicing reusing `slice.c` (P2.4).
9. **PR-9** `where` + bounded boolean-mask indexing (P2.3).
10. **PR-10** `arange/full/eye/identity/linspace` (P2.5) + missing math fns (P2.6).
11. **PR-11+** Phase 3/4 items, each gated on its dependency.

> Sequencing rationale: PRs 1–5 raise the **trust floor** so that subsequent
> breadth additions inherit a sound element-wise/encoding substrate. Building
> reductions/comparisons (6–7) on top of an unsound element-wise path would
> propagate the 4.1 bug.

---

## 8. Regression-test strategy

The existing 329-test suite is broad but **literal-heavy**. Strengthen it along
the axes that actually exercise the solver:

1. **Symbolic-element tests for every op.** For each ufunc/reduction, add a
   `nondet_*()`-seeded variant so the SMT backend (not the folder) is exercised.
   This is the single biggest coverage gap and is what would have caught 4.1.
2. **Differential pairs.** A `*_fold` (literal) and `*_symbolic` test for each
   op, ideally cross-checked under `--no-numpy-fold` (P1.2) so folder and
   encoder must agree.
3. **Dual-solver gate.** Per repo policy, soundness-sensitive numpy tests should
   pass under both Bitwuzla and Z3 (the IREP2 numpy PRs already do this for the
   329 suite).
4. **CPython oracle.** Keep using `scripts/check_python_tests.sh` so each
   `main.py` is a valid reference program under real CPython+NumPy before
   trusting ESBMC's verdict. Add NumPy to the venv it uses.
5. **Boundary tests per feature** (NumPy semantics): empty arrays, 0-d scalars,
   broadcasting edges (`(2,1)+(1,3)`), integer overflow per dtype width,
   division/`fmod` by zero, NaN/Inf propagation, negative/`-0.0`.
6. **Negative/`unsupported` tests** for everything still rejected, asserting the
   *exact* diagnostic (locks the support boundary — already done well; keep it).
7. **Performance guard.** A few size-scaled nested-list tests with a CI timeout
   to detect regressions of the #5121 wall.
8. **Mode-C dead/live proofs** for any PR that adds/removes a branch in the
   frontend (per repo workflow rules).

---

## 9. Concrete expected-behaviour examples per area

These illustrate the *intended* post-roadmap contract; `# now:` notes current
behaviour.

```python
# --- Array creation (P2.5) ---
a = np.arange(5)            # now: unsupported; want: [0 1 2 3 4]
b = np.full((2,2), 7)       # now: unsupported; want: [[7 7][7 7]]

# --- Element-wise float arithmetic (P1.1 soundness) ---
x = np.array([nondet_float(), nondet_float()])
y = np.array([1.0, 2.0])
z = np.add(x, y)            # now: int64 reinterpret (UNSOUND); want: z[i]==x[i]+y[i]
assert z[0] == x[0] + 1.0

# --- Reductions (P2.1) ---
s = np.sum(np.array([1, 2, 3]))      # now: unsupported; want: 6
m = np.max(np.array([[1,9],[4,2]]))  # now: unsupported; want: 9

# --- Comparison ufunc + where (P2.2/P2.3) ---
a = np.array([1, -2, 3])
mask = a > 0                          # now: unsupported; want: [True False True]
pos = np.where(a > 0, a, 0)           # want: [1 0 3]

# --- Slicing (P2.4) ---
a = np.array([10,20,30,40])
assert a[1:3][0] == 20                # now: general slice unsupported

# --- Linear algebra (already partial; P3.3 extends) ---
A = np.matmul([[1,2],[3,4]], [[5,6],[7,8]])
assert A[0][0] == 19                  # now: SUPPORTED [T]
d = np.linalg.det([[1,2],[3,4]])
assert d == -2                        # now: SUPPORTED (const 2x2) [T]

# --- Broadcasting (check works; eval float symbolic is the gap) ---
np.add([[1,2,3]], [[1,2]])            # now: clean broadcast error [T]

# --- Integer overflow per dtype (P1.4 / existing scalar) ---
assert np.power(2, 7, dtype=np.int8) == -128   # now: SUPPORTED scalar [T]
```

---

## 10. Open questions for follow-up

- **[?]** Does the symbolic-float `umath` path (§4.1) actually mis-verify, or is
  there a guard that rejects it before lowering? (Strong code evidence it does
  not — needs a runtime witness; this is PR-1's failing test.)
- **[?]** Do host-folded FP results ever diverge from ESBMC's SMT FP encoding on
  a satisfiability boundary? (P1.2 differential run answers this.)
- **[?]** What is the intended view/copy contract? Currently value-copy
  everywhere — is any user relying on view aliasing semantics?
- **[?]** Should the 2-D ceiling be lifted, or is 2-D the supported contract?
  This decision gates Phase 3/4 scope.
- **[?]** Target dtype for default Python `int` arrays in arithmetic — NumPy uses
  platform int / int64; ESBMC's fixed-width assumption (issue #4642) interacts
  with this.

---

*Prepared from code inspection of `numpy_call_expr.cpp`, `linalg.c`, `umath.c`,
`models/numpy*.py`; the 329-test `regression/numpy/` suite; and the
issue/PR history (#2258–#5251). Findings tagged [V]/[T] are evidence-backed;
[I]/[?] require runtime confirmation and are called out as such.*
