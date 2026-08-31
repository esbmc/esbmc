# Plan — cost paid for paths that cannot execute (the #7361 defect class)

**Status:** W1, W2, W3, W5.1 and W6 implemented and in review (PRs
[#7432](https://github.com/esbmc/esbmc/pull/7432),
[#7435](https://github.com/esbmc/esbmc/pull/7435),
[#7436](https://github.com/esbmc/esbmc/pull/7436) — stacked on #7435 —
[#7437](https://github.com/esbmc/esbmc/pull/7437),
[#7438](https://github.com/esbmc/esbmc/pull/7438)). **W4 was re-scoped after
measurement and not implemented as written**: §6 records why, and what replaces
it — the revised item landed as
[#7440](https://github.com/esbmc/esbmc/pull/7440), with the underlying
simplifier gap fixed by [#7441](https://github.com/esbmc/esbmc/pull/7441).
§6.4 records the one taxed operation deliberately left alone, `remove`, and the
measurement that rejected the obvious fix. W5.2 is now localised to a single
condition (§7) but is still not a specified change.
measurement that rejected the obvious fix. W5.2 is still an unlocalised
investigation.
**Origin:** [#7361](https://github.com/esbmc/esbmc/pull/7361), *"[python] Avoid
duplicated shifts in `list.remove()`"*, which split a search loop and a shift
loop that had been nested. This plan generalises that fix into a screening test
(§2) and applies it across the Python and C++ operational models and the
`memcpy`/`memcmp` intrinsics.
**Last updated:** 2026-08-30 (§6 re-scoped against measurement).

**Measurement environment.** All numbers below were measured on an aarch64 macOS
host against `build/src/esbmc/esbmc`, ESBMC 8.5.0, built from master
`49995366ca`. Every quantity quoted is a **count**, not a timing: symex
assignments and generated VCCs come from the two lines

```
Symex completed in: 0.13s (1026 assignments)
Generated 693 VCC(s), 294 remaining after simplification
```

and loop-unwinding counts from `--symex-ssa-trace`. All three are produced
before the solver is invoked, so they are deterministic, host-independent and
solver-independent, and no interleaved A/B runner is needed to compare them.
Every row below was reproduced at least twice with identical counts.

---

## 1. The defect class

The diagnosis in #7361 was that `list.remove()` nested its shift loop inside
its search loop, so symex emitted one full shift per candidate index — quadratic
in the list length for a routine that shifts exactly once. The general form is:

> **Symex pays the full unwound cost of every branch whose guard it cannot
> decide, so a branch that is semantically unreachable but syntactically live
> costs its whole weight on every visit.**

Two things make an operational model unusually prone to this. Its guards read
fields out of heap objects (`elem->size`, `l->items[i].type_id`), which do not
constant-fold, so an `if`/`else` chain that a human reads as a dispatch is, to
symex, a set of paths all of which must be explored. And its fallback arms tend
to be `memcpy`/`memcmp` over a symbolic length, whose byte loops then unwind to
whatever `--unwind` the user happened to pass.

The consequence is a cost the user cannot see or attribute: it is charged inside
a model they did not write, and it scales with a bound their program does not
use.

---

## 2. The screening test

The defect class has a signature that needs no source reading:

> **Hold the program entirely constant and raise `--unwind`. Any growth in
> assignments is work on a path that cannot execute.**

A program with no symbolic input and no loops has nothing whose cost may depend
on the unwinding bound. Applying it across both frontends:

| program (all operands constant) | u8 | u16 | u32 | ratio |
|---|---:|---:|---:|---:|
| `xs[1:]` on `[0,1,2,3]` | 3,480 | 9,192 | 28,296 | **8.1×** |
| `vector` ×4 `push_back` after `reserve(11)` | 1,374 | 2,190 | 3,822 | **2.8×** |
| `"abcdefgh".replace("a","z")` | 89 | 90 | 90 | 1.0× |
| `{1:10, 2:20, 3:30}[2]` | 574 | 574 | 574 | 1.0× |
| `std::string a = "abc"; a + "def"` | 319 | 319 | 319 | 1.0× |

The two growing rows are the subject of this plan. The three flat rows are the
existence proof that flatness is achievable in this codebase — the Python string
path reached it through the constant-folding work in #7373/#7374/#7375.

**Control.** `list.remove()`, the routine #7361 fixed, now measures flatter than
`append`: growth exponents 0.54 and 0.73 respectively across n = 4→16. The
method reproduces the result it was derived from.

---

## 3. W1 — `list.sort()` explores the string arm on integer lists

**Where.** `src/c2goto/library/python/list.c:1242`, `__ESBMC_list_sort`.

The comparison dispatch leads with the field read:

```c
if (prev->size == 8 && type_flag == 0)        /* integer */
else if (prev->size == 8 && type_flag == 1)   /* float */
else if (prev->size == 8 && type_flag == 3)   /* mixed */
else if (prev->size == 1)                     /* bool */
else                                          /* string: memcmp */
```

`prev->size` is read through `&l->items[j-1]`, so symex cannot decide it and
walks the whole chain — including the `memcmp` arm, whose byte loop over a
symbolic length unwinds to the bound. `type_flag` *is* concrete: the call site
is `list_sort(xs, 0, 0)` in the GOTO program, a literal supplied by the
frontend, and it alone determines which arm applies.

**Measured.** Sorting the reverse-ordered constant list `[7,6,5,4,3,2,1,0]` —
every element an integer, so the string arm is unreachable:

| | u8 | u12 | u20 |
|---|---:|---:|---:|
| `xs.sort()` assignments | 17,635 | 23,907 | 36,451 |
| generated VCCs | 16,144 | 21,968 | 33,616 |

At `--unwind 20` the trace shows **28 `__memcmp_impl` unwind groups**, one per
inner-loop comparison (28 = 7+6+…+1), every one of them reaching the bound. The
slope is ≈ +1,570 assignments per unit of `--unwind`; extrapolated to zero it
leaves ≈ 5,100, so roughly **70 % of the cost at `--unwind 20` is the arm that
cannot run**.

**The 36× comparison.** On the identical list, `sorted(xs)` costs **1,005**
assignments against `sort()`'s 36,451. They are not the same code: `sorted()` is
lowered by the frontend into an injected Python model and never reaches
`__ESBMC_list_sort`. Two implementations of one operation, differing by 36×.

**Fix.** Dispatch on the concrete parameter first, so the string arm is pruned
at symex rather than explored:

```c
if (type_flag == 2) { /* memcmp */ }
else                { /* numeric arms, dispatching on prev->size */ }
```

The dispatch then also hoists out of both loops. Whether `list.sort()` should
additionally be routed through `sorted()`'s lowering is a separate question this
plan does not decide — it would close the 36× gap outright, but it changes which
implementation is authoritative for in-place sort semantics.

**Second, independent item.** `__ESBMC_list_sort` is an insertion sort that
moves whole `PyObject` structs, O(n²) of them. Sorting an index permutation and
applying one pass at the end is the direct analogue of #7361's search/shift
split — n copies instead of n²/2. It stacks with the dispatch fix and is not
subsumed by it.

---

## 4. W2 — `list.extend()` never got the constant-size parameter

**Where.** `src/c2goto/library/python/list.c:698`, `__ESBMC_list_extend`;
frontend call site `src/python-frontend/python-list/list_mutation.cpp:901`,
`python_list::build_extend_list_call`.

`extend` copies each element with

```c
__ESBMC_copy_value(elem->value, elem->size, elem->type_id, 0, NULL, 0);
```

The length comes out of the element header, so it is symbolic, so the copy
misses every 8/16/24/32-byte fast path in `__ESBMC_copy_value` (`list.c:102`)
and lands in `memcpy`.

**The fix already exists twenty lines away.** `__ESBMC_list_copy_shallow_sz`
(`list.c:1004`) and `__ESBMC_list_store_elem` (`list.c:1037`) take an
`elem_size` parameter carrying the frontend's statically-known element width,
precisely so the copy sees a compile-time constant. The comment at `list.c:1033`
states the reason. `extend` was never given the same parameter, and its frontend
call site passes only `(list, other)`.

**Measured.** Assignments / generated VCCs at `--unwind 60`:

| | n=4 | n=8 | n=16 | memcpy unwinds @ n=16 |
|---|---:|---:|---:|---:|
| `xs.extend(ys)` | 5,478 / 4,809 | 14,970 / 15,637 | 46,338 / 55,437 | **944** |
| `xs.copy()` *(threads `_sz`)* | 837 / 497 | 1,633 / 1,045 | 3,513 / 2,429 | **0** |

944 is 16 calls × 59 iterations, every call capped. Growth exponent for `extend`
is 1.54 against `copy`'s 1.03.

**Fix.** Add an `elem_size` parameter to `__ESBMC_list_extend` and route the
per-element write through `__ESBMC_list_store_elem`, mirroring
`__ESBMC_list_copy_shallow_sz`. The frontend already computes the same width for
the copy path. `__ESBMC_set_add` and `__ESBMC_set_discard` share this code and
inherit the improvement.

---

## 5. W3 — the scalar copy fallback, reached by slicing

**Where.** `src/c2goto/library/python/scalar.c:24`,
`__python_scalar_bytes_copy`; and `src/c2goto/library/python/list.c:102`,
`__ESBMC_copy_value`.

Both dispatch on a `size` that reaches them from an element header, and both
end:

```c
else
  memcpy(dst, src, size);
```

Scalar payloads are 1, 2, 4 or 8 bytes, so the fallback is never taken — but it
is always explored. Python list slicing is lowered through an injected model
that calls `__python_scalar_bytes_copy`, which is why slicing is the most
expensive list operation measured.

**Measured.** `ys = xs[1:]` over the four-element constant list `[0,1,2,3]`:

| `--unwind` | 6 | 10 | 20 | 40 |
|---|---:|---:|---:|---:|
| assignments | 2,452 | 4,668 | 13,008 | 41,688 |

At `--unwind 20` that is **380 `__memcpy_impl` byte-loop unwindings**, 20 of
them capped, for a three-element result. At `--unwind 60` the same slice over 16
elements does not finish within 200 s.

**Fix.** Thread the constant width wherever the frontend knows it, as in W2.
Where it genuinely cannot, give the scalar path an explicit width cap so the
byte loop cannot be entered at all — the fallback exists for payload shapes the
Python frontend does not produce.

---

## 6. W4 — re-scoped: folding is not this plan's defect class

**This item is not what the survey said it was.** As first written, W4 proposed
constant-folding list operations, on the strength of the `str`-versus-`list`
table below. Re-running §2's screening test against a build carrying W1, W2 and
W3 shows that the operations it named do **not** pay an unwind tax, so they are
outside the class this plan is about. What replaces W4 is stated at the end of
this section.

### 6.1 The original observation, which still holds

`str` operations over constant operands are decided at conversion time and cost
essentially nothing, flat in the length of the string; `list` operations are
handed to the C model. `--unwind 24`, assignments:

| operation | n=4 | n=8 | n=16 |
|---|---:|---:|---:|
| `s.find` / `s.count` / `"x" in s` | 78 | 78 | 78 |
| `s[1:]` / `s + s` | 79 | 79 | 79 |
| `s.replace(…)` | 86 | 90 | 98 |
| `s.split(…)` | 196 | 196 | 249 |
| `s.upper()` | 129 | 161 | 225 |

That asymmetry is real, and folding list operations would still be a speedup.
It is simply a *different* optimisation: it removes real work by evaluating it
early, rather than removing work that can never run.

### 6.2 What the screening test says

§2's test is the arbiter: hold the program constant and raise `--unwind`; any
growth is work on a path that cannot execute. Applied to every list operation
at n=16, on a build carrying W1+W2+W3:

| operation | u20 | u40 | u60 | unwind tax |
|---|---:|---:|---:|---|
| `append` | 996 | 996 | 996 | no |
| `insert(0, …)` | 1,291 | 1,291 | 1,291 | no |
| `pop(0)` | 1,165 | 1,165 | 1,165 | no |
| `index` / `count` | 939 | 939 | 939 | no |
| `reverse` | 1,172 | 1,172 | 1,172 | no |
| `x in xs` | 1,894 | 1,894 | 1,894 | no |
| `xs == ys` | 3,671 | 3,671 | 3,671 | no |
| `copy` | 3,513 | 3,513 | 3,513 | no |
| `sorted(xs)` | 1,877 | 1,877 | 1,877 | no |
| **`remove`** | 2,122 | 2,382 | **2,642** | **yes** |
| **`sum`** | 2,637 | 4,337 | **6,037** | **yes** |
| **`max`** | 2,762 | 4,502 | **6,242** | **yes** |
| **`xs[1:]`** | 4,556 | 8,136 | **11,716** | **yes** |

Every operation W4 named — `index`, `count`, `in`, `==` — is flat. Two of them
are moot for a further reason: `[10, 20, 30].index(20)` is **already folded**
today (77 assignments, 1 VCC), and `x in [1, 2, 3]` costs 361 assignments.

### 6.3 What is actually left, and why it is one item

The four taxed rows share a single cause, confirmed from their unwinding
profiles: **a loop whose trip count symex cannot decide.**

| operation | the loop | why the bound does not fold |
|---|---|---|
| `sum` | injected Python model, line 265 | bound is the list length read at runtime |
| `max` | injected Python model, line 41 | same |
| `xs[1:]` | the slice lowering's own loop | bound is `__ESBMC_list_size(...)` |
| `remove` | `list.c:1208`, the **shift** loop | starts at the symbolic found-index, so `j < l->size - 1` stays open |

The `remove` row is worth stating plainly: #7361, the origin of this plan, split
that routine's search and shift loops. The **search** loop now runs exactly *n*
times. The **shift** loop still unwinds to the bound, because after the search
`j` begins at a symbolic index. The fix that named this defect class left a
residue of it behind.

**W4 (revised).** Give a list model's loop a bound symex can decide when the
list length is statically known. One change, three of the four operations, and
it is the same residual already flagged in #7436's description for slicing.
Landed as [#7440](https://github.com/esbmc/esbmc/pull/7440): `__ESBMC_list_size`
returned `l ? l->size : 0`, and a conditional on a pointer does not
constant-propagate even when `l` is a concrete address, so every `len()`-bounded
loop unwound to the bound. `sum` and `max` are flat afterwards.
[#7441](https://github.com/esbmc/esbmc/pull/7441) fixes the underlying
simplifier gap for C code, where the guard is syntactically an address-of.

### 6.4 `remove` — measured, and deliberately not fixed

The fourth taxed row does not fall to the same treatment, and the attempt is
recorded here so it is not repeated. Walking every slot and guarding the shift
on the found index does make the trip count concrete, and it works:

| `xs.remove(0)`, 8 elements | u20 | u40 | u60 |
|---|---:|---:|---:|
| start the shift at `i` (master) | 2,122 | 2,382 | 2,642 |
| start at 0, guard on `j >= i` | 2,041 | 2,041 | 2,041 |

A 28-case differential agreed throughout, and Mode C discharged both arms of the
new guard under Bitwuzla and Z3. **`regression/python/list_remove1` then failed,
and the failure was real.** That test removes from a five-element list at
`--unwind 4`: starting the shift at `i` needs `size - 1 - i` = 3 iterations,
starting at 0 needs `size - 1` = 4, so the model reports
`unwinding assertion loop 144` and only passes from `--unwind 5`.

That is a user-visible change to the bound every `remove` requires, not merely
a cost trade — an existing program with a tight `--unwind` would begin reporting
a spurious unwinding violation. Against it, `remove` is the *smallest* of the
four taxed rows at +13 assignments per unit of `--unwind`. The trade is bad, so
the change was reverted.

**What this row would actually need** is a decidable *start* index, not a
decidable length: `i` is the search's result and is genuinely symbolic whenever
the removed element is. Folding the search for a constant list and a constant
argument would give that, which is the original W4's constant-folding idea
arriving by another route — and it belongs in the same frontend-performance item
(§6.1), not here.

**The original W4 is not dropped, but it is not this plan.** Constant-folding
list operations remains worthwhile on its own terms; it belongs in a
frontend-performance item, not here, and should not be justified by this plan's
screening test.

---

## 7. W5 — `std::vector` falls off a cliff at its eleventh element

**Where.** `src/cpp/library/vector:324` (default constructor, `_capacity = 10`),
`:703` (`reserve`), `:792` (`push_back`).

The default constructor pre-allocates ten slots. While a vector stays inside
them, `reserve`'s body never runs and everything is concrete. The first time the
body does run, `push_back` stops being able to decide `_size == _capacity`, and
from then on **every `push_back` symbolically executes `reserve`'s element-copy
loop to the full `--unwind` bound**, whether or not the vector needs to grow.

**Measured**, eight `push_back`s at `--unwind 40`, varying only the reserved
capacity:

| shape | reserve-loop trace lines | assignments |
|---|---:|---:|
| `reserve(8)` + 8 `push_back` | 1 | 389 |
| `reserve(20)` + 8 `push_back` | 321 | **11,072** |
| `reserve(100)` + 8 `push_back` | 321 | **11,072** |
| `reserve(200)` + 8 `push_back` | 321 | **11,072** |

The threshold is **element count, not byte size**: `vector<char>`,
`vector<int>` and `vector<long long>` all fold at 10 and break at 11. Growing
without an explicit `reserve` shows the same wall — 381 assignments at n=8,
**3,697** at n=16, **42,249** at n=32 (`--unwind` n+4).

That constant propagation is the mechanism is confirmed directly: passing
`--no-propagation` to the cheap `reserve(8)` case reproduces the expensive
behaviour (1 trace line → 151).

**Root cause, reduced and then corrected.** The reduction removes C++,
templates, `std::vector` and `free` entirely. The first write-up of this section
blamed *a write through a pointer with two candidate objects*. A finer ablation
shows that is wrong on both counts: the write is not involved, and neither is
the number of candidate objects.

```c
struct V { int *buf; int size; };

v.buf = malloc(40); __ESBMC_assume(v.buf); v.size = 0;

int *nb = malloc(44); __ESBMC_assume(nb);
v.buf = nb;                     /* this line alone is the trigger */

v.size++;
int j = 0; while (j < v.size - 1) j++;   /* unwinds to the bound */
assert(j == 0);
```

Ablation, `--unwind 20`, counting all loop-unwinding lines:

| variant | unwinds |
|---|---:|
| `v.size++` alone | 0 |
| write through `v.buf`, constant index | 0 |
| write through `v.buf`, index is `v.size` | 0 |
| **`v.buf = <malloc'd pointer>`, no write at all** | **20** |
| `v.buf = &local` | 0 |
| `v.buf = NULL` | 0 |
| `v.buf = nb` where `nb` holds `&local` | 0 |

The value sets are identical in the folding and non-folding cases — NULL plus
one dynamic object either way — so "two candidate objects" was never the
mechanism.

**The mechanism.** `goto_symex_statet::constant_propagation` propagates a
struct only when *every* update in its `with` chain is itself propagatable
(`goto_symex_state.cpp:313`). A malloc'd address is not propagatable, an
`address_of` or NULL is. So storing a heap pointer into one field silently
drops the propagated constants of **every other field of that struct** — which
is why `_size` and `_capacity` stopped folding the moment `reserve` allocated.
The existing comment there already anticipates half of this ("a propagatable
pointer-typed field update ... does not poison the whole struct"); the gap is
that a heap pointer is never propagatable, so it still poisons.

**Fix, two levels, independently useful.**

1. *In the model.* `reserve` should `realloc` in place the way `assign()`
   already does — `vector:467` records that "realloc preserves the existing
   objects, so `[0, min(_size, n))` stays live". No fresh heap pointer is
   stored into `buf`, so the trigger never fires. Landed as
   [#7437](https://github.com/esbmc/esbmc/pull/7437).
2. *In symex.* Aggregate propagation is all-or-nothing, and it need not be: a
   `with` chain whose unknown field is left symbolic would still answer reads
   of the other fields correctly. **Now localised to one condition**, which is
   the advance over the previous draft — but still not a specified change. The
   obstacle is representational: propagated values stand in for a symbol later,
   so embedding a non-constant sub-expression is only sound if that
   sub-expression is a fixed L2 snapshot. Anyone taking this on should treat
   the soundness of that substitution as the whole problem, and should expect
   to need the full regression suite rather than a subset.

Level 1 is sufficient to fix `std::vector` and is the smaller change. Every
container model that reallocates is exposed to the same trigger and should be
audited once level 1 lands: `std::string`, `std::deque`, and the Python list
arena.

---

## 8. W6 — `memcpy`'s intrinsic gives up where `memcmp`'s does not

**Where.** `src/goto-symex/builtin_functions/memory_ops.cpp:662`,
`goto_symext::intrinsic_memcpy_impl`, against `:1076`, `intrinsic_memcmp`.

This is the amplifier under W1, W2 and W3: it is what turns "the fallback arm is
explored" into "the fallback arm costs `--unwind` iterations".

`intrinsic_memcpy_impl` bumps to the C byte loop as soon as `n` fails to
simplify to a constant:

```cpp
simplify(n_arg);
if (!is_constant_int2t(n_arg))
{
  bump_call(func_call, bump_name);   /* -> __memcpy_impl's byte loop */
  return;
}
```

`intrinsic_memcmp`, four hundred lines below in the same file, already solves
exactly this. For symbolic `n` it resolves both operands, takes
`min(avail1, avail2)` as a static bound, unrolls with each position guarded by
`i < n`, caps the unrolled width at `MAX_MEMCMP_UNROLL = 64`, and — the part
that makes it sound — claims `n <= avail` so a read past an object is still
reported as a dereference failure rather than silently dropped.

**Fix.** Port that path to `intrinsic_memcpy_impl`. It removes the `--unwind`
sensitivity from every operational-model copy at once, independently of whether
the individual models in W1–W3 are also fixed. `intrinsic_memmove` shares the
routine and is covered by the same change; whether `memset` needs the analogous
treatment is open.

---

## 9. Sequencing

| item | change is in | independent of | note |
|---|---|---|---|
| W6 | symex core | all others | widest reach for one change; do first |
| W1 | Python OM | W6 | two-line reorder, 36× on the measured case |
| W2 | Python OM + frontend | W6 | fix pattern already in the same file |
| W3 | Python OM | W2 | shares W2's mechanism |
| W5.1 | C++ OM | all others | `realloc` in `reserve` |
| W4 (revised) | Python models + frontend | W1–W3, W6 | decidable loop bounds; see §6.3 |
| W5.2 | symex core | W5.1 | investigation, not a specified change |

W6 and W1–W3 overlap deliberately: W6 makes the fallback cheap, W1–W3 stop
reaching it. Either alone is an improvement; both together are the correct end
state, because a model should not depend on an intrinsic's cap to be affordable.

The order above was written before anything was built, and §10 warned it was
ranked by reach rather than measured savings. W6 was ranked first for widest
reach; that was wrong for the Python models, because a pointer with more than
one candidate object defeats `memcmp_resolve_operand` before `n` is considered
at all — 8 unwindings with two candidates against 0 with one, at *constant* `n`.
Every Python list payload is its own `alloca`, so W6 does not fire there. W1–W3
sidestep the intrinsic instead of relying on it, which is why they carried the
Python gains.

---

## 10. What this plan does *not* establish

- **W1, W2, W3, W5.1 and W6 are implemented and in review; nothing is merged.**
  The measurements quoted inside each section are the survey's, taken before any
  patch; the before/after numbers live in the pull requests. W4 was re-scoped
  rather than implemented (§6) and W5.2 is still unlocalised, so two of the
  seven items remain proposals.
- **No verdict changes were observed or looked for.** Every program measured
  returns `VERIFICATION SUCCESSFUL` on master; this plan is about cost, and none
  of the items is a soundness finding.
- **W5's symex half (§7, level 2) is localised but unfixed.** It is the
  all-or-nothing aggregate rule at `goto_symex_state.cpp:313`. What is not
  established is whether propagating a partially-symbolic aggregate is sound;
  that, not finding the line, is the remaining work.
- **The regression-suite impact is unmeasured.** How much of the suite's wall
  time these items account for has not been quantified, so the ordering in §9
  was by reach and cost of the change, not by measured suite savings — which is
  why it needed the correction recorded there.
- **Per the repo's own rule, each item needs a regression *pair* and a
  mutation check.** For cost-only changes the pair pins the semantics the fix
  must preserve, and passes both before and after — so it cannot be
  mutation-checked by reverting the fix, and must be checked against a
  deliberately *wrong* version of the change instead.
- **The counted cost oracle this section called for now exists.** Each
  implemented item carries a test pinning `^Generated [0-9]{1,N} VCC\(s\)` at a
  bound chosen so the digit count separates the patched binary from the
  unpatched one, following the exact-count precedent in
  `regression/esbmc/descending_pointer_walk`. Pick N by measurement: at
  `--unwind 40` W5.1's control still fitted four digits and the oracle did not
  bite, which `--unwind 80` fixed.
