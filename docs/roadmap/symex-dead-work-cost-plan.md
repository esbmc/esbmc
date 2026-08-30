# Plan — cost paid for paths that cannot execute (the #7361 defect class)

**Status:** survey complete, six work items identified, **none started**. No
patch has been written or verified against any of them.
**Origin:** [#7361](https://github.com/esbmc/esbmc/pull/7361), *"[python] Avoid
duplicated shifts in `list.remove()`"*, which split a search loop and a shift
loop that had been nested. This plan generalises that fix into a screening test
(§2) and applies it across the Python and C++ operational models and the
`memcpy`/`memcmp` intrinsics.
**Last updated:** 2026-08-30.

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

## 6. W4 — constant lists are not folded, though constant strings are

**Where.** `src/python-frontend/`. The constant folding added for `str` in
PRs #7373, #7374 and #7375 has no `list` counterpart.

A Python program whose lists are entirely literal is decidable at conversion
time. Strings already are; lists are handed to the C model and symbolically
executed in full. This is the widest single gap in the Python frontend's cost
profile, and it subsumes the common case of W1 and W3 without touching the
model.

`str` operations over constant operands, `--unwind 24`, assignments:

| operation | n=4 | n=8 | n=16 |
|---|---:|---:|---:|
| `s.find` / `s.count` / `"x" in s` | 78 | 78 | 78 |
| `s[1:]` / `s + s` | 79 | 79 | 79 |
| `s.replace(…)` | 86 | 90 | 98 |
| `s.split(…)` | 196 | 196 | 249 |
| `s.upper()` | 129 | 161 | 225 |

`list` operations over constant lists, `--unwind 60`, assignments:

| operation | n=4 | n=8 | n=16 |
|---|---:|---:|---:|
| `xs.append(…)` | 360 | 572 | 996 |
| `xs.index` / `xs.count` | 303 | 515 | 939 |
| `x in xs` | 454 | 870 | 1,894 |
| `xs == ys` | 707 | 1,503 | 3,671 |
| `xs.copy()` | 837 | 1,633 | 3,513 |
| `xs.extend(ys)` | 5,478 | 14,970 | 46,338 |
| `xs.sort()` | 15,683 | 99,609 | error |
| `xs[1:]` | 86,368 | 117,544 | timeout |

**The two tables use different bounds and their absolute values are not
comparable.** What is comparable is the shape: the `str` rows are flat in *n*,
the `list` rows are flat in nothing. (`timeout` is the 200 s cap; `error` is
`xs.sort()` at n=16 under `--unwind 60`.)

**Fix.** Extend constant folding to list operations whose receiver and arguments
are compile-time constants, in increasing order of difficulty: `index`, `count`,
`in`, `==`, slicing with constant bounds, `sort`.

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

**Root cause, reduced.** The reduction removes C++, templates, `std::vector` and
`free` entirely. What breaks propagation is a struct field acquiring a *second*
candidate dynamic object:

```c
struct V { int *buf; int size; };

v.buf = malloc(40); __ESBMC_assume(v.buf); v.size = 0;

int *nb = malloc(44); __ESBMC_assume(nb);
v.buf = nb;                     /* buf now has two candidate objects */

v.buf[v.size] = 7; v.size++;    /* a write through it ... */
int j = 0; while (j < v.size - 1) j++;   /* ... drops size's constant */
assert(j == 0);                 /* loop unwinds to the bound */
```

Ablation, `--unwind 20`, counting all loop-unwinding lines:

| variant | unwinds |
|---|---:|
| baseline, no reallocation | 0 |
| second `malloc`, result unused | 0 |
| **second `malloc`, assigned to `v.buf`** | **20** |
| `free(v.buf)` then `malloc` into `v.buf` | 20 |
| full free-and-swap | 20 |

`free` is not involved. The write through the two-candidate pointer clobbers the
propagated constant of `size`, an *unrelated field of the same struct*.

**Fix, two levels, independently useful.**

1. *In the model.* `reserve` should `realloc` in place the way `assign()`
   already does — `vector:467` records that "realloc preserves the existing
   objects, so `[0, min(_size, n))` stays live". One dynamic object, no copy
   loop, and the trigger in the reduction never fires.
2. *In symex.* A write through a pointer whose value set resolves to a single
   known object should not drop propagation for sibling fields. **This half is
   not yet traced to a line in symex** — the reduction above localises the
   trigger, not the code that acts on it. Treat level 2 as a scoped
   investigation, not a specified change.

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
| W4 | Python frontend | W1, W3 | largest, subsumes their common case |
| W5.2 | symex core | W5.1 | investigation, not a specified change |

W6 and W1–W3 overlap deliberately: W6 makes the fallback cheap, W1–W3 stop
reaching it. Either alone is an improvement; both together are the correct end
state, because a model should not depend on an intrinsic's cap to be affordable.

---

## 10. What this plan does *not* establish

- **No fix has been written, built or verified.** Every "fix" above is a
  proposal derived from the measurement, not a tested change.
- **No verdict changes were observed or looked for.** Every program measured
  returns `VERIFICATION SUCCESSFUL` on master; this plan is about cost, and none
  of the items is a soundness finding.
- **W5's symex half (§7, level 2) is unlocalised.** The reduction identifies the
  trigger; it does not identify the code that responds to it.
- **The regression-suite impact is unmeasured.** How much of the suite's wall
  time these items account for has not been quantified, so the ordering in §9 is
  by reach and cost of the change, not by measured suite savings.
- **Per the repo's own rule, each item needs a regression *pair* and a
  mutation check.** For cost-only changes the pair pins the semantics the fix
  must preserve, not the cost; a cost regression needs its own counted oracle,
  which does not yet exist in `regression/`.
