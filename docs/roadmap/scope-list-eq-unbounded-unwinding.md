# Scope — `__ESBMC_list_eq` unwinds without bound

> Opened 2026-08-17. Blocks Phase 3 of `frontends-to-irep2.md` (the Python
> `python_adjust` flip), whose G4 whole-corpus census cannot classify a test it
> cannot run. Supersedes the diagnosis in
> `scope-coupled-arith-assign-conversion.md` §19.2–§19.7.

## 1. The symptom, and what it is not

```python
m: list = ["y"]
assert m == ["y"]
```

No verdict, default flags, no loops in the program. §19.2 recorded this as "the
equality comparison of a list whose elements are **strings** does not
terminate", because integer and float lists return instantly.

**That attribution is wrong.** Integer, float and boolean lists are fast because
the frontend never calls the model for them: `python-list/list_query.cpp` elides
list equality when it has concrete type-map entries for both operands
(`:116-117`, and the nested-list elision at `:193-195`). Strings do not take
that path — that is the only thing special about them.

Measured: `--goto-functions-only` shows a `list_eq(...)` call in
`python_user_main` for the string program and **none** for the float or boolean
ones. Forcing the call with an integer list, by comparing through a parameter
whose type map is not concrete, reproduces the hang:

```python
def g(x: list) -> bool:
    return x == [1]

m: list = [1]
assert g(m)          # no verdict either
```

So the defect is: **any program that actually reaches `__ESBMC_list_eq`
diverges**, whatever its element type. The elisions have been hiding it, and
they are the reason the corpus looked healthy.

## 2. Why it diverges

`__ESBMC_list_eq` (`src/c2goto/library/python/list.c:298`) walks an explicit
worklist:

```c
int top = 1;
while (top > 0) { ... top--; ... top++; ... }
```

`top` is written on data-dependent paths, so the guard never simplifies to false
structurally, and unbounded symex has nothing to stop it. It is not a
non-terminating loop: with `--unwind 4` and unwinding assertions **on**, the
program verifies SUCCESSFUL with no unwinding-assertion violation, so every
reachable execution exits within three iterations. §19.7 established this and it
is reconfirmed here.

## 3. Why this counts as a defect rather than ordinary BMC

ESBMC is a bounded model checker, and a C loop whose trip count is not
structurally decidable ordinarily requires `--unwind`. That argument does not
apply here: **the Python program contains no loop.** The unwinding is introduced
entirely by an operational model the user did not write and cannot see, so
"pass `--unwind`" is not an instruction that can reasonably be given. The
principle this scope asserts is that *an operational model must not introduce
unbounded unwinding for a program that has none.*

## 4. The fix, and the decision it needs

The model already has the shape of the answer. Nested descent is bounded by
`depth_limit`, and exceeding it is **reported** rather than silently truncated
(`list.c:381-390`):

```c
if ((size_t)top >= depth_limit)
{
  __ESBMC_assert(0, "list comparison depth limit exceeded "
                    "(use --python-list-compare-depth to increase)");
  return false;
}
```

The same treatment applied to the worklist as a whole — a counter with a
concrete bound, asserting when exhausted — makes `iter < CAP && top > 0`
structurally false after `CAP` unwindings, and symex terminates.

**What needs deciding, and why this scope stops here.** The bound is a direct
cost: symex unwinds the loop body `CAP` times for *every* comparison that
reaches the model, because `top > 0` stays satisfiable. So `CAP` trades
completeness for termination, and it adds a failure mode to a shared operational
model. A sensible default is small (a list of a few elements needs 2–3
iterations per frame), but the number is a judgement call for whoever owns the
model, not something to pick unilaterally. Candidates:

| option | effect |
|---|---|
| fixed `CAP` constant | simplest; every comparison costs `CAP` unwindings |
| `CAP` derived from `--python-list-compare-depth` | reuses the existing knob, so the escape hatch is already documented |
| widen the frontend elision to strings | §19.6's proposal; hides the defect again rather than fixing it, and leaves the parameter case of §1 broken |

The third is what §19.6 recommended and §19.7 already rejected; it is listed
because it is cheap and would unblock Phase 3's census on its own, at the cost
of leaving the model unusable.

## 5. What this scope contributes now

- The attribution in §19.2 is corrected: not string-specific.
- A second reproducer that does not involve strings at all (§1).
- The reason the corpus never caught it: the elisions (§1).
- The argument that makes it a defect rather than a missing `--unwind` (§3).
- A regression test pinning the model's *correctness* at a bound, so the fix
  above cannot silently change what the comparison computes
  (`regression/python/list_eq_bounded`).

No unbounded test is added: it would consume the full 120 s CI cap on every run
and report nothing that this document does not.
