# Verifying a PCIe driver family with ESBMC

A template for applying the approach in `regression/cxl/` and
`regression/cxl-linux/` to another driver family — NVMe, USB, a vendor NIC.
The MMIO, DMA and IRQ models in `src/c2goto/library/cxl_driver.c` are generic
PCIe infrastructure and are reusable as-is; what follows is the rest of it.

Everything below is drawn from building the CXL suite. The traps in §3 each
cost real time there, and every one of them is a mistake that produced a green
test that established nothing.

---

## 1. Decide what "verified" will mean before writing anything

Two very different things are worth doing, and conflating them wastes effort:

| | Synthetic suite (`regression/cxl/`) | Real-source harnesses (`regression/cxl-linux/`) |
|---|---|---|
| Compiles | your model | actual driver `.c` files |
| Needs | nothing | a configured kernel tree |
| Runs in CI | yes, cheaply | yes, but needs the tree prepared first |
| Proves | the *pattern* is checkable, and ESBMC finds this bug class | something about the driver |
| Cost per test | minutes | hours, mostly on stub surface |

Start synthetic to build the models and learn the bug classes; move to real
source for the functions that matter. **Only the second kind says anything
about the driver.** The CXL suite has 84 synthetic tests and 4 verified real
functions; the 84 are useful, but they are not the same claim.

## 2. Structure

```
src/c2goto/headers/<distro>/<kernel>/include/linux/<family>.h   API declarations
src/c2goto/library/<family>_driver.c                            model bodies
regression/<family>/<test>/main.c + test.desc                   synthetic tests
regression/<family>-linux/harness_*.c                           real-source harnesses
scripts/<family>_model_coverage.py                              the metric that matters
```

Two rules that are not optional:

- **Every model function must be declared in a header the tests include.** A
  function that is defined but not declared is reached through an implicit
  declaration returning `int`. This happened twice in the CXL work —
  `__kmalloc()` and `readsl()` — and in both cases the model's own code had
  never executed.
- **Tests must not redefine model functions.** A test that provides its own
  `foo()` shadows the model's and verifies itself. Roughly two thirds of the
  early CXL suite did this, which is why `cxl_driver.c` could call a `static`
  function through an implicit declaration for its entire existence unnoticed.

## 3. Traps

Each of these produced a passing test that checked nothing, or a result that
could not be reproduced.

**`__ESBMC_assume()` constrains values, not unwinding.** `assume(n <= 2)`
followed by `for (i = 0; i < n; i++)` still unwinds forever — symex walks the
loop syntactically. Give such tests an explicit `--unwind`, one past the
assumed maximum, and leave unwinding assertions **on** so the bound is proved
rather than taken on trust. A CXL harness reached iteration 4562 and exhausted
12 GB before this was understood.

**Loops bounded by a mutable counter go symbolic.** `for (i = 0; i < g_count; i++)`
over a static counter stops being concrete as soon as any failure path merges.
Prefer a compile-time bound and an occupancy flag over a running count.

**Never size an allocation by a symbolic count.** `kmalloc(n * sizeof(T))` with
symbolic `n` makes every later `a[i]` a bounds check against a
nondeterministically sized object. A four-element walk did not solve in 280 s;
allocating the array at its maximum and reporting a symbolic "how many are
populated" brought the same test to 9 s.

**`__VERIFIER_nondet_int() % N` yields negative values.** C's `%` keeps the
sign of the dividend. This reached an array index, an enum, and a device-type
field in the CXL model. Draw an unsigned nondet and constrain it with
`__ESBMC_assume()` instead.

**Side tables keyed on a pointer are expensive.** Looking per-device state up
by scanning an array costs a symbolic pointer comparison per slot per call; a
seven-call test did not solve in 300 s. Put the state in the device struct, as
the kernel does. The same test then solved in 0.14 s.

**Headers must be self-contained.** A header that uses `size_t` without
including `<stddef.h>` compiles fine in every test that happens to include it
first. The one that doesn't fails to parse.

**A negative test that does not fail is not evidence of correctness — it is
evidence the test is wrong.** Two CXL negative harnesses passed vacuously: one
because a stub returned a nondet pointer instead of invoking the match
callback, another because a fixed 16-entry array was larger than any value the
"overflowing" field could encode. Patch-and-reverify every failing test: fix
the modelled bug in a scratch copy and confirm it flips to SUCCESSFUL.

## 4. Declare what each test checks

ESBMC's memory-leak, overflow, data-race and deadlock checks are **opt-in**. A
`test.desc` with an empty flags line runs on defaults and guards none of them.
The first time `--memory-leak-check` was enabled across the CXL suite it found
a CWE-401 leak in a test committed green the same day.

```
THOROUGH
main.c
--memory-leak-check --overflow-check --unsigned-overflow-check
^VERIFICATION SUCCESSFUL$
```

Line 3 is load-bearing in a second way: if omitted, the expected-output regex
slides onto it and is passed to ESBMC as arguments. The test then has no regex
to match and passes without verifying anything.

## 5. Measure coverage, not test count

Test count measures output. The number worth tracking is **how many modelled
functions any test actually calls**, which `scripts/cxl_model_coverage.py`
computes — copy it and change the paths.

The CXL suite went from 21% to 100% model coverage. Reaching those functions
for the first time found **16 defects in the model itself**, three of which
meant a function could not have worked for any caller: a divide-by-zero on a
table nothing ever populated, an entire error-reporting layer keyed on the
wrong thing, and a function defined but never declared. None of them were
found by adding tests; all of them were found by *reaching code*.

## 6. Real-source harnesses

The step that makes the work mean something, and the expensive one.

- Compile one driver `.c` from a configured kernel tree, with the generated
  headers on the include path. `scripts/cxl_prepare_kernel.sh` does the
  preparation and is family-independent — point it at any pinned version.
- **Pin the kernel version.** Harnesses assert properties of specific functions;
  tracking mainline turns an unrelated upstream refactor into a CI failure here.
- **Every stub is an assumption, and belongs in writing.** Enumerate them.
  `device_find_child()` must invoke its callback rather than return a nondet
  pointer, or the property under test is never exercised.
- **Watch for config options probed from the host compiler.** The CXL harness
  results were irreproducible outside one build directory because
  `CONFIG_CC_HAS_COUNTED_BY` depends on whichever compiler ran `defconfig`, and
  ESBMC cannot convert the resulting `CountAttributedType`. Anything of that
  shape must be neutralised in the preparation script and recorded as an
  assumption.
- **Put the expected verdicts in a script, not a prose table.** Writing
  `run_all.sh` immediately found that the documented invocation could not work
  and that two verdicts had been recorded without the flags that produce them.

## 7. Recurring bug shapes

Across 35 bug-detecting CXL tests, essentially every modelled driver defect is
one of two things. Look for these first in a new family:

**A dropped return value.** The call is made, the result discarded, and the
driver proceeds as though it succeeded. Requesting a device transition is not
the same as it having happened; sending a mailbox command is not receiving a
reply; asking to unlock is not being unlocked.

**A confused encoding.** A register field is treated as the value it encodes.
The CXL DVSEC `HDM_COUNT` field is two bits, so it can report 3 while the
array it indexes holds 2; the CFMWS interleave-ways field is an encoding where
8 means *three* ways. A driver that reads either as a count writes out of
bounds.

## 8. Checklist

- [ ] Model functions all declared in a header; no test redefines one
- [ ] `test.desc` line 3 declares the properties each test checks
- [ ] Every failing test patch-and-reverified
- [ ] Explicit `--unwind` wherever a bound comes from an assumption, assertions on
- [ ] No allocation sized by a symbolic count
- [ ] Coverage measured and reported alongside the test count
- [ ] Real-source harness verdicts encoded in a script, with resource caps
- [ ] Stub surface written down as the assumption list it is
