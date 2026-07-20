# JBMC end-to-end fixtures

These two fixtures are the first Java coverage on the `--binary` path. They pin
that ESBMC ingests a **JBMC-produced, fully lowered** goto binary and reaches
JBMC's own verdict, on a program exercising object allocation, inheritance,
virtual dispatch, `instanceof` and array allocation.

Source: `scripts/jbmc-poc-corpus/T4Virtual.java` and `T4VirtualFail.java`. The
two differ only in `assert s.area() == 25` versus `== 24`, so a verdict flip is
attributable to that assertion and nothing else. The counterexample names
`T4VirtualFail.java` line 19.

Background and measurements: `docs/jbmc-goto-binary-poc-plan.md`.

## Why a checked-in binary

Regenerating needs a JDK, `core-models.jar`, and a `jbmc` patched with
`--write-goto-binary` (`scripts/jbmc-write-goto-binary.patch`), which is not
upstream — so reproducing the pipeline in CI would mean building CBMC from
source on every run. Checking in the lowered binary is what
`docs/jbmc-goto-binary-poc-plan.md` §4.2 prescribes, and matches the 130-odd
existing `cbmc_*` fixtures here.

## Regenerating

```sh
git clone --depth 1 --branch cbmc-6.8.0 https://github.com/diffblue/cbmc.git
cd cbmc && git apply /path/to/esbmc/scripts/jbmc-write-goto-binary.patch
cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release -DWITH_JBMC=ON -GNinja
ninja -C build jbmc

cd /path/to/esbmc/scripts/jbmc-poc-corpus
javac -d classes T4Virtual.java
cbmc/build/bin/jbmc jbmcpoc.T4Virtual -cp classes \
    --no-refine-strings --write-goto-binary out.goto
goto-instrument --drop-unused-functions out.goto jbmc_virtual_dispatch.goto
```

Produced with **jbmc 6.8.0** (`GOTO_BINARY_VERSION` 6) and JDK 26.0.1. A
`GOTO_BINARY_VERSION` bump invalidates both fixtures at once; regenerate rather
than patch them.

## Two flags that are load-bearing

`--no-refine-strings` is required. JBMC enables string refinement by default,
which emits `cprover_associate_array_to_pointer_func` and the rest of the
`ID_cprover_string_*` family; ESBMC has no string-refinement backend and
declines. With the flag, those primitives disappear entirely and a string-free
program goes through. This narrows the plan's §5 "strings" stop to programs
that actually *use* strings.

`--force-malloc-success` is required because ESBMC models `malloc` as able to
fail and JBMC does not, so the `String[] args` allocation otherwise reports a
NULL dereference. That is a known pointer-model difference, not a JVM-semantics
gap.

## No `core-models.jar`, deliberately

The binaries are built without the models jar, which cuts them from ~440 KB to
~40 KB. That is sound *only* because these programs are self-contained: they
call no library method beyond `java.lang.Object`, so lazy loading resolves
everything they reach (see the Run 4 qualification in the plan). The fail
fixture still reports the violation at the right line, which is what shows the
assertions are live rather than stubbed.

**Do not copy this shortcut to a library-using program.** There, omitting the
jar silently produces a short model and a meaningless verdict — the failure
mode §5 calls the PoC's most serious soundness risk.

## `--unwind 6` is sound here, not a truncation

Unwinding assertions are left **enabled**. At `--unwind 2` ESBMC reports
`unwinding assertion loop 1` and fails; at 6 it passes with them on, so no loop
is truncated and the SUCCESSFUL verdict is not an artefact of the bound. Do not
add `--no-unwinding-assertions` to make a future failure go away — that flag
yields false SUCCESSFUL on truncated loops and would make this fixture
worthless.
