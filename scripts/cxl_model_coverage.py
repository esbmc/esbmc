#!/usr/bin/env python3
"""Report how much of the CXL operational model the cxl suite actually exercises.

Test count is not a coverage figure. A test that declares its own structs and
driver functions verifies code written inside the test file, and would still
pass if src/c2goto/library/cxl_driver.c were deleted. That is not hypothetical:
the model called a static __kmalloc() through an implicit declaration for its
entire existence, because no test had ever executed the model's allocation
path.

Two numbers are reported.

  Linked   - the test's GOTO program contains functions defined in
             cxl_driver.c. Semantic, but it over-reports badly and should not
             be tracked as the coverage number: linking is per translation
             unit, so needing one primitive drags in the whole model. It
             currently reads 33 of 37 while only 13 tests call the model for
             anything. Kept because a test that does *not* link the model
             provably cannot exercise it.

  Called   - the test calls a modelled function and does not define that
             function itself. This is a *static approximation*: it does not
             prove the call is reachable, only that the test does not shadow
             the model with a local definition. Shadowing is the failure mode
             that actually occurred here (cxl_mem_attach_01, cxl_port_enum_01
             and the HDM alignment tests all do it), which is what this
             catches.

Usage:
    scripts/cxl_model_coverage.py [--esbmc PATH] [--quiet]

Without --esbmc the linked figure is skipped and only the static analysis runs.
Exits non-zero if no test exercises the model at all.
"""

import argparse
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
MODEL = REPO / "src/c2goto/library/cxl_driver.c"
TESTS = REPO / "regression/cxl"

# A definition at column 0 that is not static and not a bare declaration.
DEF = re.compile(r"^(?!static\b)[A-Za-z_][\w \t\*]*?(\w+)\s*\($")
DEF_ONELINE = re.compile(r"^(?!static\b)[A-Za-z_][\w \t\*]*?(\w+)\s*\([^;]*\)\s*$")
# A GOTO definition header: "name (c:@F@name):"
GOTO_DEF = re.compile(r"^(\w+) \(c:@F@\1\):")


def model_functions():
    """Exported function names defined in the operational model."""
    names = []
    for line in MODEL.read_text(encoding="utf-8").splitlines():
        if line.startswith((" ", "\t", "#", "/", "*", "}")) or line.endswith(";"):
            continue
        m = DEF.match(line) or DEF_ONELINE.match(line)
        if m:
            names.append(m.group(1))
    return sorted(set(names))


def defines_locally(src, fn):
    return re.search(rf"^[A-Za-z_][\w \t\*]*\b{re.escape(fn)}\s*\(", src, re.M) is not None


def calls(src, fn):
    return re.search(rf"\b{re.escape(fn)}\s*\(", src) is not None


def linked(esbmc, main_c):
    """True if the test's GOTO program contains any model-defined function."""
    try:
        p = subprocess.run(
            [esbmc, str(main_c), "--goto-functions-only"],
            capture_output=True, text=True, timeout=300,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    # The dump goes to stderr.
    present = {m.group(1) for m in (GOTO_DEF.match(l) for l in p.stderr.splitlines()) if m}
    return bool(present & set(model_functions()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esbmc", help="esbmc binary; enables the linked figure")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    fns = model_functions()
    tests = sorted(d for d in TESTS.iterdir() if (d / "main.c").exists())

    covered = set()
    exercising = []
    for d in tests:
        src = (d / "main.c").read_text(encoding="utf-8")
        hit = {f for f in fns if calls(src, f) and not defines_locally(src, f)}
        if hit:
            exercising.append(d.name)
            covered |= hit

    print(f"model functions:              {len(fns)}")
    print(f"  called by some test:        {len(covered)} ({100 * len(covered) // max(len(fns), 1)}%)")
    print(f"tests:                        {len(tests)}")
    print(f"  calling into the model:     {len(exercising)}")

    if args.esbmc:
        n = sum(1 for d in tests if linked(args.esbmc, d / "main.c"))
        print(f"  linking the model (GOTO):   {n}")

    if not args.quiet:
        uncovered = [f for f in fns if f not in covered]
        if uncovered:
            print("\nmodelled but never called by any test:")
            for f in uncovered:
                print(f"  {f}")

    return 0 if covered else 1


if __name__ == "__main__":
    sys.exit(main())
