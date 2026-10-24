# Debugging Verification Issues

When ESBMC produces an unexpected VERIFICATION FAILED or SUCCESSFUL result, use
these techniques:

**1. Inspect the GOTO program** — Use `--goto-functions-only` to dump the
intermediate GOTO representation. This reveals exactly what code ESBMC is
verifying, including how frontend constructs are lowered:

```sh
esbmc test.py --unwind 9 --goto-functions-only 2>&1 | grep -A50 "python_user_main"
```

Look for the `python_user_main` function to see how Python source maps to GOTO
instructions (ASSIGN, FUNCTION_CALL, ASSERT). This is especially useful for
catching compile-time optimizations that incorrectly pre-resolve values.

**2. Bisect with simpler test cases** — When a test fails, create variants that
isolate the problem.

**3. Read the counterexample trace** — ESBMC's `[Counterexample]` section shows
the state at each step. Track field assignments in structs (e.g., `PyObject`'s
`.value`, `.type_id`, `.size`) through the trace.

**4. Key files for Python frontend debugging:**

- `src/python-frontend/python_converter.cpp` — Main expression/statement
  conversion
- `src/python-frontend/python-list/` — List operations (split by concern:
  construction, mutation, access, query, string ops, comprehension, set ops,
  type map, type inference)
- `src/python-frontend/function_call_expr.cpp` — Method call handling
- `src/c2goto/library/python/list.c` — C operational model for list operations

**5. Hypothesis tests** — Property-based tests in `unit/python-frontend/` test
ESBMC's models against CPython. Run with:
`uv run python -m pytest unit/python-frontend/ -v`
