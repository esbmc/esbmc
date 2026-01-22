---
title: Random OM
---

## Addition of `random` Module Stub

**File:** `src/python-frontend/models/random.py`

- Implemented a stub for the `random` module.
- Defined the `randint(a: int, b: int) -> int` method using `nondet_int()`.
- Ensured that the generated value satisfies the constraint `a <= value <= b` using `__ESBMC_assume()`.

## Integration with Python Converter

**File:** `src/python-frontend/python_converter.cpp`

- Modified the `model_files` list to include the `random` module for operational modeling.
- Ensured the `random` module is loaded during conversion.

## Type Utility Adjustments

**File:** `src/python-frontend/type_utils.h`

- Updated the `is_builtin_function()` method to recognize `randint` as a built-in function.
- This ensures correct type handling and integration within ESBMC.

## Rationale

The addition of the `random` operational model allows ESBMC to verify Python programs that use `random.randint()`, enabling more comprehensive verification of probabilistic behaviors.

## Testing

- Verified functionality by running test cases that generate random numbers within a specified range.
- Confirmed that ESBMC correctly assumes constraints on the nondeterministic integer values.

## Future Work

- Extend support for additional functions in the `random` module (e.g., `random`, `uniform`).
- Improve constraint modeling to enhance verification efficiency.
