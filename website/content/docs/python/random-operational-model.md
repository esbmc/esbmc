---
title: Random Module
weight: 6
---

ESBMC models Python's `random` module using nondeterministic values constrained by `__ESBMC_assume`. This replaces true pseudo-randomness with symbolic nondeterminism, allowing ESBMC to explore all values a random call could produce and verify properties that must hold for any outcome.

## Supported Functions

### `random.randint(a, b)`

Returns a nondeterministic `int` N such that `a ≤ N ≤ b` (inclusive on both ends, matching CPython semantics).

```python
import random
x: int = random.randint(1, 10)
assert x >= 1 and x <= 10  # always holds
```

### `random.random()`

Returns a nondeterministic `float` in the half-open interval `[0.0, 1.0)`.

```python
import random
x: float = random.random()
assert x >= 0.0 and x < 1.0  # always holds
```

### `random.uniform(a, b)`

Returns a nondeterministic `float` in `[min(a,b), max(a,b)]`. Handles reversed bounds (`a > b`) by swapping the constraint direction.

```python
import random
x: float = random.uniform(2.5, 5.0)
assert x >= 2.5 and x <= 5.0  # always holds
```

### `random.getrandbits(k)`

Returns a nondeterministic non-negative `int` in `[0, 2**k − 1]`. Raises `ValueError` if `k < 0`; returns `0` when `k == 0`.

```python
import random
x: int = random.getrandbits(8)
assert x >= 0 and x <= 255  # always holds
```

### `random.randrange(stop)` / `random.randrange(start, stop[, step])`

Returns a nondeterministic integer selected from the same range as `range(start, stop, step)`. Raises `ValueError` for a zero step or an empty range.

```python
import random
x: int = random.randrange(10)          # in [0, 9]
y: int = random.randrange(0, 20, 2)   # one of 0, 2, 4, …, 18
assert x >= 0 and x < 10
assert y % 2 == 0 and y >= 0 and y < 20
```

## Unsupported Functions

`random.choice()`, `random.shuffle()`, `random.sample()`, `random.seed()`, and other functions not listed above are not yet modelled.

## Modelling Approach

Each supported function calls an ESBMC nondeterministic primitive (`nondet_int()` or `nondet_float()`) and immediately constrains the result with `__ESBMC_assume()`. From the solver's perspective the return value is a free variable bounded by the assume — equivalent to quantifying over all values in the range.

This means:

- **Assertions that must hold for all outcomes** are verified soundly.
- **`random.seed()` has no effect** — the model is stateless; seeding cannot make results deterministic.
- **Distribution is irrelevant** — ESBMC considers every value in the range equally, so statistical properties (expected value, variance) cannot be verified with this model.
