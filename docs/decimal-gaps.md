# ESBMC Decimal Model: Gaps vs CPython

This document describes the differences between ESBMC's `decimal.Decimal` operational model and CPython's `decimal` module.

## What is Implemented

### Construction
- `Decimal("3.14")`, `Decimal(42)`, `Decimal(-5)`, `Decimal(3.14)`, `Decimal()`
- Special values: `Decimal("NaN")`, `Decimal("sNaN")`, `Decimal("Infinity")`, `Decimal("-Infinity")`
- All construction is resolved at preprocess time using CPython's own `decimal` module

### Class Methods
- `Decimal.from_float()` — class method construction from float values

### Comparisons
- `==`, `!=`, `<`, `<=`, `>`, `>=` — full IEEE 854 semantics
- NaN is unordered (all comparisons return False)
- `-0 == +0` returns True
- Exponent normalization (`1.0 == 1.00`)

### Arithmetic
- `+`, `-`, `*`, `/`, `//`, `%` — binary operators
- `-d` (negation), `abs(d)`, `+d` (unary plus)
- NaN propagation (sNaN → qNaN conversion)
- Infinity arithmetic (`Inf + Inf = Inf`, `Inf - Inf = NaN`, `Inf * 0 = NaN`)
- Division precision: 28 digits (CPython default)

### Reverse Operators
- `__radd__()`, `__rsub__()`, `__rmul__()`, `__rtruediv__()`, `__rfloordiv__()`, `__rmod__()` — called when the left operand is an int (e.g., `3 + Decimal("1.5")`)

### Boolean Conversion
- `__bool__()` — Decimal in boolean contexts (`if d:`, `while d:`)
- Zero is falsy, all non-zero (including NaN and Infinity) are truthy

### Numeric Conversion
- `__int__()` — convert Decimal to int via `int(d)` (truncates toward zero)

### Query Methods
- `is_nan()`, `is_snan()`, `is_qnan()`, `is_infinite()`, `is_finite()`, `is_zero()`, `is_signed()`
- `is_normal()`, `is_subnormal()` — uses CPython default Emin=-999999

### Copy Operations
- `copy_abs()`, `copy_negate()`, `copy_sign(other)`

### Other Methods
- `adjusted()` — returns adjusted exponent
- `compare()`, `compare_signal()` — return Decimal (-1, 0, 1)
- `max(other)`, `min(other)` — NaN-aware selection
- `fma(other, third)` — fused multiply-add (exact arithmetic)

## What is NOT Implemented

### String Conversion
- `__str__()`, `__repr__()` — the model has no string type; Decimal values cannot be printed or converted to strings

### Float Conversion
- `__float__()` — converting Decimal to Python float

### Hashing
- `__hash__()` — needed for using Decimal as dict keys or in sets

### Context-Dependent Methods
- `normalize()`, `to_integral_value()`, `quantize()` — depend on the decimal context (precision, rounding mode)
- `sqrt()`, `exp()`, `ln()`, `log10()` — transcendental functions requiring iterative algorithms

### Other Missing Methods
- `as_tuple()` — returns `DecimalTuple(sign, digits, exponent)` (requires tuple support)
- `to_eng_string()` — engineering notation string

### Context Operations
- `decimal.getcontext()`, `decimal.localcontext()` — context management
- Context attributes: `prec`, `rounding`, `Emin`, `Emax`, `traps`, `flags`

## Workarounds

| CPython Pattern | ESBMC Workaround | Notes |
|----------------|-----------------|-------|
| `str(d)` | N/A | No string support in model |
| `d.normalize()` | Use as-is | Values are exact; normalization is cosmetic |
