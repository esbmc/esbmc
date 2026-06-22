import numpy as np

# Regression: np.round(x, decimals) — the 2-argument form.  Before the fix
# this aborted with "Unknown operator: round" because `round` was not in the
# scalar-fold table and has no operator_map() entry, so it fell through to the
# BinOp path (the same trap copysign/fmax/fmin were already special-cased for).

# Positive decimals
assert np.round(2.567, 1) == 2.6
assert np.round(2.567, 2) == 2.57

# Zero decimals == rounding to integer
assert np.round(2.567, 0) == 3.0

# Negative decimals round to tens/hundreds
assert np.round(12345.0, -2) == 12300.0

# Round-half-to-even (banker's rounding), matching numpy
assert np.round(2.5, 0) == 2.0
assert np.round(3.5, 0) == 4.0
