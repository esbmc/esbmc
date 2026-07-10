# Falsification harness for math.floor (src/python-frontend/models/math.py).
#
# floor() is only defined for finite inputs: the model guards its body with
# `assert not isinf(x)` and `assert not isnan(x)`.  This harness drops the
# finiteness precondition, so the fully non-deterministic float may be NaN or
# +/-inf and the guard must fire.  It demonstrates that:
#   (a) the model correctly rejects non-finite inputs at runtime, and
#   (b) the finiteness precondition in harness_math_floor_ceil is necessary,
#       not incidental.
#
# WRONG SETUP (expected to be falsified):
#   F1: calling floor(x) with an unconstrained x reaches the model's
#       "Input cannot be NaN" / "Input cannot be infinity" assertion.
import math

x: float = nondet_float()

# Intentionally no __ESBMC_assume finiteness bound.
f: int = math.floor(x)

assert f <= x       # unreachable once the model's guard fails first
