import math

assert math.isnan(math.hypot(1.4e308, 1.4e308, math.nan))
assert math.isinf(math.hypot(1.4e308, math.nan, math.inf))
x = nondet_float()
__ESBMC_assume(x == 1.4e308)
assert math.isnan(math.hypot(x, x, math.nan))
assert math.isinf(math.hypot(math.nan, x, -math.inf))
