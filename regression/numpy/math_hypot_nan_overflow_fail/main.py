import math

x = nondet_float()
__ESBMC_assume(x == 1.4e308)
assert not math.isnan(math.hypot(x, x, math.nan))
