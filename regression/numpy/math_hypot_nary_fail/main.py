import math

x = nondet_float()
__ESBMC_assume(x >= 1.0 and x <= 2.0)
assert math.hypot(x, 2.0, 2.0) < 3.0
