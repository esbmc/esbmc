import math

x = nondet_float()
__ESBMC_assume(x >= 1.0 and x <= 2.0)
assert math.hypot() == 0.0
assert math.hypot(-3.0) == 3.0
assert math.hypot(x) == x
r = math.hypot(1.0, 2.0, 2.0)
assert r > 2.9999 and r < 3.0001
assert math.hypot(x, 2.0, 2.0) >= 2.0 and math.hypot(x, 2.0, 2.0) < 4.5
assert math.hypot(math.nan, 1.0, math.inf) == math.inf
n = math.hypot(1.0, math.nan, 2.0)
assert n != n
assert math.hypot(0.0, -0.0, 0.0) == 0.0
