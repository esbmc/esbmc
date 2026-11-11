import math

x = nondet_float()
__ESBMC_assume(x >= 0.5 and x <= 1.0)
y = math.erf(x)
assert y > 0.52 and y < 0.85
c = math.erfc(x)
assert c > 0.15 and c < 0.48
assert math.erf(0.0) == 0.0
assert math.copysign(1.0, math.erf(-0.0)) == -1.0
assert math.erf(math.inf) == 1.0 and math.erf(-math.inf) == -1.0
assert math.erfc(0.0) == 1.0
assert math.erfc(math.inf) == 0.0 and math.erfc(-math.inf) == 2.0
assert math.isnan(math.erf(math.nan)) and math.isnan(math.erfc(math.nan))
