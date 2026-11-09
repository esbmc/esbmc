import math

x = nondet_float()
__ESBMC_assume(x == 0.0)
assert math.erfc(x) != 1.0
