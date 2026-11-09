import math

x = nondet_float()
__ESBMC_assume(x == 1.0)
assert math.erf(x) != 0.8427007929497149
