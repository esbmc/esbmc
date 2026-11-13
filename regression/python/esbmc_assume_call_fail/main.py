import math

y = nondet_float()
__ESBMC_assume(math.isinf(y))
assert y > 0.0
