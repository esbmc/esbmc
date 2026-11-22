import math

x = nondet_float()
__ESBMC_assume(x == 1.0)
assert math.erfc(x) != 0.15729920705028513
