import math


def positive(v: int) -> bool:
    return v > 0


y = nondet_float()
__ESBMC_assume(math.isinf(y))
assert y == y
assert y != 0.0

n = nondet_int()
__ESBMC_assume(positive(n))
assert n >= 1
