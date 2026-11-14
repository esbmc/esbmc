# GitHub #10554: a user function call passed directly to __ESBMC_assume.
from esbmc import nondet_int, __ESBMC_assume


def above_three(a: int) -> bool:
    return a > 3


y = nondet_int()
__ESBMC_assume(above_three(y))
assert y > 3
