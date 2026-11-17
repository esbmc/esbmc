# GitHub #10554: a user function call passed directly to __ESBMC_assert.
from esbmc import nondet_int, __ESBMC_assert


def above_three(a: int) -> bool:
    return a > 3


y = nondet_int()
__ESBMC_assert(above_three(y), "y is above three")
