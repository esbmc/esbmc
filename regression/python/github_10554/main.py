# GitHub #10554: a call passed directly to __ESBMC_assume crashed in SMT conversion.
import math
from esbmc import nondet_float, __ESBMC_assume

y = nondet_float()
__ESBMC_assume(math.isinf(y))
assert y == y
