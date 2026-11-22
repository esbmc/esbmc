# GitHub #10554: the assumed call constrains y, so y = -inf is a counterexample.
import math
from esbmc import nondet_float, __ESBMC_assume

y = nondet_float()
__ESBMC_assume(math.isinf(y))
assert y > 0
