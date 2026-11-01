# math.trunc() of a float wider than 32 bits keeps its value (#9395).
import math

assert math.trunc(3000000000.5) == 3000000000
a: float = nondet_float()
__ESBMC_assume(a >= -1e10 and a <= -3e9)
assert math.trunc(a) <= -3000000000
