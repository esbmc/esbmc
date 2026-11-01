# math.trunc() of a float wider than 32 bits keeps its value (#9395).
import math

a: float = nondet_float()
__ESBMC_assume(a >= 3e9 and a <= 1e10)
assert math.trunc(a) < 2147483648
