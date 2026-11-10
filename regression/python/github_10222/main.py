# Issue #10222: the ZeroDivisionError of zero to a negative power is catchable,
# and no other base or exponent raises it.
import math
import numpy as np

try:
    x: float = 0.0 ** -1
except ZeroDivisionError:
    x = 5.0
assert x == 5.0

assert 0.0 ** -math.inf == math.inf
assert np.power(0.0, -1.0) == np.inf

n: int = nondet_int()
b: float = nondet_float()
__ESBMC_assume(b != 0.0 or n >= 0)
__ESBMC_assume(n > -3 and n < 3)
y = b ** n
z = pow(2, -1)
assert z == 0.5
