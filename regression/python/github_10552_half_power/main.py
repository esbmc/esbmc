import math
from esbmc import nondet_int, nondet_float, __ESBMC_assume

t = 2.0
assert t**0.5 == 1.4142135623730951

n = nondet_int()
__ESBMC_assume(1 <= n and n <= 1000)
r = int(n**0.5)
assert r * r <= n and n < (r + 1) * (r + 1)

x = nondet_float()
if x == 0.0:
    assert math.copysign(1.0, x**0.5) == 1.0
    assert math.copysign(1.0, math.pow(x, 0.5)) == 1.0

y = nondet_float()
if y == float("-inf"):
    assert y**0.5 == float("inf")
    assert math.pow(y, 0.5) == float("inf")
