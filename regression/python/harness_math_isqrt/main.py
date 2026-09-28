# Verification harness for math.isqrt (src/python-frontend/models/math.py).
#
# math.isqrt(n) returns the integer square root of n: the largest integer r
# such that r*r <= n.  The model uses Newton's method.
#
# REQUIRES:
#   R1: n is a bounded non-negative integer (isqrt raises for n < 0, and a
#       bound keeps Newton's iteration fully unwound).
#
# ENSURES:
#   E1: r >= 0                                   [square root is non-negative]
#   E2: r*r <= n                                 [r is a lower bound]
#   E3: n < (r+1)*(r+1)                          [r is the *largest* such root]
#
# E2 and E3 together pin r to the unique correct value and rule out
# off-by-one errors in the model's convergence.
import math

n: int = nondet_int()

__ESBMC_assume(n >= 0)
__ESBMC_assume(n <= 1000)

r: int = math.isqrt(n)

assert r >= 0  # E1
assert r * r <= n  # E2
assert n < (r + 1) * (r + 1)  # E3
