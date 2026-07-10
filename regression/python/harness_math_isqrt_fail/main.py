# Falsification harness for math.isqrt (src/python-frontend/models/math.py).
#
# Validates that a violated postcondition is detected: it asserts that isqrt
# is an *exact* square root, which only holds for perfect squares.
#
# REQUIRES:
#   R1: n is a bounded non-negative integer.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: r*r == n.  False for any non-square n (e.g. n == 2 gives r == 1,
#       and 1*1 != 2).
import math

n: int = nondet_int()

__ESBMC_assume(n >= 0)
__ESBMC_assume(n <= 1000)

r: int = math.isqrt(n)

assert r * r == n            # F1 — falsifiable (non-perfect-square n)
