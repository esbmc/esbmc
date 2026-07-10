# Verification harness for math.factorial (src/python-frontend/models/math.py).
#
# math.factorial(n) returns n! via an iterative product.
#
# REQUIRES:
#   R1: n is a bounded non-negative integer (factorial raises for n < 0; the
#       bound keeps the product loop fully unwound and avoids overflow).
#
# ENSURES:
#   E1: factorial(n) >= 1                         [n! is always positive]
#   E2: factorial(n) == n * factorial(n - 1)      [defining recurrence, n >= 1]
#   E3: factorial(n) >= factorial(n - 1)          [monotone non-decreasing]
import math

n: int = nondet_int()

__ESBMC_assume(n >= 1)
__ESBMC_assume(n <= 8)

f: int = math.factorial(n)
fm: int = math.factorial(n - 1)

assert f >= 1            # E1
assert f == n * fm       # E2
assert f >= fm           # E3
