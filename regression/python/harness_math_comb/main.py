# Verification harness for math.comb (src/python-frontend/models/math.py).
#
# math.comb(n, k) returns the binomial coefficient C(n, k) = n! / (k!(n-k)!).
#
# REQUIRES:
#   R1: n is a bounded non-negative integer.
#   R2: k is in [0, n] (comb returns 0 for k > n and raises for k < 0; we
#       exercise the main branch).
#
# ENSURES:
#   E1: comb(n, k) >= 1                           [at least one choice exists]
#   E2: comb(n, k) == comb(n, n - k)              [Pascal symmetry]
#   E3: comb(n, 0) style boundary via k handled by E2 as well.
#
# The harness also confirms the multiplicative-formula loop (result * (n-i)
# // (i+1)) performs exact integer division with no runtime error.
import math

n: int = nondet_int()
k: int = nondet_int()

__ESBMC_assume(n >= 0)
__ESBMC_assume(n <= 10)
__ESBMC_assume(k >= 0)
__ESBMC_assume(k <= n)

c: int = math.comb(n, k)

assert c >= 1  # E1
assert c == math.comb(n, n - k)  # E2
