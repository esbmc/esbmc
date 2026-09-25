# Verification harness for math.perm (src/python-frontend/models/math.py).
#
# math.perm(n, k) returns the number of k-permutations of n items,
# P(n, k) = n! / (n-k)!.
#
# REQUIRES:
#   R1: n bounded non-negative integer.
#   R2: k in [0, n].
#
# ENSURES:
#   E1: perm(n, k) >= 1                           [at least the empty/identity
#       arrangement]
#   E2: perm(n, k) == comb(n, k) * factorial(k)   [P(n,k) = C(n,k) * k!]
#   E3: perm(n, n) == factorial(n)                [all items arranged]
import math

n: int = nondet_int()
k: int = nondet_int()

__ESBMC_assume(n >= 0)
__ESBMC_assume(n <= 7)
__ESBMC_assume(k >= 0)
__ESBMC_assume(k <= n)

p: int = math.perm(n, k)

assert p >= 1  # E1
assert p == math.comb(n, k) * math.factorial(k)  # E2
assert math.perm(n, n) == math.factorial(n)  # E3
