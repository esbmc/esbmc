# Verification harness for math.gcd (src/python-frontend/models/math.py).
#
# math.gcd(a, b) computes the greatest common divisor of the absolute
# values of a and b, using the Euclidean algorithm.
#
# REQUIRES (preconditions, established with __ESBMC_assume):
#   R1: a, b are integers taken from a bounded symbolic range so that the
#       Euclidean loop is fully unwound by k-induction.
#   R2: a and b are not both zero (so the gcd is >= 1, the interesting case).
#
# ENSURES (postconditions, checked with assert):
#   E1: the result is positive (>= 1)                     [gcd is non-negative;
#       under R2 it cannot be 0]
#   E2: the result divides a                              [common divisor]
#   E3: the result divides b                              [common divisor]
#   E4: gcd is symmetric: gcd(a, b) == gcd(b, a)          [Euclid is symmetric]
#
# The harness also implicitly checks the model is free of runtime errors
# (no division by zero inside the a % b step, no assertion failures).
import math

a: int = nondet_int()
b: int = nondet_int()

# R1 + R2
__ESBMC_assume(a >= 0)
__ESBMC_assume(a <= 20)
__ESBMC_assume(b >= 0)
__ESBMC_assume(b <= 20)
__ESBMC_assume(a + b > 0)

g: int = math.gcd(a, b)

assert g >= 1  # E1
assert a % g == 0  # E2
assert b % g == 0  # E3
assert math.gcd(b, a) == g  # E4
