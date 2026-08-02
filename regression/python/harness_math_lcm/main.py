# Verification harness for math.lcm (src/python-frontend/models/math.py).
#
# math.lcm(a, b) returns the least common multiple, computed as
# (a // gcd(a, b)) * b on absolute values.
#
# REQUIRES:
#   R1: a, b are bounded positive integers (so lcm is >= 1 and the gcd loop
#       is fully unwound).
#
# ENSURES:
#   E1: lcm >= 1                                  [positive for positive inputs]
#   E2: lcm % a == 0                              [common multiple of a]
#   E3: lcm % b == 0                              [common multiple of b]
#   E4: lcm(a, b) * gcd(a, b) == a * b            [fundamental gcd/lcm identity]
import math

a: int = nondet_int()
b: int = nondet_int()

__ESBMC_assume(a >= 1)
__ESBMC_assume(a <= 20)
__ESBMC_assume(b >= 1)
__ESBMC_assume(b <= 20)

l: int = math.lcm(a, b)
g: int = math.gcd(a, b)

assert l >= 1  # E1
assert l % a == 0  # E2
assert l % b == 0  # E3
assert l * g == a * b  # E4
