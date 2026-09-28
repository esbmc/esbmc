# Verification harness for random.randint (src/python-frontend/models/random.py).
#
# randint(a, b) returns a non-deterministic integer in the closed range [a, b].
#
# REQUIRES:
#   R1: a <= b (randint requires a non-empty range).
#   R2: a, b bounded so the harness stays finite.
#
# ENSURES (v = randint(a, b)):
#   E1: v >= a                                   [lower bound honoured]
#   E2: v <= b                                   [upper bound honoured]
import random

a: int = nondet_int()
b: int = nondet_int()

__ESBMC_assume(a <= b)
__ESBMC_assume(-1000 <= a <= 1000)
__ESBMC_assume(-1000 <= b <= 1000)

v: int = random.randint(a, b)

assert v >= a       # E1
assert v <= b       # E2
