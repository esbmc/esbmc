# Verification harness for random.uniform (src/python-frontend/models/random.py).
#
# uniform(a, b) returns a non-deterministic float in the interval bounded by a
# and b, regardless of which endpoint is larger.
#
# REQUIRES:
#   R1: a, b are finite nondet floats (bounded interval).
#
# ENSURES (v = uniform(a, b)):
#   E1: when a <= b, a <= v <= b                 [ordered endpoints]
#   E2: when a >  b, b <= v <= a                 [reversed endpoints]
import random

a: float = nondet_float()
b: float = nondet_float()

__ESBMC_assume(-1000.0 <= a <= 1000.0)
__ESBMC_assume(-1000.0 <= b <= 1000.0)

v: float = random.uniform(a, b)

if a <= b:
    assert a <= v and v <= b        # E1
else:
    assert b <= v and v <= a        # E2
