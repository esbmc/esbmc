# Verification harness for random.getrandbits
# (src/python-frontend/models/random.py).
#
# getrandbits(k) returns a non-deterministic integer using exactly k random
# bits, i.e. a value in [0, 2**k - 1]; k == 0 yields 0.
#
# REQUIRES:
#   R1: k is a bounded non-negative integer (getrandbits raises for k < 0).
#
# ENSURES (v = getrandbits(k)):
#   E1: v >= 0                                   [k random bits are unsigned]
#   E2: v <= (1 << k) - 1                        [at most k bits set]
import random

k: int = nondet_int()

__ESBMC_assume(0 <= k <= 16)

v: int = random.getrandbits(k)

assert v >= 0                   # E1
assert v <= (1 << k) - 1        # E2
