# Verification harness for random.randrange
# (src/python-frontend/models/random.py).
#
# randrange(start, stop, step) returns a non-deterministic value of the form
# start + i*step that lies in [start, stop).
#
# REQUIRES:
#   (none) — concrete arguments exercise the stepped-range arithmetic.
#
# ENSURES (v = randrange(3, 20, 4)):
#   E1: 3 <= v < 20                              [result lies in the range]
#   E2: (v - 3) % 4 == 0                         [result is on the step grid]
#   E3: v in {3, 7, 11, 15, 19}                  [exact set of legal values]
import random

v: int = random.randrange(3, 20, 4)

assert 3 <= v            # E1 lower
assert v < 20            # E1 upper
assert (v - 3) % 4 == 0  # E2
assert v == 3 or v == 7 or v == 11 or v == 15 or v == 19   # E3
