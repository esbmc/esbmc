# Verification harness for random.random (src/python-frontend/models/random.py).
#
# random() returns a non-deterministic float in the half-open interval [0, 1).
#
# REQUIRES:
#   (none) — random() takes no arguments.
#
# ENSURES (r = random()):
#   E1: r >= 0.0                                 [lower bound is inclusive]
#   E2: r < 1.0                                  [upper bound is exclusive]
import random

r: float = random.random()

assert r >= 0.0     # E1
assert r < 1.0      # E2
