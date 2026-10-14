# Falsification harness for random.randint (src/python-frontend/models/random.py).
#
# randint(a, b) is inclusive of b, so a strict upper bound must be falsifiable.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: randint(a, b) < b.  False whenever the model returns the endpoint b.
import random

a: int = nondet_int()
b: int = nondet_int()

__ESBMC_assume(a <= b)
__ESBMC_assume(-100 <= a <= 100)
__ESBMC_assume(-100 <= b <= 100)

v: int = random.randint(a, b)

assert v < b        # F1 — falsifiable (b is a legal result)
