# Falsification harness for math.remainder (src/python-frontend/models/math.py).
#
# The IEEE remainder is signed: it can be negative when x/y rounds up.  A
# non-negativity claim must therefore be falsifiable.
#
# REQUIRES:
#   R1: x is a finite nondet float (bounded interval).
#
# WRONG PROPERTY (expected to be falsified):
#   F1: remainder(x, 4.0) >= 0.0.  False e.g. for x == 3.0 (nearest multiple
#       of 4 is 4, so remainder is -1.0).
import math

x: float = nondet_float()

__ESBMC_assume(-100.0 <= x <= 100.0)

r: float = math.remainder(x, 4.0)

assert r >= 0.0         # F1 — falsifiable (remainder is signed)
