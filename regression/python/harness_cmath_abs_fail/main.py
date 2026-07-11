# Falsification harness for complex magnitude (abs)
# (src/python-frontend/models/cmath.py).
#
# abs(3 + 4j) is 5, not 4, so the wrong magnitude claim must be falsifiable.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: abs(complex(3, 4)) == 4.0.  The modulus is 5.0.
import cmath

z: complex = complex(3.0, 4.0)

assert abs(z) == 4.0        # F1 — falsifiable (modulus is 5.0)
