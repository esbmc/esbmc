# Falsification harness for float.is_integer
# (src/python-frontend/models/float.py).
#
# Not every float is integral, so asserting that a nondet float has no
# fractional part must be falsifiable.
#
# REQUIRES:
#   R1: a nondet float x bounded to a finite interval.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: x.is_integer() for all x.  False e.g. for x == 0.5.
x: float = nondet_float()
__ESBMC_assume(-1000.0 <= x <= 1000.0)

assert x.is_integer()  # F1 — falsifiable (non-integral floats exist)
