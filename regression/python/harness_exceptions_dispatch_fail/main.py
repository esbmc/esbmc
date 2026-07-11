# Falsification harness for exception dispatch
# (src/python-frontend/models/exceptions.py).
#
# The ValueError handler runs on the x > 0 path, not the x <= 0 path, so
# claiming the opposite correlation must be falsifiable.
#
# REQUIRES:
#   R1: a non-deterministic integer selects the raise site.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: (which == 1) == (x <= 0).  The ValueError handler (which == 1) runs
#       iff x > 0, so this is inverted.
x: int = nondet_int()
which: int = 0

try:
    if x > 0:
        raise ValueError("pos")
    else:
        raise TypeError("nonpos")
except ValueError:
    which = 1
except TypeError:
    which = 2

assert (which == 1) == (x <= 0)         # F1 — falsifiable (inverted)
