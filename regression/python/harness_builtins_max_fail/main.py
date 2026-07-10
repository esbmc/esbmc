# Falsification harness for max (src/python-frontend/models/builtins.py).
#
# Validates that a violated postcondition is detected: max is only >= each
# element, never strictly greater than all of them (it equals the maximum).
#
# REQUIRES:
#   R1: the list holds three non-deterministic integers.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: max(l) > l[0].  False whenever l[0] is the maximum (e.g. a >= b, c).
a: int = nondet_int()
b: int = nondet_int()
c: int = nondet_int()
l: list[int] = [a, b, c]

m: int = max(l)

assert m > a            # F1 — falsifiable (a may be the maximum)
