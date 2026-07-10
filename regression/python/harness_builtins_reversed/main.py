# Verification harness for reversed (src/python-frontend/models/builtins.py).
#
# reversed(l) returns a new list with the elements of l in reverse order.
#
# REQUIRES:
#   R1: three fully non-deterministic integers.
#
# ENSURES (for r = reversed(l)):
#   E1: len(r) == len(l)                        [length preserved]
#   E2: r == [c, b, a]                           [order is reversed]
#   E3: reversed(reversed(l)) restores l         [reversal is an involution]
a: int = nondet_int()
b: int = nondet_int()
c: int = nondet_int()
l: list[int] = [a, b, c]

r: list[int] = reversed(l)

assert len(r) == 3  # E1
assert r[0] == c and r[1] == b and r[2] == a  # E2

rr: list[int] = reversed(r)
assert rr[0] == a and rr[1] == b and rr[2] == c  # E3
