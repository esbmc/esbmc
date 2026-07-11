# Verification harness for collections.deque
# (src/python-frontend/models/collections.py).
#
# deque(iterable) is modelled as a list backing store exposing append / pop /
# len / subscript.  This harness checks the list-copy construction and append.
#
# REQUIRES:
#   R1: three fully non-deterministic integers seed the deque.
#
# ENSURES:
#   E1: len(deque([a, b, c])) == 3               [construction copies all items]
#   E2: element order is preserved               [d[i] matches the source]
#   E3: append grows the deque by one at the tail
from collections import deque

a: int = nondet_int()
b: int = nondet_int()
c: int = nondet_int()

d: list[int] = deque([a, b, c])

assert len(d) == 3                              # E1
assert d[0] == a and d[1] == b and d[2] == c    # E2

d.append(99)
assert len(d) == 4                              # E3
assert d[3] == 99                               # E3
