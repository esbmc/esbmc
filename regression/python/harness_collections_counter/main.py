# Verification harness for collections.Counter
# (src/python-frontend/models/collections.py).
#
# The Counter model maps (int, int) keys to integer counts.  __getitem__
# returns 0 for a key that was never written; __setitem__ records the count.
#
# REQUIRES:
#   R1: nondet count values exercise arbitrary stored counts.
#
# ENSURES:
#   E1: an unwritten key reads as 0              [default-count contract]
#   E2: a written key reads back its value       [set/get round-trip]
#   E3: distinct keys are independent            [no aliasing between keys]
from collections import Counter

p: int = nondet_int()
q: int = nondet_int()

c: Counter = Counter()

assert c[(1, 2)] == 0           # E1

c[(1, 2)] = p
assert c[(1, 2)] == p           # E2

c[(3, 4)] = q
assert c[(3, 4)] == q           # E3
assert c[(1, 2)] == p           # E3 (first key unchanged)
assert c[(9, 9)] == 0           # E1 (still-unwritten key)
