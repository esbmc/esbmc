# Verification harness for max/min (src/python-frontend/models/builtins.py).
#
# max(l) / min(l) return the largest / smallest element of a non-empty list.
#
# REQUIRES:
#   R1: the list holds three fully non-deterministic integers, so every
#       ordering of the elements is explored.
#
# ENSURES (for m = max(l), n = min(l)):
#   E1: m is an upper bound: m >= every element        [max dominates]
#   E2: m is attained: m equals some element           [max is a member]
#   E3: n is a lower bound: n <= every element          [min is dominated]
#   E4: n is attained: n equals some element           [min is a member]
#   E5: n <= m                                          [min never exceeds max]
a: int = nondet_int()
b: int = nondet_int()
c: int = nondet_int()
l: list[int] = [a, b, c]

m: int = max(l)
n: int = min(l)

assert m >= a and m >= b and m >= c  # E1
assert m == a or m == b or m == c  # E2
assert n <= a and n <= b and n <= c  # E3
assert n == a or n == b or n == c  # E4
assert n <= m  # E5
