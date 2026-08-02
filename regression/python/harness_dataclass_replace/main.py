# Verification harness for dataclasses.replace on a @dataclass
# (preprocessor src/python-frontend/preprocessor/dataclass_mixin.py).
#
# replace(obj, **changes) returns a new instance with the named fields
# overridden and the rest copied.
#
# REQUIRES:
#   R1: three nondet integers (two initial fields, one replacement value).
#
# ENSURES (p = Point(a, b); p2 = replace(p, x=c)):
#   E1: the replaced field takes the new value
#   E2: the untouched field is copied from the original
#   E3: the original instance is unchanged
from dataclasses import dataclass, replace


@dataclass
class Point:
    x: int
    y: int


a: int = nondet_int()
b: int = nondet_int()
c: int = nondet_int()

p = Point(a, b)
p2 = replace(p, x=c)

assert p2.x == c  # E1
assert p2.y == b  # E2
assert p.x == a  # E3
assert p.y == b  # E3
