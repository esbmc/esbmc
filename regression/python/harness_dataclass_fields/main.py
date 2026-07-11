# Verification harness for @dataclass synthesis
# (preprocessor src/python-frontend/preprocessor/dataclass_mixin.py; the
# src/python-frontend/models/dataclasses.py stubs are overridden by it).
#
# @dataclass synthesizes __init__ (positional field assignment) and __eq__
# (field-wise comparison). This harness verifies both over nondet inputs.
#
# REQUIRES:
#   R1: two nondet integers seed the fields, exploring all field values.
#
# ENSURES (for Point(a, b)):
#   E1: positional arguments are stored in the declared fields
#   E2: the generated __eq__ makes two instances equal iff all fields match
#   E3: differing a field makes them unequal
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int


a: int = nondet_int()
b: int = nondet_int()

p = Point(a, b)
assert p.x == a         # E1
assert p.y == b         # E1

q = Point(a, b)
assert p == q           # E2

r = Point(a + 1, b)
assert p != r           # E3
