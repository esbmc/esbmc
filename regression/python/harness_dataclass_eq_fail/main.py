# Falsification harness for @dataclass generated __eq__
# (preprocessor src/python-frontend/preprocessor/dataclass_mixin.py).
#
# The generated __eq__ compares all fields, so instances differing in any field
# are unequal; asserting equality then must be falsifiable.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: Point(a, b) == Point(a, b + 1).  The y fields differ.
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int


a: int = nondet_int()
b: int = nondet_int()

p = Point(a, b)
q = Point(a, b + 1)

assert p == q       # F1 — falsifiable (y fields differ)
