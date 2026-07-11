# Verification harness for enum.Enum equality/value
# (src/python-frontend/models/enum.py).
#
# Enum members compare by identity within a class: a member equals itself and
# differs from every other member; .value exposes the assigned integer. This
# harness pins those contracts for a concrete enum. Members are fixed literals,
# so the program is concrete.
#
# ENSURES:
#   E1: a member equals itself (reflexivity), and != is its negation
#   E2: distinct members are not equal
#   E3: __ne__ is exactly the negation of __eq__
#   E4: .value returns the assigned integer
from enum import Enum


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


assert Color.RED == Color.RED                       # E1
assert not (Color.RED != Color.RED)                 # E1
assert Color.RED != Color.GREEN                     # E2
assert not (Color.RED == Color.GREEN)               # E2
assert (Color.RED != Color.BLUE) == (not (Color.RED == Color.BLUE))   # E3

assert Color.RED.value == 1                         # E4
assert Color.GREEN.value == 2                       # E4
assert Color.BLUE.value == 3                        # E4
