# Attribute access on an inline (unnamed) instance now works: C().attr used
# to abort with "Unsupported Attribute value type: Call". A named instance
# (c = C(); c.x) already worked; this covers the inline case, including
# dataclasses and use inside a larger expression.
class C:
    def __init__(self, v):
        self.x = v


assert C(5).x == 5
assert C(3).x + C(4).x == 7

from dataclasses import dataclass


@dataclass
class P:
    x: int
    y: int


assert P(1, 2).x == 1
assert P(1, 2).y == 2
