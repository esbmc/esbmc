from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int


a = Point(3, 4)
b = Point(3, 4)
c = Point(1, 2)
assert a == b
assert not (a == c)
assert a != c
