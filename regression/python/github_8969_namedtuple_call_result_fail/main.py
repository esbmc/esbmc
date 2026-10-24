from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])


def make():
    return Point(1, 2)


c = make()
assert c.y == 1
