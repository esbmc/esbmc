from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])


def make():
    return Point(1, 2)


def pick(flag):
    if flag:
        return Point(3, 4)
    return Point(x=5, y=6)


c = make()
assert c.x == 1 and c.y == 2
d = pick(nondet_bool())
assert d.x == 3 or d.y == 6
