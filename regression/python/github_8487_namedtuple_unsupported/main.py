from collections import namedtuple

P = namedtuple("P", ["x", "y"])
p = P(1, 2)
assert isinstance(p, P)
