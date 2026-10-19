from collections import namedtuple

P = namedtuple("P", ["x", "y"])


def f(q):
    return q.x


assert f(P(1, 2)) == 1
