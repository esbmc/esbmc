from collections import namedtuple

P = namedtuple("P", ["x", "y"])
Q = namedtuple("Q", "a, b c", defaults=[0, 5])


def norm1(v: int) -> int:
    return abs(v)


def main() -> None:
    p = P(1, 2)
    assert p.x == 1
    assert p[1] == 2
    assert isinstance(p, tuple)
    assert p == (1, 2)
    assert len(p) == 2
    x, y = p
    assert x + y == 3
    q = Q(7, c=3)
    assert q.a == 7 and q.b == 0 and q.c == 3
    r = P(y=4, x=norm1(-9))
    assert r.x == 9 and r.y == 4


main()
