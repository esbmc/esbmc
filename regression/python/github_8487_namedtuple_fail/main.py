from collections import namedtuple

P = namedtuple("P", ["x", "y"])


def main() -> None:
    p = P(y=1, x=2)
    assert p.y == 1
    assert p.x == 1


main()
