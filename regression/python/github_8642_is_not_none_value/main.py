# A None test on a value read from an attribute guards an attribute read of
# that value in the same expression, wherever the expression is used (#8642).
from typing import Optional


class Node:

    def __init__(self, successor: Optional["Node"] = None):
        self.successor = successor
        self.v = 1


def same(b: bool) -> bool:
    return b


def returned(n: Node) -> bool:
    m = n.successor
    return m is not None and m.v == 1


def conditional(n: Node) -> int:
    m = n.successor
    return m.v if m is not None else 0


def assigned(n: Node) -> bool:
    m = n.successor
    x = m is None or m.v == 1
    return x


def asserted(n: Node) -> None:
    m = n.successor
    assert m is None or m.v == 1


def passed(n: Node) -> bool:
    return same(n.successor is not None and n.successor.v == 1)


for n in [Node(), Node(Node())]:
    assert returned(n) == (n.successor is not None)
    assert conditional(n) == (0 if n.successor is None else 1)
    assert assigned(n)
    asserted(n)
    assert passed(n) == returned(n)
