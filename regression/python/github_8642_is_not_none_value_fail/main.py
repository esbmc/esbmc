from typing import Optional


class Node:

    def __init__(self, successor: Optional["Node"] = None):
        self.successor = successor
        self.v = 1


def returned(n: Node) -> bool:
    m = n.successor
    return m is not None and m.v == 1


assert not returned(Node())
assert not returned(Node(Node()))
