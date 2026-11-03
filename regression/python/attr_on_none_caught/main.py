from typing import Optional


class Node:

    def __init__(self, successor: Optional["Node"] = None):
        self.successor = successor
        self.v = 1


def g(n: Node) -> int:
    m = n.successor
    try:
        k = m.v
    except AttributeError:
        k = 0
    return k


assert g(Node()) == 0
