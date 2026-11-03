from typing import Optional


class Node:

    def __init__(self, successor: Optional["Node"] = None):
        self.successor = successor
        self.v = 1


def g(n: Node) -> int:
    m = n.successor
    k = 0
    if m is not None and m.v == 1:
        k += 1
    if m and m.v == 1:
        k += 1
    while m is not None and m.v > 5:
        k += 1
    return k


assert g(Node()) == 0
