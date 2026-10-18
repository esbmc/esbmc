class Node:

    def __init__(self, successor=None):
        self.successor = successor


def f(n: Node) -> int:
    m = n.successor
    if m.successor is None:
        return 1
    return 0


assert f(Node()) == 0
