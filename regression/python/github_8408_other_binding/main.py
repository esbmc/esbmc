class Node:

    def __init__(self, v: int, nxt) -> None:
        self.v = v
        self.nxt = nxt


class Other:
    pass


def last(n: Node) -> int:
    cur = Node(0, n)
    while cur.nxt is not None:
        cur = cur.nxt
    return cur.v


assert last(Node(1, Node(2, None))) == 2
