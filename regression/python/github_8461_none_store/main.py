class Node:
    def __init__(self, v: int) -> None:
        self.v = v
        self.next = None


a = Node(1)
b = Node(2)
a.next = b
assert a.next.v == 2
a.next = None
assert a.next is None
