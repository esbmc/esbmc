from typing import Optional


class Node:
    def __init__(self) -> None:
        self.nxt: Optional[Node] = None
        self.val: int = 0


a = Node()
a.nxt = Node()
a.nxt.val = 3
if nondet_bool():
    a.nxt.nxt = Node()
x = a.nxt.val
assert x == 3
try:
    y = a.nxt.nxt.val
except AttributeError:
    y = -1
assert y == 0 or y == -1
