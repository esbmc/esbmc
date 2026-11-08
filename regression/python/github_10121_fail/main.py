# #10121: a whole-string "Optional[...]" annotation is the Optional it spells.
from typing import Optional


class Node:
    def __init__(self, nxt: "Optional[Node]") -> None:
        self.nxt = nxt


n = Node(None)
m = Node(n)
assert m.nxt is None
