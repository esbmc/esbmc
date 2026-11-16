# #10121: a whole-string "Optional[...]" annotation is the Optional it spells.
from typing import Literal, Optional


class Node:
    def __init__(self, nxt: "Optional[Node]") -> None:
        self.nxt = nxt


def key(k: Literal["a[0]"]) -> int:
    return 1


n = Node(None)
assert n.nxt is None
m = Node(n)
assert m.nxt is n
assert key("a[0]") == 1
