from typing import Optional


class Node:
    def __init__(self) -> None:
        self.nxt: Optional[Node] = None
        self.val: int = 0

    def second(self) -> int:
        return self.nxt.val


a = Node()
if nondet_bool():
    a.nxt = Node()
v = a.second()
