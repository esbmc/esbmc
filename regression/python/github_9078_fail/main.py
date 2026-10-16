# github #9078: a.nxt may be None, so a.nxt.val raises AttributeError
from typing import Optional


class Node:
    def __init__(self) -> None:
        self.nxt: Optional[Node] = None
        self.val: int = 0


a = Node()
if nondet_bool():
    a.nxt = Node()
x = a.nxt.val
