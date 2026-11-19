from typing import Optional


def rep(second: Optional[int] = None) -> int:
    if second is None:
        return 0
    return second


class Node:

    def __init__(self) -> None:
        self.val: int = 1

    def second(self) -> int:
        return self.val


a = Node()
v = a.second()
assert v == 2
