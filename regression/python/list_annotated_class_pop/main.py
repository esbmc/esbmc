# An annotation that disagrees with what is actually stored must not win over
# the recorded element type: the object popped back out is still the object,
# not an int reinterpreted from its pointer.
from typing import List


class Node:
    def __init__(self, v: int) -> None:
        self.v: int = v


a = Node(1)
q: List[int] = []
q.append(a)
b = q.pop()
assert b is a
assert b.v == 1
