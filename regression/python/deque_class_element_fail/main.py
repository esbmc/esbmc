# The identity is really compared, not assumed: the front of the deque is not
# the element appended last.
from collections import deque


class Node:
    def __init__(self, v: int) -> None:
        self.v: int = v


a = Node(1)
b = Node(2)

q = deque()
q.append(a)
q.append(b)

assert q.popleft() is b
