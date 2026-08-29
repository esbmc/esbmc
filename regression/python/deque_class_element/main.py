# The breadth_first_search idiom: objects go through a deque and come back out
# with their identity intact. One syntactic popleft() used to consume two
# element-type entries, so the second conversion fell back to the deque model's
# `list[int]` annotation and the popped object arrived as an int.
from collections import deque


class Node:
    def __init__(self, v: int) -> None:
        self.v: int = v


a = Node(1)
b = Node(2)

q = deque()
q.append(a)
q.append(b)

front = q.popleft()
assert front is a
assert front.v == 1
assert q.popleft() is b
