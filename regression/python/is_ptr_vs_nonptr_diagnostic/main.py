# An object popped from a deque used to come back with an integer type, so
# comparing it with the object by identity built an equality whose operands
# differed in width; the frontend reported that rather than aborting. The
# popped element now keeps its type, so the identity holds as CPython reports.
from collections import deque as Queue


class Node:
    def __init__(self, v: int) -> None:
        self.v = v


q = Queue()
a = Node(1)
q.append(a)
n = q.popleft()

assert n is a
