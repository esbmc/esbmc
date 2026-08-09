# An object popped from a deque comes back with an integer type, so comparing
# it with the object by identity built an equality whose operands differ in
# width -- every solver backend asserted on it. The construct is still
# unsupported; it must report that rather than abort.
from collections import deque as Queue


class Node:
    def __init__(self, v: int) -> None:
        self.v = v


q = Queue()
a = Node(1)
q.append(a)
n = q.popleft()

assert n is a
