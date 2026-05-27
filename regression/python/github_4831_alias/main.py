class Node:
    def __init__(self, vid):
        self.vid = vid
        self.nxt = None


class Queue:
    def __init__(self, head):
        self.head = head


a = Node(0)
q = Queue(a)

# Python reference semantics: `a` and `q.head` are aliases for the same
# Node, so a mutation through `a` must be visible via `q.head`.
a.vid = 42
assert q.head.vid == 42
