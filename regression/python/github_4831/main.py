class Node:
    def __init__(self, vid):
        self.vid = vid
        self.nxt = None


class Queue:
    def __init__(self, head, count):
        self.head = head
        self.count = count


a = Node(0)
b = Node(1)
a.nxt = b
q = Queue(a, 2)

n = nondet_int()
__ESBMC_assume(0 <= n <= 2)

if n != 0:
    assert q.count >= n
    q.count -= n
    curr = q.head
    for _ in range(n):
        assert curr is not None
        curr = curr.nxt
    q.head = curr

assert q.count == 2 - n
