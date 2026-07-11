# Verification harness for queue.Queue (src/python-frontend/models/queue.py).
#
# queue.Queue is a list-backed FIFO: put() appends, get() removes and returns
# the oldest element. Under sequential symbolic execution there is no blocking.
#
# REQUIRES:
#   R1: three fully non-deterministic integers exercise arbitrary payloads.
#
# ENSURES (after put(a), put(b), put(c)):
#   E1: qsize() reports 3, and empty() is False    [enqueue tracking]
#   E2: get() returns a, then b, then c            [first-in-first-out order]
#   E3: the queue is empty afterwards               [every element dequeued once]
from queue import Queue

a: int = nondet_int()
b: int = nondet_int()
c: int = nondet_int()

q: Queue = Queue()
q.put(a)
q.put(b)
q.put(c)

assert q.qsize() == 3  # E1
assert not q.empty()  # E1
assert q.get() == a  # E2
assert q.get() == b  # E2
assert q.get() == c  # E2
assert q.empty()  # E3
assert q.qsize() == 0  # E3
