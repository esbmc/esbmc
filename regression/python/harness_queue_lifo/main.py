# Verification harness for queue.LifoQueue (src/python-frontend/models/queue.py).
#
# queue.LifoQueue is a list-backed LIFO (stack): put() appends, get() removes
# and returns the most recently added element.
#
# REQUIRES:
#   R1: three fully non-deterministic integers exercise arbitrary payloads.
#
# ENSURES (after put(a), put(b), put(c)):
#   E1: qsize() reports 3                          [enqueue tracking]
#   E2: get() returns c, then b, then a            [last-in-first-out order]
#   E3: the queue is empty afterwards               [every element popped once]
from queue import LifoQueue

a: int = nondet_int()
b: int = nondet_int()
c: int = nondet_int()

q: LifoQueue = LifoQueue()
q.put(a)
q.put(b)
q.put(c)

assert q.qsize() == 3        # E1
assert q.get() == c          # E2
assert q.get() == b          # E2
assert q.get() == a          # E2
assert q.empty()             # E3
