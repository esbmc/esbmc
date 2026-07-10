# Falsification harness for queue.Queue (src/python-frontend/models/queue.py).
#
# A FIFO returns the FIRST element enqueued, not the last. Asserting LIFO order
# on a Queue must be falsifiable.
#
# REQUIRES:
#   R1: two distinct non-deterministic integers (a != b) so the order matters.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: after put(a), put(b), get() == b.  A FIFO returns a first.
from queue import Queue

a: int = nondet_int()
b: int = nondet_int()
__ESBMC_assume(a != b)

q: Queue = Queue()
q.put(a)
q.put(b)

assert q.get() == b          # F1 — falsifiable (FIFO returns a first)
