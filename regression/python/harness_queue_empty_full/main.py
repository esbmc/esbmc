# Verification harness for Queue.empty / full / qsize
# (src/python-frontend/models/queue.py).
#
# empty() is True iff qsize() == 0; full() is True iff maxsize > 0 and
# qsize() >= maxsize. The model tracks maxsize but put() does not block on it.
#
# REQUIRES:
#   R1: two non-deterministic integer payloads; a bounded maxsize of 2.
#
# ENSURES:
#   E1: a fresh bounded queue is empty and not full
#   E2: below capacity: not empty, not full, qsize tracks the count
#   E3: at capacity: full() becomes True exactly when qsize == maxsize
#   E4: after a get(), it is neither full nor empty again
from queue import Queue

a: int = nondet_int()
b: int = nondet_int()

q: Queue = Queue(2)  # maxsize == 2

assert q.empty()  # E1
assert not q.full()  # E1

q.put(a)
assert not q.empty()  # E2
assert q.qsize() == 1  # E2
assert not q.full()  # E2

q.put(b)
assert q.full()  # E3 (qsize 2 >= maxsize 2)
assert q.qsize() == 2  # E3

x: int = q.get()
assert not q.full()  # E4
assert not q.empty()  # E4
