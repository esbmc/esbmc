# queue.Queue is modelled as a list-backed FIFO and queue.LifoQueue as a
# list-backed stack. Exercises the qualified `queue.Queue()` form (whose
# .get() must dispatch to the class method, not dict.get), the direct
# `from queue import LifoQueue` form, and maxsize/full tracking.
import queue
from queue import LifoQueue

# FIFO: items come out in the order they went in
q = queue.Queue()
q.put(1)
q.put(2)
q.put(3)
assert q.qsize() == 3
assert not q.empty()
assert q.get() == 1
assert q.get() == 2
assert q.get() == 3
assert q.empty()

# bounded queue: full() tracks maxsize
b = queue.Queue(2)
b.put(10)
assert not b.full()
b.put(20)
assert b.full()

# LifoQueue is a stack: last in, first out
s = LifoQueue()
s.put(7)
s.put(8)
assert s.get() == 8
assert s.get() == 7

# A genuine dict's .get()/.pop() must still route to the dict handler after the
# receiver-type dispatch guard (regression guard for the queue.Queue.get fix).
d = {1: 11, 2: 22}
assert d.get(1) == 11
assert d.get(3, -1) == -1
assert d.pop(2) == 22
