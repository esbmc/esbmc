# Negative variant: queue.Queue is FIFO, so the first item out is the first
# put in (1), not the last (3). The wrong assertion below must be reported as
# a violation — guarding against a LIFO/nondet mis-model of get().
import queue

q = queue.Queue()
q.put(1)
q.put(2)
q.put(3)
assert q.get() == 3
