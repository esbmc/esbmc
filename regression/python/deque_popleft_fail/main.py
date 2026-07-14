# Negative variant: deque.popleft() removes from the FRONT, so the first
# element out is the first appended (1), not the last (2). The wrong assertion
# below must be reported as a violation — guarding against a popleft that
# mis-maps to pop() (back) or returns a nondet element.
from collections import deque

d = deque()
d.append(1)
d.append(2)
assert d.popleft() == 2
