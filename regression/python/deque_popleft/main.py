# collections.deque is modelled as a list backing store, so its front-end
# operations map to list ops: popleft() == pop(0), appendleft(x) == insert(0, x).
# Also exercises the aliased import `from collections import deque as Queue`
# (the breadth_first_search idiom), which must bind the alias to the deque model.
from collections import deque
from collections import deque as Queue

# popleft / appendleft preserve FIFO order
d = deque()
d.append(2)
d.append(3)
d.appendleft(1)
assert d.popleft() == 1
assert d.popleft() == 2
assert d.popleft() == 3
assert len(d) == 0

# FIFO drain via the aliased import
q = Queue()
q.append(10)
q.append(20)
total = 0
while q:
    total += q.popleft()
assert total == 30
