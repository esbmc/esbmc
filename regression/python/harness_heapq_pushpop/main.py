# Verification harness for heapq.heappushpop and heapq.heapreplace
# (src/python-frontend/models/heapq.py).
#
# Both operations are size-preserving pop/push combinations:
#   heappushpop(h, item) pushes item then pops and returns the smallest.
#   heapreplace(h, item) pops and returns the smallest, then pushes item.
#
# Concrete inputs are used deliberately: the list model raises a spurious
# IndexError when a symbolic item flows through the append inside heappush,
# and when a grown list is indexed afterwards, so this harness exercises the
# size-preserving pop path (heapreplace's _siftup) without those triggers.
#
# ENSURES:
#   E1: heappushpop with an item below the current minimum returns that item
#       and leaves the size unchanged                 [push-then-pop, item wins]
#   E2: heapreplace returns the previous minimum and leaves the size unchanged
#                                                       [pop-then-push]
import heapq

h: list[int] = [2, 4, 6]
heapq.heapify(h)

# item 1 is smaller than the heap minimum (2), so heappushpop returns it.
r: int = heapq.heappushpop(h, 1)
assert r == 1           # E1
assert len(h) == 3      # E1 (size preserved)

# heapreplace pops the current minimum (2) and pushes 5.
s: int = heapq.heapreplace(h, 5)
assert s == 2           # E2
assert len(h) == 3      # E2 (size preserved)
