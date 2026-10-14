# heapq must be reachable (issue #5931) and must agree with CPython's
# Lib/heapq.py on the array layout, not merely on "some minimum".
import heapq

# heapify establishes the invariant: the root is the minimum.
items: list[int] = [3, 1, 2]
heapq.heapify(items)
assert items[0] == 1

# heappop returns the root and preserves the invariant.
assert heapq.heappop(items) == 1
assert items[0] == 2

# heappush keeps the smallest at the root.
heapq.heappush(items, 0)
assert items[0] == 0

# heappop on a list that is NOT a valid heap returns heap[0], not the
# minimum. A model that scanned for the minimum would answer 1 here.
raw: list[int] = [3, 1, 2]
assert heapq.heappop(raw) == 3

# heapreplace pops the root and pushes item; size is unchanged.
sized: list[int] = [1, 4, 7]
assert heapq.heapreplace(sized, 5) == 1
assert len(sized) == 3
assert sized[0] == 4

# heappushpop returns item itself when item is no larger than the root.
assert heapq.heappushpop(sized, 2) == 2

# ...and swaps in item, returning the old root, when item is larger.
assert heapq.heappushpop(sized, 9) == 4
assert sized[0] == 5
