import heapq
import time
from heapq import heappush

heap = [3, 1, 2]
r = heapq.heapify(heap)
assert r is None
assert heap[0] == 1
p = heappush(heap, 0)
assert p is None
assert heap[0] == 0
assert time.sleep(0) is None
