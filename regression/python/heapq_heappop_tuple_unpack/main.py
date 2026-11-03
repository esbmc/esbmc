# heappop on a tuple heap returns a tuple that can be unpacked (#4799).
import heapq
from heapq import heappop, heappush

h = [(3, 7), (1, 9)]
heapq.heapify(h)
heappush(h, (2, 8))
d, n = heappop(h)
assert d == 1 and n == 9
d, n = heapq.heappop(h)
assert d == 2 and n == 8
