from heapq import heappush, heappop

h = []
heappush(h, (2, 20))
heappush(h, (1, 10))
d, n = heappop(h)
assert d == 2 and n == 20
