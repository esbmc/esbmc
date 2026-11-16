from heapq import heappush, heappop

a = 4
b = 9
h = []
heappush(h, (5, a))
heappush(h, (3, b))
d, n = heappop(h)
assert n == 4
