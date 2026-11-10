from heapq import heappush

h = []
heappush(h, (5, 2))
heappush(h, (3, 9))
total = 0
for d, n in h:
    total += d * n
assert total == 37
