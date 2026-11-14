from heapq import heappush, heappop

h = []
heappush(h, (2, 6))
other = []
other.insert(0, 5)
for a, b in h:
    assert a + b == 8
assert other[0] == 5
