import heapq

h = [(5, 1), (3, 2)]
heapq.heapify(h)
a = heapq.heappop(h)
assert a != (3, 2)
