import heapq

# heapify's result is read, so the heap keeps the untyped model.
heap = [3, 1, 2]
r = heapq.heapify(heap)
assert heapq.heappop(heap) == 1
