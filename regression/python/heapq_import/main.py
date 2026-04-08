import heapq


heap = [3, 1]
heapq.heapify(heap)
heapq.heappush(heap, 2)

assert heapq.heappop(heap) == 1
assert heapq.heappushpop(heap, 0) == 0
assert heapq.heappop(heap) == 2
