import heapq

heap = [5, 3, 4, 1, 2]
heapq.heapify(heap)
assert heapq.heappushpop(heap, 6) == 1
expected1 = [2, 3, 4, 5, 6]
assert heap == expected1
