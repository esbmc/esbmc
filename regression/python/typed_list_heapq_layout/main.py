import heapq

heap = [5, 3, 4, 1, 2]
heapq.heapify(heap)
expected1 = [1, 2, 4, 3, 5]
assert heap == expected1
assert heapq.heappushpop(heap, 0) == 0
expected2 = [1, 2, 4, 3, 5]
assert heap == expected2
assert heapq.heappushpop(heap, 6) == 1
expected3 = [2, 3, 4, 6, 5]
assert heap == expected3
assert heapq.heappop(heap) == 2
expected4 = [3, 5, 4, 6]
assert heap == expected4
