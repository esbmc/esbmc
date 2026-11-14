import heapq

heap = [1]
assert heapq.heappop(heap) == 1
assert heapq.heappushpop(heap, 7) == 7
try:
    heapq.heappop(heap)
    assert False
except IndexError:
    pass
