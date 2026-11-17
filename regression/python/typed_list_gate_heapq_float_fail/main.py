import heapq

# A float pushed onto the heap keeps it untyped, and the float comes back.
heap = [3, 1, 2]
heapq.heapify(heap)
assert heapq.heappushpop(heap, 0.5) == 1
