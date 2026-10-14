# The heapq shape: the tuple element type must reach reorder() across the module
# boundary and then reach _swap(), whose argument is itself a parameter. Tuples
# compare lexicographically, so (1, 2) sorts before (1, 5).
import heaplike

heap: list = [(1, 5), (1, 2)]
heaplike.reorder(heap)
first = heap[0]
assert first[1] == 2
