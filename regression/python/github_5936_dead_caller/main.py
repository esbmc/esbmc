# A helper called both from a live function (with tuples) and from a function
# that is never called must still be typed from the live call site (#5936).
import heaplike

heap: list = [(1, 5), (1, 2)]
heaplike.reorder(heap)
first = heap[0]
assert first[1] == 2
