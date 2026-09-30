import heaplike

heap: list = [(1, 5), (1, 2)]

heaplike.reorder(heap)

first = heap[0]

assert first[1] == 2