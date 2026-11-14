import heapq

# A bool heap keeps the untyped model: heappop returns a bool.
heap = [True, False, True]
heapq.heapify(heap)
assert heapq.heappop(heap) == False
assert heapq.heappop(heap) == True
