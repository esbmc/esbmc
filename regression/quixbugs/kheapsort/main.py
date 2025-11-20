
def kheapsort(arr, k):
    import heapq

    heap = arr[:k]
    heapq.heapify(heap)

    for x in arr[k:]:
        yield heapq.heappushpop(heap, x)

    while heap:
        yield heapq.heappop(heap)


assert list(kheapsort([1, 2, 3, 4, 5], 0)) == [1, 2, 3, 4, 5]
assert list(kheapsort([3, 2, 1, 5, 4], 2)) == [1, 2, 3, 4, 5]
assert list(kheapsort([5, 4, 3, 2, 1], 4)) == [1, 2, 3, 4, 5]
assert list(kheapsort([3, 12, 5, 1, 6], 3)) == [1, 3, 5, 6, 12]
