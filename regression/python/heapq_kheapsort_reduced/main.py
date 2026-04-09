def f(arr, k):
    import heapq

    heap = arr[:k]
    heapq.heapify(heap)

    for x in arr[k:]:
        yield heapq.heappushpop(heap, x)

    while heap:
        yield heapq.heappop(heap)


assert list(f([3, 2, 1], 2)) == [1, 2, 3]
