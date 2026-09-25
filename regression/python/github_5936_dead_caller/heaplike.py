def _swap(heap: list[int], i: int, j: int) -> None:
    tmp = heap[i]
    heap[i] = heap[j]
    heap[j] = tmp


def reorder(heap: list[int]) -> None:
    if heap[1] < heap[0]:
        _swap(heap, 0, 1)


# Never called. Its `heap` parameter therefore has no caller to type it, and the
# _swap call below must not be counted: doing so would conflict away the tuple
# element type reorder() supplies and leave _swap mis-typed. This is exactly the
# shape of heapq, where heappop/heapreplace/heappushpop all call _siftup.
def rotate(heap: list[int]) -> None:
    _swap(heap, 0, 1)
