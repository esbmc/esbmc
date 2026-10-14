def _swap(heap: list[int], i: int, j: int) -> None:
    tmp = heap[i]
    heap[i] = heap[j]
    heap[j] = tmp


def reorder(heap: list[int]) -> None:
    if heap[1] < heap[0]:
        _swap(heap, 0, 1)
