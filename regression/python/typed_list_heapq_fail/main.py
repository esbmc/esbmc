import heapq


def main() -> None:
    arr = [nondet_int(), nondet_int(), nondet_int(), nondet_int()]
    heap = arr[:]
    heapq.heapify(heap)
    out = []
    while heap:
        out.append(heapq.heappop(heap))
    assert out == arr


main()
