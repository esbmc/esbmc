import heapq


def main() -> None:
    h = []
    heapq.heappush(h, (5, 1))
    heapq.heappush(h, (3, 2))
    a = heapq.heappop(h)
    assert a != (3, 2)


main()
