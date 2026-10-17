import heapq


def main() -> None:
    h = []
    heapq.heappush(h, (5, 1))
    heapq.heappush(h, (3, 2))
    heapq.heappush(h, (4, 3))
    a = heapq.heappop(h)
    b = heapq.heappop(h)
    assert a == (3, 2) and b == (4, 3)
    assert b[1] == 3


main()
