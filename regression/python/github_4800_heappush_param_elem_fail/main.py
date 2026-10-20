from heapq import heappush, heappop


def f(start: int) -> int:
    h = []
    heappush(h, (0, start))
    d, n = heappop(h)
    return n


assert f(7) == 8
