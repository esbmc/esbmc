from heapq import heappush, heappop


def f(w: float, v: float) -> float:
    h = []
    heappush(h, (3, w))
    heappush(h, (1, v))
    d, n = heappop(h)
    return n


assert f(2.5, 0.5) == 2.5
