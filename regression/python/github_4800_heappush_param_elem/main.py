from heapq import heappush, heappop


def f(start: int, nxt: int) -> int:
    h = []
    heappush(h, (3, start))
    heappush(h, (1, nxt))
    d, n = heappop(h)
    assert d == 1 and n == nxt
    t = heappop(h)
    return t[1]


assert f(7, 8) == 7
