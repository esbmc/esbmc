from heapq import heappush, heappop

h: list[tuple[int, int]] = []
heappush(h, (5, 4))
d, n = heappop(h)
assert d == 5 and n == 4


def run(k: int) -> int:
    q: list[tuple[int, int]] = []
    heappush(q, (k, 3))
    _, m = heappop(q)
    return m


assert run(4) == 4
