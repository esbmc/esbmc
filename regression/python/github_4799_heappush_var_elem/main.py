from heapq import heappush, heappop

a = 4
b = 9
h = []
heappush(h, (5, a))
heappush(h, (3, b))
d, n = heappop(h)
assert d == 3 and n == 9
d, n = heappop(h)
assert d == 5 and n == 4


def run(k: int) -> int:
    q = []
    node = k * 2
    heappush(q, (k, node))
    heappush(q, (1, 0))
    heappop(q)
    _, m = heappop(q)
    return m


assert run(4) == 8
