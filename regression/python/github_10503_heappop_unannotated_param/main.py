from heapq import heappush, heappop
def g(h) -> int:
    d, n = heappop(h)
    return n
h: list[tuple[int, int]] = []
heappush(h, (2, 5))
heappush(h, (1, 7))
assert g(h) == 7
