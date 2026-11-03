from heapq import heappush, heappop

w = 2.5
s = "ab"
g = []
heappush(g, (w, s))
heappush(g, (1.5, "c"))
x, t = heappop(g)
assert x == 1.5 and t == "c"
x, t = heappop(g)
assert x == 2.5 and t == "ab"
