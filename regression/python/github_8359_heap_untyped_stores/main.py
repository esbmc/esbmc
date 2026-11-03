# Lists filled in ways the tuple-retyping pass does not follow keep the
# int heap model.
from heapq import heappush, heappop

a = []
a.append(4)
heappush(a, 2)

b = []
b.extend([5])
heappush(b, 3)

c = []
c.insert(0, 6)
heappush(c, 1)

d = []
d += [7]
heappush(d, 8)

e = []
e = [9]
heappush(e, 0)

f = []
f.append(0)
f[0] = 3
heappush(f, 2)

assert heappop(a) == 2
assert heappop(b) == 3
assert heappop(c) == 1
assert heappop(d) == 7
assert heappop(e) == 0
assert heappop(f) == 2
