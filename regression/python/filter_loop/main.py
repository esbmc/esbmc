def pos(n: int) -> bool:
    return n > 0

xs = [-1, 2, -3, 4]
s = 0
for x in filter(pos, xs):
    s = s + x
assert s == 6

ys = [0, 1, 0, 2]
t = 0
for y in filter(None, ys):
    t = t + y
assert t == 3
