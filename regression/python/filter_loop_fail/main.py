def pos(n: int) -> bool:
    return n > 0

xs = [-1, 2, -3, 4]
s = 0
for x in filter(pos, xs):
    s = s + x
assert s == 5
