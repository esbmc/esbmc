import random

r = random.randint(0, 1) == 1
l = [3, 1, 2]
s = sorted(l, key=lambda x: -x, reverse=r)
if r:
    assert s == [1, 2, 3]
else:
    assert s == [3, 2, 1]
t = sorted(l, key=lambda x: -x, reverse=True)
assert t == [1, 2, 3]
p = [(1, "a"), (0, "b"), (1, "c")]
q = sorted(p, key=lambda e: e[0], reverse=True)
assert q[0][1] == "a"
assert q[1][1] == "c"
