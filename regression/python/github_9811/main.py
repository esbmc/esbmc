import random

s = 1 or None
for v in [s]:
    assert v == 1

t = random.randint(0, 1) or None
n = 0
for w in [t, 2]:
    assert w is None or w >= 1
    n += 1
assert n == 2

r = [u for u in [s]]
assert r[0] == 1
