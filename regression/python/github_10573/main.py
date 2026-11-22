import random

if random.random() < 0.5:
    x = 1
else:
    x = "ab"
r = x + x
if not isinstance(x, int):
    assert r.startswith(x)
