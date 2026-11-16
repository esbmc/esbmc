import random
def f(n):
    return abs(n)
r = f(random.uniform(-1.0, 1.0))
assert r >= 0.0
assert isinstance(r, float)
