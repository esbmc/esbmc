import random

n = random.randint(-(10**18), -1)
if random.random() < 0.5:
    a = "x"
else:
    a = n * n
if isinstance(a, int):
    assert a < 0
else:
    assert a == "y"
