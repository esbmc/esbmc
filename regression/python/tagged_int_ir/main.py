import random

n = random.randint(-(2**63), 2**63 - 1)
if random.random() < 0.5:
    a = "Hello world"
else:
    a = n + 1
if not isinstance(a, str):
    assert a == n + 1
