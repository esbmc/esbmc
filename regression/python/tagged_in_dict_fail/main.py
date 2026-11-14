import random

d = {1: "x", "1": "y"}
n = random.randint(0, 2)
k = n if random.random() < 0.5 else str(n)
if k in d:
    assert n == 2
