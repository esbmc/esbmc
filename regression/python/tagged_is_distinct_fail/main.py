import random

if random.random() < 0.5:
    x, y = 1, 2
else:
    x, y = "a", "b"
assert x is y
