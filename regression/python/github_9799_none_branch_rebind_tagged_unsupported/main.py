import random

y = None
if random.random() < 0.5:
    x = 1
else:
    x = "s"
if random.random() < 0.5:
    y = x + x
assert y is None or y == 2 or y == "ss"
