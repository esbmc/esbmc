import random

y = None
if random.random() < 0.5:
    x = 1
else:
    x = "s"
y = x + x
assert y == 2 or y == "ss"
