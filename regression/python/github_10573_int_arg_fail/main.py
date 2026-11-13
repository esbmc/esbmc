import random

if random.random() < 0.5:
    x = 1
else:
    x = "ab"
assert "abc".startswith(x)
