import random

n = random.randint(0, 10)
d = {"a": (n, "b")}
y = d["a"]
assert isinstance(y[0], str)
