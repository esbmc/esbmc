import random

n = random.randint(0, 10)
d = {"a": (n, "b")}
for k, x in d.items():
    assert isinstance(x[0], int)
y = d["a"]
assert isinstance(y[0], int)
assert not isinstance(y[0], str)
