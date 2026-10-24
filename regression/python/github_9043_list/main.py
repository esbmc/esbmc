import random

n = random.randint(0, 10)
l = [(n, "b")]
x = l[0]
assert isinstance(x[0], int)
assert not isinstance(x[0], str)
assert isinstance(x[1], str)
assert not isinstance(x[1], int)
