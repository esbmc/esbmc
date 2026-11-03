import random

n = random.randint(0, 10)
l = [(n, "b")]
x = l[0]
assert not isinstance(x[0], int)
