import random

a = random.randint(0, 1)
d = {a: "one", a + 1: "two"}
x = 2
del d[x]
assert 2 not in d
