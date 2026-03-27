import random

a = random.randint(0, 1)
d = {a: "one", a + 1: "two"}
x = a
del d[x]
assert 1 not in d
