import random

n = random.randint(-(2**63), 2**63 - 1)
x = n + n
d = {"k": x}
assert d["k"] == x
e = {0: x, 1: "s"}
assert e[0] == x
