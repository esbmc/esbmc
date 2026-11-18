import random

n = random.randint(-(2**63), 2**63 - 1)
x = n + n
d = {"k": x}
assert d["k"] != x
