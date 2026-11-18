import random

n = random.randint(-(2**63), 2**63 - 1)
x = n + n
a = [x, "s"]
assert a[0] == x
b = [1.5]
b.append(x)
assert b[1] == x
