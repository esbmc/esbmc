import random

flag = random.random() < 0.5
ls = [[1], 2] if flag else [[3], 4]
x = ls[0][0]
assert x == 1 or x == 3

a = [[5], 6]
b = [[7], 8]
ms = a if flag else b
y = ms[0][0]
assert y == 5 or y == 7

ns = a if flag else a
z = ns[0][0]
assert z == 5
