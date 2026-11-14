import random

flag = random.random() < 0.5
a = [[1], 2]
b = [[3], 4]
ls = a if flag else b
x = ls[0][0]
assert x == 1 or x == 3
