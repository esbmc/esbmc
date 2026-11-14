import random

flag = random.random() < 0.5
ls = [[1], 2] if flag else [[3], 4]
x = ls[0][0]
assert x == 1
