import random

flag = random.random() < 0.5
x = 1
assert x == (1 if flag else "a")
