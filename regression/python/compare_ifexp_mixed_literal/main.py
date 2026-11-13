import random

flag = random.random() < 0.5
r = 1 if flag else "a"
assert r == (1 if flag else "a")
