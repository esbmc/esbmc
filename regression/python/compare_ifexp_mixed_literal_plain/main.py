import random

flag = random.random() < 0.5
x = 1
y = "a"
assert (x == (1 if flag else "a")) == flag
assert ((1 if flag else "a") != y) == flag
