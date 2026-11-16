import random
import m
import n

if random.random() < 2:
    x = n
else:
    x = m
assert x.add(1) == 2
