import random
import m
import n

x = m if random.random() < 2 else n
assert x.func() == 4
