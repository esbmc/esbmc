import random

r = random.randint(0, 1) == 1
l = [3, 1, 2]
s = sorted(l, key=lambda x: -x, reverse=r)
assert s[0] == 2
