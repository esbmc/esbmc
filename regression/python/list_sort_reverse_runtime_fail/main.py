import random

r = random.randint(0, 1) == 1
k = [3, 1, 2]
k.sort(reverse=r)
assert k[0] == 2
