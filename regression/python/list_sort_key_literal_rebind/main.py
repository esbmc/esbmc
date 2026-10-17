import random

n = random.randint(5, 9)
l = [3, n, 2]
l.sort(key=lambda x: -x)
assert l[0] == n
assert l[2] == 2
