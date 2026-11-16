import random
e = random.random() > 2
c = None if e else [5]
assert c[0] == 5
