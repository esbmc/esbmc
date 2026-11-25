import random

x = float(random.randint(0, 9))
s = str(x)
assert len(s) == 3 and s[-2:] == ".0"
