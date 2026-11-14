import random

x = float(random.randint(0, 9))
s = "%f" % x
assert s[-7:] == ".000001"
