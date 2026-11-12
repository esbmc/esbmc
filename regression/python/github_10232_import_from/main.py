import random
from pkg import sub
from pkg.sub import g

if random.random() < 0.5:
    f = g
else:
    f = sub
try:
    r = f()
except TypeError:
    r = 1
assert r == 1
