import random
from pkg import sub

if random.random() < 0.5:
    f = sub.g
else:
    f = sub
try:
    r = f()
except TypeError:
    r = 1
assert r == 1
