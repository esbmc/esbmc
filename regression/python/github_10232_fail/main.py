import random
from pkg import sub

if random.random() < 2:
    f = sub
else:
    f = sub.g
assert f() == 1
