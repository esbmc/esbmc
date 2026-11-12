import random
from pkg import sub
from pkg.sub import g

if random.random() < 2:
    f = sub
else:
    f = g
assert f() == 1
