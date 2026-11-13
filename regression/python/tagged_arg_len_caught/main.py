import random

use_int = random.random() < 0.5
if use_int:
    r = 12
else:
    r = "ab"
try:
    n = len(r)
except TypeError:
    n = -1
if use_int:
    assert n == -1
else:
    assert n == 2
