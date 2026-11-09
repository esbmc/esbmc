# GitHub #10027: a name bound to one of two star-imported functions
import random
from m import *

if random.random() < 0.5:
    f = f1
else:
    f = f2
r = f()
if f is f1:
    assert r == 42
else:
    assert r == "s"
