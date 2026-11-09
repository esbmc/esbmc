# GitHub #10027: the star import rebinds f1, so f1 is m.f1, not this def
import random


def f1():
    return 7


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
