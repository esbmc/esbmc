# GitHub #10027: a star-imported function called through a parameter
import random
from m import *


def call(g):
    return g()


if random.random() < 0.5:
    r = call(f1)
else:
    r = call(f2)
assert r == 42 or r == "s"
