import math
from random import random

try:
    math()
    assert False
except TypeError:
    pass
assert random() < 1.0
