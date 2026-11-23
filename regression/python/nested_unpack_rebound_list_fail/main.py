# GitHub #10600
import random

pairs = [(1, 2)]
if random.random() < 0.5:
    pairs = [(2, 1)]
(a, b), = pairs
assert a == 1
