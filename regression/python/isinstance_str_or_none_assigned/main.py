# 10434: a name assigned a str on one path keeps None on the other.
import random

s = None
if random.random() < 0.5:
    s = "a"
assert isinstance(s, str) == (s is not None)
