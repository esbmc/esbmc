# 10434: a str-or-None name that holds None is not a str.
import random

s = "a" if random.random() < 0.0 else None
assert isinstance(s, str)
