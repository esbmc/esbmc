# 10434: a str-or-None name that holds None is not a str.
import random

s = "a" if random.random() < 0.5 else None
if s is None:
    assert not isinstance(s, str)
else:
    assert isinstance(s, str)
