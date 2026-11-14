# 10434: a str or None returned by path.
import random


def pick(flag):
    if flag:
        return "a"
    return None


s = pick(random.random() < 0.5)
assert isinstance(s, str) == (s is not None)
