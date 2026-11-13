# 10434: a str or None returned by path.
import random


def pick(flag):
    if flag:
        return "a"
    return None


s = pick(random.random() < 0.0)
assert isinstance(s, str)
