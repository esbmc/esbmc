# 10434: a str-or-None argument to an unannotated parameter.
import random


def func(x):
    if isinstance(x, str):
        return 1
    return 0


s = "a" if random.random() < 0.5 else None
if s is None:
    assert func(s) == 0
else:
    assert func(s) == 1
