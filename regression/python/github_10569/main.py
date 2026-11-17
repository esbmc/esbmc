import random


def is_int(x):
    return isinstance(x, int)


def is_str(x):
    return isinstance(x, str)


i = random.randint(0, 9)
for e in [i]:
    assert is_int(e)
    assert not is_str(e)
for e in [i, 3]:
    assert is_int(e)
