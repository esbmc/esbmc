import random


def is_str(x):
    return isinstance(x, str)


i = random.randint(0, 9)
for e in [i, 3]:
    assert is_str(e)
