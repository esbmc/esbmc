import random


def pick(flag):
    if flag:
        return 5


v = pick(random.random() < 0.5)
assert v is None or v == 5
