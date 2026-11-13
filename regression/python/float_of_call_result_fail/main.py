import random


def pick():
    return random.randint(0, 9)


f = float(pick())
assert f >= 0.0
assert float(random.randint(1, 3)) > 3.0
