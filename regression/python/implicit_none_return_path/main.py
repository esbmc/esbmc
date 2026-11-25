import random


# pick falls off its end only when x is in [0, 10).
def pick(x):
    if x < 0:
        return 1
    if x >= 10:
        return 2


x = random.randint(-20, 20)
v = pick(x)
if 0 <= x < 10:
    assert v is None
else:
    assert v == (1 if x < 0 else 2)
