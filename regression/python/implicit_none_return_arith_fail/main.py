import random


# pick falls off its end for every x in [0, 10), so v + 1 is a TypeError.
def pick(x):
    if x < 0:
        return -1
    if x >= 10:
        return 1


v = pick(random.randint(0, 9))
w = v + 1
