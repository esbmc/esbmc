import random


def add(x, y):
    return x + y


def square(x):
    return x * x


a = random.randint(0, 9)
assert square(add(a, 2)) == (a + 2) * (a + 2)
f = float(random.randint(0, 9))
assert square(add(f, 0.5)) == (f + 0.5) * (f + 0.5)
