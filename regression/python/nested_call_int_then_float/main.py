def add(x, y):
    return x + y


def square(x):
    return x * x


assert square(add(1, 2)) == 9
assert square(add(0.5, 1.0)) == 2.25
