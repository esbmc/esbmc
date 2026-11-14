def add(x, y):
    return x + y


def square(x):
    return x * x


def cube(x):
    return square(x) * x


assert cube(add(1, 2)) == 27
assert cube(add(0.5, 1.0)) == 3.375
