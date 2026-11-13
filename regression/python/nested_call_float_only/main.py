def add(x, y):
    return x + y


def square(x):
    return x * x


assert square(add(0.5, 1.0)) == 2.25
assert square(add(1.5, 1.5)) == 9.0
assert isinstance(square(add(0.5, 1.0)), float)
