def add(x, y):
    return x + y


def square(x):
    return x * x


assert isinstance(square(add(1, 2)), int)
assert isinstance(square(add(0.5, 1.0)), float)
