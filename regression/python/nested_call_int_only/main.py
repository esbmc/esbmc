def add(x, y):
    return x + y


def square(x):
    return x * x


assert square(add(1, 2)) == 9
assert square(add(3, 4)) == 49
assert isinstance(square(add(1, 2)), int)
