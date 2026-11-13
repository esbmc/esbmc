def square(x):
    return x * x


def pass_on(x):
    return square(x)


assert pass_on(2) == 4
assert pass_on(1.5) == 2.25
