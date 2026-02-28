def apply(f, x):
    return f(x)


def square(n):
    return n * n


assert apply(square, 5) == 25
