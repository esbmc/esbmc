def twice(f):
    def wrapper(a, b):
        r = f(a, b)
        return r + r

    return wrapper


def swap(f):
    """Pass the arguments in the other order."""
    def inner(a, b):
        return f(b, a)

    return inner


@twice
@swap
def join(a, b):
    return a + b


c = join("x", "y")
assert c == "yxyx"
assert len(c) == 4
