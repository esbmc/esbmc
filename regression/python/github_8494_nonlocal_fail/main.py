def outer(s):
    x = s

    def inner():
        nonlocal x
        x += 1
        return x

    return inner()


assert outer(1) == 1
