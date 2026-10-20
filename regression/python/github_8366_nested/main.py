from mod import func1


def outer() -> int:

    def inner() -> int:
        f = func1
        return f()

    return inner()


assert outer() == 42
