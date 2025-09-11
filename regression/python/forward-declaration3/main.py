def outer() -> int:

    def inner() -> int:
        return helper()

    return inner()


def helper() -> int:
    return 123


assert outer() == 123
