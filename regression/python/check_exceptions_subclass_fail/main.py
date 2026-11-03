# A subclass of a checked type is checked, as `except IndexError` would
# catch it.
class MyError(IndexError):
    pass


def main() -> None:
    b = nondet_int()
    if b == 3:
        raise MyError("b is 3")


main()
