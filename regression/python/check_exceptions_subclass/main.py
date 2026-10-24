# A subclass of IndexError is not a KeyError, so with only KeyError checked
# its escape ends the path unreported.
class MyError(IndexError):
    pass


def main() -> None:
    b = nondet_int()
    if b == 3:
        raise MyError("b is 3")


main()
