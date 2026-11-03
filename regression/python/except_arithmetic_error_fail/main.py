# ZeroDivisionError is not a LookupError, so it escapes the handler.
def f(x: int) -> int:
    try:
        return 10 // x
    except LookupError:
        return -1


assert f(0) == -1
