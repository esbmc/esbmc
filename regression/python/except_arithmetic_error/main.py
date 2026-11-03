# ZeroDivisionError is an ArithmeticError, so the handler catches it.
def f(x: int) -> int:
    try:
        return 10 // x
    except ArithmeticError:
        return -1


assert f(0) == -1
