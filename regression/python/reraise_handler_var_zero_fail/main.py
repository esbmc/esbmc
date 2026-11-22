def f(a: int) -> int:
    try:
        return 10 // a
    except ArithmeticError as err:
        raise err


f(0)
