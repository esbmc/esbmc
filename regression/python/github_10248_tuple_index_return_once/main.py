# GitHub #10248: a returned tuple subscript evaluates its index once.
def f(i: int) -> int:
    t = (1, 2, 3)
    return t[(i := i + 1)]


assert f(0) == 2
