# GitHub #10248: an out-of-range tuple index escapes as IndexError.
def f(i: int) -> int:
    t = (1, 2, 3)
    return t[i]


f(3)
