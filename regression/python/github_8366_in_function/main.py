from mod import func1, num


def g() -> int:
    f = func1
    n = num
    return f() + n


assert g() == 45
