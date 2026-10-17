# A conditional expression choosing between functions (#8355).
def f() -> int:
    return 1


def g() -> int:
    return 2


c = nondet_bool()
h = f if c else g
assert h() == 1 or h() == 2
