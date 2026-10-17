# The functions return different types, so the choice is refused (#8355,
# #8404).
def f() -> int:
    return 1


def g() -> str:
    return "a"


h = f if nondet_bool() else g
assert h() == 1 or h() == "a"
