# The functions a variable may hold return different types. The call is
# dispatched per function, so each path sees its own result type (#8970).
def f1() -> int:
    return 1


def f2() -> str:
    return "a"


if nondet_bool():
    g = f1
else:
    g = f2
r = g()
assert r == 1 or r == "a"
