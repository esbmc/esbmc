# The functions a variable may hold return different types, so the result
# of calling it has no single type and the call is refused (#8351, #8404).
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
