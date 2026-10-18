# The module's annotations must survive converting a function in between.
y: int = 1


def f() -> int:
    return 1


assert f() == 1
y = "str"
assert y == "str"
