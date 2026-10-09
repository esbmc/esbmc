# str % with non-constant arguments lowers like the matching f-string; %d
# renders a bool through int() as CPython does (#8224).
def f(i):
    return "id %d" % i


def g(name: str, n: int, x: float) -> str:
    return "%s:%i %s %%" % (name, n, x)


def h(b: bool, n: int) -> str:
    return "%d/%d" % (b, n)


assert f(3) == "id 3"
assert g("ab", 4, 2.5) == "ab:4 2.5 %"
assert h(True, -3) == "True/-3"
