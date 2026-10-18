# Call sites pass functions returning different types, so the call through
# the parameter is refused (#8351, #8404).
def apply(f, a: int):
    r = f(a)
    return r


def inc(a: int) -> int:
    return a + 1


def show(a: int) -> str:
    return "x"


assert apply(inc, 1) == 2
assert apply(show, 1) == "x"
