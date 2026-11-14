# Call sites pass functions returning different types through the parameter
# (#8351, #8404). Each copy of `apply` calls its function directly (#10679).
def apply(f, a: int):
    r = f(a)
    return r


def inc(a: int) -> int:
    return a + 1


def show(a: int) -> str:
    return "x"


assert apply(inc, 1) == 2
assert apply(show, 1) == "x"
