# Issue #9076: a bytes literal must not size a return type that another
# return of unknown length shares; that would prove len(b) == 2 below.
def f(flag: bool, x: bytes) -> bytes:
    if flag:
        return b"ab"
    return x


c = nondet_bool()
b = f(c, b"xyz")
assert len(b) == 2
