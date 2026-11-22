# #10309: a call in a default runs when the def runs, even if f is never
# called.
def k(x: int) -> int:
    return 10 // x


def f(y: int = k(0)) -> int:
    return y
