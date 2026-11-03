class E(Exception):
    def __init__(self, v: int) -> None:
        self.v = v


def f(n: int) -> None:
    e = E(n)
    raise e


x = nondet_int()
try:
    f(x)
except E as err:
    assert err.v != x
