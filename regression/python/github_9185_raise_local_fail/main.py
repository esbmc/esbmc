# Issue #9185: the DEAD of a raised local ran before the throw copied it.
class E(Exception):
    def __init__(self, v: int):
        self.v = v


def raise_local() -> None:
    e = E(3)
    raise e


try:
    raise_local()
except E as x:
    assert x.v == 3
    assert x.v == 4
