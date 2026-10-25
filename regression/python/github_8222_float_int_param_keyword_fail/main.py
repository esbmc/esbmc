class A:

    def m(self, y: int) -> int:
        return y


def half() -> float:
    return 2.5


a = A()
assert a.m(y=half()) == 2
