# A class symbol carries no enclosing-scope component, so two functions each
# defining a class named A would share one symbol and g() would silently run
# f()'s constructor and return 1. Refuse rather than answer wrongly (#7541).
def f() -> int:
    class A:
        def __init__(self) -> None:
            self.x = 1
    return A().x


def g() -> int:
    class A:
        def __init__(self) -> None:
            self.x = 2
    return A().x


assert g() == 2
