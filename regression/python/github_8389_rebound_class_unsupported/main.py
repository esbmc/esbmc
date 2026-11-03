class A:

    def __init__(self) -> None:
        self.x = 1


class B:

    def __init__(self) -> None:
        self.y = 2


def rebind() -> None:
    global A
    A = B


def check(v, h) -> bool:
    rebind()
    return isinstance(v, h)


assert check(A(), A)
