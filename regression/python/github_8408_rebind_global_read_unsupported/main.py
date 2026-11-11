class A:

    def __init__(self) -> None:
        self.x = 1


class C:

    def __init__(self) -> None:
        self.y = 2


def get() -> int:
    return a.y


a = A()
a = C()
assert get() == 2
