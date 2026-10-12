class A:
    def __init__(self, s: str) -> None:
        self.s = s

    def func(self) -> str:
        return self.s


a = A(" Hello ")
assert a.func().upper() == " Hello "
