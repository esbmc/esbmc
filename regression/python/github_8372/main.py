class A:
    def __init__(self, s: str) -> None:
        self.s = s

    def func(self) -> str:
        return self.s


a = A(" Hello ")
assert a.func().upper() == " HELLO "
assert a.func().lower() == " hello "
assert a.func().strip() == "Hello"
assert a.func().startswith(" He")
