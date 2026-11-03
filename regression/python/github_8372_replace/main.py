class A:
    def __init__(self, s: str) -> None:
        self.s = s

    def func(self) -> str:
        return self.s


def swap(s: str, old: str) -> str:
    return "Hello".replace(old, s)


a = A("Hello")
assert a.func().replace("H", "J") == "Jello"
assert a.s.replace("H", "J") == "Jello"
assert swap("J", "H") == "Jello"
