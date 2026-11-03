class A:
    def __init__(self, s: str) -> None:
        self.s = s


a = A("Hello")
assert a.s.replace("H", "J") == ""
