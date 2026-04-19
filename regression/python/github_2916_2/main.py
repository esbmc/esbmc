class Foo:
    def __init__(self, s: str, t: str) -> None:
        self.s = s
        self.t = t

    def bar(self, *, s: str, t: str) -> bool:
        print("bar called with", s, t)
        return s == self.s and t == self.t

f = Foo("hello", "world")
assert f.bar(s="hello", t="world")

