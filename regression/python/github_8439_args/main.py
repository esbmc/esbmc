class C:

    def __init__(self) -> None:
        self.x = 0

    def add(self, k: int) -> int:
        self.x += k
        return self.x


first = C()
o = first
f = o.add
o = C()
assert f(3) == 3
assert f(4) == 7
assert first.x == 7
assert o.x == 0
