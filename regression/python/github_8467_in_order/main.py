class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> int:
        self.x += 1
        return self.x

    def dec(self) -> int:
        self.x -= 1
        return self.x


first = C()
o = first
f = o.inc
o = C()
assert f() == 1
f = o.dec
assert f() == -1
assert first.x == 1
assert o.x == -1
