class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> int:
        self.x += 1
        return self.x

    def dec(self) -> int:
        self.x -= 1
        return self.x


o = C()
if len([0]) == 1:
    f = o.inc
else:
    f = o.dec
assert f() == 1
