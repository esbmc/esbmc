class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1

    def dec(self) -> None:
        self.x -= 1


o = C()
p = C()
c = len([0]) == 1
if c:
    f = o.inc
else:
    f = p.dec
f()
assert o.x == 1
assert p.x == 0
