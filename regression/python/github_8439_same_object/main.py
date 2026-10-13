class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1


o = C()
f = o.inc
o.x = 5
f()
assert o.x == 6
p = o
o = C()
f()
assert p.x == 7
assert o.x == 0
