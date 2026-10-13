class C:
    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1


o = C()
f, n = o.inc, 0
o = C()
f()
assert o.x == 1
