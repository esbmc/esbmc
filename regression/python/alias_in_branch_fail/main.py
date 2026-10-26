class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1


o = C()
if len([0]) == 1:
    r = o
    r.inc()
assert o.x == 0
