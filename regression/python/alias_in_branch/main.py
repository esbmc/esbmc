class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1


def run(c: bool) -> None:
    o = C()
    p = C()
    if c:
        r = o
    else:
        r = p
    r.inc()
    if c:
        assert o.x == 1 and p.x == 0
    else:
        assert o.x == 0 and p.x == 1


run(True)
run(False)
o = C()
if len([0]) == 1:
    f = o.inc
    f()
assert o.x == 1
