class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1


first = C()
o = first
for i in range(3):
    f = o.inc
    o = C()
    f()
    assert o.x == 0
assert first.x == 1
