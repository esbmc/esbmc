class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1

    def dec(self) -> None:
        self.x -= 1


first = C()
o = first
p = C()
for i in range(3):
    if i == 1:
        f = p.dec
    else:
        f = o.inc
    o = C()
    f()
assert p.x == 0
