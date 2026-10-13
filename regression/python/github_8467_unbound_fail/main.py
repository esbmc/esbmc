class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1

    def dec(self) -> None:
        self.x -= 1


o = C()
p = C()
n = len([0])
if n == 2:
    f = o.inc
elif n == 3:
    f = p.dec
f()
