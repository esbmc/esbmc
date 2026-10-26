class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1

    def dec(self) -> None:
        self.x -= 1


o = C()
p = C()
for i in range(2):
    if i == 1:
        f()
    if len([0]) == 1:
        f = o.inc
    else:
        f = p.dec
