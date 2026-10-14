class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1


o = C()
f = o.inc
for i in range(2):
    o = C()
    f()
assert o.x == 1
