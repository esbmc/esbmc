class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1

    def dec(self) -> None:
        self.x -= 1


def call() -> None:
    f()


o = C()
if len([0]) == 1:
    f = o.inc
else:
    f = o.dec
call()
assert o.x == 1
