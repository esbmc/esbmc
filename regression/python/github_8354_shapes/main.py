class C:
    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1

    def get(self) -> int:
        return self.x


o = C()
(read, bump) = (o.get, o.inc)
[step, n] = [o.inc, 3]
bump()
step()
assert read() == 2
assert o.x == 2
assert n == 3
