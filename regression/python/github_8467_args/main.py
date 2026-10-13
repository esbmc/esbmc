class C:

    def __init__(self) -> None:
        self.x = 0
        self.n = 0

    def next(self) -> int:
        self.n += 1
        return self.n

    def add(self, a: int, b: int) -> None:
        self.x += 10 * a + b

    def sub(self, a: int, b: int) -> None:
        self.x -= 10 * a + b


for c in [True, False]:
    k = C()
    o = C()
    if c:
        f = o.add
    else:
        f = o.sub
    f(k.next(), b=k.next())
    assert k.n == 2
    if c:
        assert o.x == 12
    else:
        assert o.x == -12
