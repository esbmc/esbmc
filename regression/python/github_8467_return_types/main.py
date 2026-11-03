class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> int:
        self.x += 1
        return self.x

    def name(self) -> str:
        return "c"


for c in [True, False]:
    o = C()
    if c:
        f = o.inc
    else:
        f = o.name
    y = f()
    if c:
        assert y == 1
    else:
        assert y == "c"
