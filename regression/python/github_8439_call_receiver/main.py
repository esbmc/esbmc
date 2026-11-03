class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1

    def get(self) -> int:
        return self.x


made = 0


def make() -> C:
    global made
    made += 1
    return C()


f = make().inc
g = make().get
f()
f()
assert made == 2
assert g() == 0
