class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1


made = 0


def make() -> C:
    global made
    made += 1
    return C()


f = make().inc
f()
f()
assert made == 3
