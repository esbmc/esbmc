class C:
    def __init__(self) -> None:
        self.x = 0


o = C()
o.x = 1.5
assert o.x == 1.5
