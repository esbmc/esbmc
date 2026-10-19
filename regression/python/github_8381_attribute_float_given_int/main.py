class C:
    def __init__(self) -> None:
        self.x = 0.5


o = C()
o.x = 2
assert o.x == 2
