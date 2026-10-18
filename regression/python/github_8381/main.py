class C:
    def __init__(self) -> None:
        self.b = 10


o = C()
o.b = "x"
assert o.b == "x"
