class Foo:
    def __init__(self, x: int) -> None:
        self.x = x

    def baz(self, y: int, *, z: int) -> bool:
        print("baz called with", y, z)
        return self.x + y == z

f = Foo(1)
assert f.baz(2, z=3)
