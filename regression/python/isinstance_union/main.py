class A:

    def __init__(self):
        self.a = 1


class B:

    def __init__(self):
        self.b = 2.0


assert isinstance(1, int | str)
assert not isinstance(1.5, int | str)
assert isinstance("a", int | str)
assert isinstance(1.5, int | str | float)
assert isinstance(1.5, (int, str | float))
assert isinstance(None, int | None)
assert not isinstance(1, str | None)
assert isinstance(B(), A | B)
assert not isinstance(A(), int | B)
x = int
assert isinstance(x, str | type)
assert not isinstance(x, int | str)
