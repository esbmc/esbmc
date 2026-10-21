class A:
    def __init__(self, v: int):
        self.a = v


def make(t) -> A:
    return A(t[0])


assert hasattr(make((0, 1)), "a")
assert not hasattr(make((0, 1)), "b")
