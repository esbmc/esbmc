class A:
    def __init__(self, v: int):
        self.a = v


def make(t) -> A:
    assert t[0] == 1
    return A(t[0])


assert hasattr(make((0, 1)), "a")
