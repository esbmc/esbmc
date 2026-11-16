def add(cls):
    cls.k = 5
    return cls


def bump(c):
    """Adds to k."""
    c.k += 10
    c.j = 1
    c.j += c.k
    return c


@bump
@add
class C:

    def get(self) -> int:
        return self.k


assert C.k == 15
assert C.j == 6
assert C().get() == 15
