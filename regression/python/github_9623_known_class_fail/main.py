# As github_9623_known_class, but the asserted value is false.
class Point:
    def __init__(self, i):
        self.x = i * 1.5

    def mx(self, other):
        if self.x < other.x:
            self.x = other.x
        return self


def mx(a, b):
    return 0


c = nondet_bool()
a = Point(0 if c else 4)
b = Point(2)
r = a.mx(b)
if c:
    assert r.x == 4.0
else:
    assert r.x == 6.0
