# Control: no module function shares the method's name, so the call resolves as before.
class Point:
    def __init__(self, i):
        self.x = i * 1.5

    def mx(self, other):
        if self.x < other.x:
            self.x = other.x
        return self


def pick(points):
    nxt = points[0]
    return nxt.mx(points[1])


r = pick([Point(0), Point(2)])
assert r.x == 3.0
